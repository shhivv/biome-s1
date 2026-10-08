"""UI-level session: the policy acts on FreeCAD's interface.

`UiSession` keeps `GuiSession`'s interface (`reset/state/valid_actions/
expert/step/progress`), so the worker protocol, the GUI server and the data
generator drive it unchanged, but its actions are UI element ids
(`ui/spec.py`):

- `cmd:<command>` for a command with a task dialog triggers the command's
  real QAction. FreeCAD creates the feature and opens its task panel.
- `set:` / `opt:` / `toggle:` edit the task panel's widgets (found by
  objectName in the live Qt tree), and `click:OK` / `click:Cancel` press its
  buttons. OK commits FreeCAD's own transaction; Cancel aborts it.
- Commands without a dialog (Body, New Sketch, Leave Sketch, Undo, ...),
  workbench switches and the canvas interactions (selection in the 3D view,
  sketch geometry and constraints) go through the command-level executors,
  exactly as `GuiSession` does.

What the policy sees is the same structured `State` as before, plus
`State.ui` (the open dialog's fields and their current values) and the list
of elements it can act on. The teacher (`ui/teacher.py`) labels every state,
including half-filled dialogs and dialogs opened by mistake.

A modal message box that FreeCAD raises while the policy acts (e.g. a
feature that fails on OK) would block the event loop; `ModalGuard` closes it
and records its text.

Must run inside the FreeCAD GUI (it can be hidden: `open -g -j` on macOS).
"""

from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import dataclass, field

import FreeCAD as App
import FreeCADGui as Gui
from PySide import QtCore, QtWidgets

from ..actions import CATALOGUE, PROFILE_FEATURES, SOLID_FEATURE_TYPES
from ..expert import Meta, ObjMeta, progress
from ..goals import StartSpec
from ..runtime.gui_session import GuiSession
from ..runtime.session import ActionError, UndoEntry
from ..schema import Goal, State
from . import spec as S
from .teacher import command_elements, dialog_elements, dialog_expert, fields_on_plan

# FreeCAD keeps 20 undo steps by default. The GUI reads this preference when
# it creates a document; with the history out of reach, transaction counts
# stay exact (at the cap, a GUI-side extra transaction cannot be told apart).
GUI_UNDO_LIMIT = 2000
QUIET_CHECKBOXES = ("checkBoxUpdateView", "showTransparentPreviewCheckBox")

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
MAX_COUNT = 100  # pattern occurrences above this are refused outright


def pump(n: int = 5) -> None:
    for _ in range(n):
        QtWidgets.QApplication.processEvents()


def wait_until(cond, timeout: float = 3.0) -> bool:
    """Process GUI events until `cond()` holds. FreeCAD closes task dialogs
    and leaves edit mode asynchronously."""
    deadline = time.time() + timeout
    while not cond():
        if time.time() > deadline:
            return False
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 10)
    return True


class ModalGuard:
    """Closes modal dialogs that open while the block runs (they would block
    the event loop) and records their text in `sink`."""

    def __init__(self, sink: list[str]) -> None:
        self.sink = sink
        self.timer = QtCore.QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.close_modal)

    def close_modal(self) -> None:
        w = QtWidgets.QApplication.activeModalWidget()
        if w is None:
            return
        text = w.text() if hasattr(w, "text") else w.windowTitle()
        self.sink.append(str(text)[:200])
        try:
            w.reject()
        except Exception:
            w.close()

    def __enter__(self) -> "ModalGuard":
        self.timer.start()
        return self

    def __exit__(self, *exc) -> None:
        pump()
        self.close_modal()
        self.timer.stop()


@dataclass
class Pending:
    """A task dialog opened by a `cmd:` element and not yet closed."""

    command: str
    on_plan: bool  # the command was the expert's choice when it was opened
    goal_ref: int
    targets: dict[str, object]
    before: Meta
    sel_before: list
    undo_before: int
    profile: str | None = None  # the sketch a profile feature must use
    feature: str | None = None  # the object FreeCAD created for the dialog
    messages: list[str] = field(default_factory=list)


class UiSession(GuiSession):
    undo_limit = GUI_UNDO_LIMIT

    def __init__(self) -> None:
        App.ParamGet("User parameter:BaseApp/Preferences/Document").SetInt("MaxUndoSize", GUI_UNDO_LIMIT)
        super().__init__()
        # Experimental (S1_QUIET_DIALOGS=1): switch off a task dialog's live
        # preview / recompute-on-change when it opens; OK still computes the
        # feature. Not exposed to the model.
        self.quiet_dialogs = os.environ.get("S1_QUIET_DIALOGS") == "1"
        self.pending: Pending | None = None
        self.messages: list[str] = []
        self._external = None  # (generator, info) of an action being performed from outside
        self.pick_fallback = False  # a wrong outside pick: book the intended selection anyway (reported)
        # FreeCAD offers Document Recovery after a previous instance died (e.g. killed by the watchdog).
        # The modal dialog blocks anything acting from outside the app; decline it whenever it shows.
        self.key_log: list[str] = []  # last key events FreeCAD received (debugging outside actions)
        if os.environ.get("S1_KEY_LOG") == "1":
            self._key_filter = _KeyLogger(self.key_log)
            QtWidgets.QApplication.instance().installEventFilter(self._key_filter)
        self._recovery_timer = QtCore.QTimer()
        self._recovery_timer.timeout.connect(self._decline_recovery)
        self._recovery_timer.start(400)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._close_dialog()
        super().close()

    def reset(self, goal: Goal, start: StartSpec | None = None) -> None:
        self._close_dialog()
        super().reset(goal, start)
        self.settle()

    def _gui_in_edit(self) -> str | None:
        gdoc = Gui.ActiveDocument
        vp = gdoc.getInEdit() if gdoc is not None else None
        return vp.Object.Name if vp is not None else None

    def edit_object(self) -> str | None:
        """Only a sketch counts as edit mode. A feature's task dialog also
        puts that feature "in edit" in the GUI; for the state that is a
        dialog (`State.ui`), not edit mode."""
        name = self._gui_in_edit()
        obj = self.doc.getObject(name) if (name and self.doc is not None) else None
        return name if obj is not None and obj.TypeId == "Sketcher::SketchObject" else None

    def _sync_gui(self) -> None:
        """Guard rail: never re-enter edit on anything but a sketch, and never
        touch the GUI's edit state while a dialog is open. Re-entering edit on
        a feature reopens its task dialog, and repeating that is how a run once
        leaked memory until the machine ran out."""
        if self.meta.edit is not None:
            obj = self.doc.getObject(self.meta.edit) if self.doc is not None else None
            if obj is None or obj.TypeId != "Sketcher::SketchObject":
                self.meta.edit = None
        if self.pending is not None:
            return
        super()._sync_gui()

    def settle(self) -> bool:
        """Wait until the GUI's edit state matches the session's (the sketch
        being edited, the open dialog's feature, or nothing)."""
        if self.doc is None:
            return True
        want = self.pending.feature if self.pending is not None else self.meta.edit
        ok = wait_until(lambda: self._gui_in_edit() == want
                        and (want is not None or not Gui.Control.activeDialog()))
        self.refresh_commands()
        pump(2)
        return ok

    def refresh_commands(self) -> None:
        """Bring toolbar buttons' enabled states up to date now. FreeCAD only
        refreshes them on a timer, so anything reading the interface from
        outside (the OS accessibility tree) would otherwise see stale states."""
        update = getattr(Gui, "updateCommands", None)
        if update is not None:
            update()
        # FreeCAD's own refresh may not run for a background window: set each catalogue command's
        # actions from isCommandActive, as FreeCAD's timer would
        for name in CATALOGUE:
            cmd = name.split(":", 1)[0]
            try:
                actions = Gui.Command.get(cmd).getAction()
            except Exception:
                continue
            active = Gui.isCommandActive(cmd)
            for a in actions or []:
                a.setEnabled(active)

    def _absorb_gui_transactions(self) -> None:
        """The GUI sometimes commits a transaction of its own after an action
        (leaving a sketch it re-entered after an Undo commits "Sketch
        recompute"). Fold such extras into the latest undo entry so one Undo
        still reverts exactly one step."""
        if self.doc is None or not self.undo_stack:
            return
        tx = sum(e.n_tx for e in self.undo_stack if e.doc_tx)
        extra = self.doc.UndoCount - tx
        if extra <= 0:
            return
        if self.doc.UndoCount >= self.undo_limit:
            raise RuntimeError(f"undo history reached {self.undo_limit}; transaction counts are no longer exact")
        top = self.undo_stack[-1]
        top.n_tx = (top.n_tx if top.doc_tx else 0) + extra
        top.doc_tx = True

    def _close_dialog(self) -> None:
        if Gui.Control.activeDialog():
            button = self._button("Cancel")
            with ModalGuard(self.messages):
                if button is not None:
                    button.click()
                else:
                    Gui.Control.closeDialog()
        self.pending = None

    # -- widget tree -----------------------------------------------------------

    def task_view(self):
        mw = Gui.getMainWindow()
        for w in mw.findChildren(QtWidgets.QWidget):
            if w.metaObject().className() == "Gui::TaskView::TaskView":
                return w
        return None

    def _widget(self, name: str):
        root = self.task_view()
        if root is None:
            return None
        for w in root.findChildren(QtWidgets.QWidget, name):
            if w.isVisibleTo(root):  # isVisible() is False in a hidden app
                return w
        return None

    def _button(self, text: str):
        root = self.task_view()
        if root is None:
            return None
        for box in root.findChildren(QtWidgets.QDialogButtonBox):
            for b in box.buttons():
                if b.text().replace("&", "") == text and b.isVisibleTo(root):
                    return b
        return None

    def ui_fields(self) -> dict[str, dict]:
        """Exposed fields of the open dialog with their current values."""
        out: dict[str, dict] = {}
        if self.pending is None:
            return out
        for fl in S.DIALOG_FIELDS.get(self.pending.command, []):
            w = self._widget(fl.name)
            if w is None:
                continue
            d: dict = {"kind": fl.kind, "enabled": bool(w.isEnabled())}
            if fl.kind == "number":
                d["value"] = self._read_number(w)
                d["unit"] = fl.unit
                d["target"] = self.pending.targets.get(fl.name)  # what the parameter stage would type (None: any)
            elif fl.kind == "choice":
                d["value"] = w.currentText()
                d["options"] = [w.itemText(i) for i in range(w.count())]
            else:
                d["value"] = bool(w.isChecked())
            out[fl.name] = d
        return out

    # -- public API ------------------------------------------------------------

    def state(self) -> State:
        st = super().state()
        if self.pending is not None:
            st.ui = {"dialog": self.pending.command, "feature": self.pending.feature, "fields": self.ui_fields()}
        return st

    def valid_actions(self, state: State | None = None) -> list[str]:
        if self.pending is not None:
            return dialog_elements({"fields": self.ui_fields()})
        return command_elements(super().valid_actions(state or self.state()))

    def ui_command_expert(self) -> list[str]:
        """The command-level expert, adjusted for GUI behaviour it does not
        model: the GUI's Pad/Pocket/... take a selected face as their profile,
        so the selection must be empty before them (the headless executors
        always use the open sketch)."""
        acts = self.command_expert()
        if self.sel_refs and any(a in PROFILE_FEATURES for a in acts):
            return ["Select:Clear"]
        return acts

    def expert(self) -> list[str]:
        if self.pending is not None:
            return dialog_expert({"fields": self.ui_fields()}, self.pending.targets, self.pending.on_plan)
        return command_elements(self.ui_command_expert())

    def step(self, element: str) -> dict:
        info = {"changed": False, "error": None, "done": False, "on_plan": element in self.expert()}
        role = S.role(element)
        command = S.underlying(element)
        dialog_widget = role in ("set", "opt", "toggle", "click")
        if self.pending is not None and not dialog_widget:
            info["error"] = "a task dialog is open"
            return info
        if not dialog_widget and command is not None and not (role == "cmd" and command in S.DIALOG_COMMANDS):
            out = super().step(command)
            if self.recent and self.recent[-1] == command:
                self.recent[-1] = element
            out["on_plan"] = info["on_plan"]
            self.settle()
            self._absorb_gui_transactions()
            return out
        self.recent.append(element)
        try:
            gen = self._dialog_action(element, info)
            request = next(gen)  # bookkeeping before the physical action
            self._actuate(request)  # the action itself, through Qt
            self._finish(gen)  # bookkeeping after it
        except ActionError as exc:
            info["error"] = str(exc)
        self.settle()
        return info

    # -- actions performed from outside FreeCAD (scripts/mesa_ax.py) ----------------
    #
    # A dialog action is a generator: it does its "before" bookkeeping, yields one
    # request describing the physical action, then does its "after" bookkeeping.
    # step() performs the request through Qt; begin_external()/end_external() let
    # a driver outside the app perform it instead (e.g. through the OS
    # accessibility tree) while the session keeps its undo / dialog / plan state.

    def _dialog_action(self, element: str, info: dict):
        role = S.role(element)
        if role == "cmd":
            return self._open_dialog(S.underlying(element), info)
        if self.pending is None:
            raise ActionError(f"{element}: no task dialog is open")
        if role in ("set", "opt", "toggle"):
            return self._edit_field(element)
        if element == S.OK:
            return self._accept(info)
        if element == S.CANCEL:
            return self._reject(info)
        raise ActionError(f"unknown element {element}")

    @staticmethod
    def _finish(gen) -> None:
        try:
            gen.send(None)
        except StopIteration:
            return
        raise RuntimeError("a dialog action yielded more than once")

    def _actuate(self, req: dict) -> None:
        kind = req["kind"]
        if kind == "command":
            req["_action"].trigger()
        elif kind == "number":
            self._set_number(self._widget(req["field"]), req["value"], req["unit"])
        elif kind == "choice":
            w = self._widget(req["field"])
            w.setCurrentIndex(w.findText(req["option"]))
        elif kind == "toggle":
            self._widget(req["field"]).click()
        elif kind == "button":
            self._button(req["button"]).click()
        else:
            raise ActionError(f"unknown request {kind}")

    def _decline_recovery(self) -> None:
        for w in QtWidgets.QApplication.topLevelWidgets():
            if w.isVisible() and "DocumentRecovery" in w.metaObject().className():
                self.messages.append("declined Document Recovery")
                w.reject()

    def diagnostics(self) -> dict:
        """What the session can see of the task panel right now (debugging external actions)."""
        tv = self.task_view()
        chain, p = [], tv
        while p is not None:
            chain.append(f"{p.metaObject().className()}#{p.objectName()} visible={p.isVisible()} hidden={p.isHidden()}")
            p = p.parentWidget()
        modal = QtWidgets.QApplication.activeModalWidget()
        return {"task_view": tv is not None, "dialog_active": bool(Gui.Control.activeDialog()),
                "pending": None if self.pending is None else self.pending.command,
                "ok_button": self._button("OK") is not None, "chain": chain[:8],
                "modal": None if modal is None else f"{modal.metaObject().className()} {modal.windowTitle()}",
                "messages": self.messages[-3:], "features": self._feature_summary(), "widgets": self._widget_summary(), "keys": self.key_log[-12:],
                "selection": [f"{o.ObjectName}:{','.join(o.SubElementNames)}" for o in Gui.Selection.getSelectionEx()],
                "preselection": str(getattr(Gui.Selection.getPreselection(), "SubElementNames", ""))}

    def _widget_summary(self) -> dict:
        out = {}
        for name in ("lengthEdit", "lengthEdit2", "chamferSize", "spinOccurrences"):
            w = self._widget(name)
            if w is not None:
                raw = w.property("rawValue")
                out[name] = {"text": w.text(), "raw": raw if isinstance(raw, (int, float)) else str(raw),
                             "focus": w.hasFocus()}
        return out

    def _feature_summary(self) -> list:
        doc = App.ActiveDocument
        props = ("Type", "Length", "Length2", "Occurrences", "Angle", "Size", "Reversed", "Midplane", "Constraints")
        out = []
        for o in doc.Objects if doc is not None else []:
            if o.TypeId.startswith("PartDesign::") and o.TypeId != "PartDesign::Body":
                vals = {}
                for k in props:
                    if hasattr(o, k):
                        v = getattr(o, k)
                        vals[k] = round(v.Value, 4) if hasattr(v, "Value") else (str(v) if not isinstance(v, (int, float, bool)) else v)
                out.append({"name": o.Name, "valid": o.isValid(), **{k: v for k, v in vals.items() if k != "Constraints"}})
        return out

    def begin_external(self, element: str, inside: tuple = (), canvas: bool = False) -> dict:
        """Prepare a dialog action to be performed from outside the app. Returns
        the request ({"kind": "command" | "number" | "choice" | "toggle" |
        "button", ...}); {"internal": True} for elements the session performs
        itself (workbench, canvas, commands without a dialog, Done). Requests
        whose kind is in `inside` are performed here too ({"inside": True, ...});
        end_external() still finishes and verifies them."""
        role = S.role(element)
        if canvas and role == "canvas" and S.underlying(element).startswith("Select:") and self.pending is None:
            return self._begin_pick(element)
        if role not in ("set", "opt", "toggle", "click") and not (role == "cmd" and S.underlying(element) in S.DIALOG_COMMANDS):
            return {"internal": True}
        info = {"changed": False, "error": None, "done": False, "on_plan": element in self.expert()}
        self.recent.append(element)
        try:
            gen = self._dialog_action(element, info)
            request = next(gen)
        except ActionError as exc:
            info["error"] = str(exc)
            self.settle()
            return {"error": str(exc), "info": info}
        self._external = (gen, info)
        if request["kind"] in inside:
            self._actuate(request)
            return {"inside": True, "kind": request["kind"]}
        return {k: v for k, v in request.items() if not k.startswith("_")}

    def _begin_pick(self, element: str) -> dict:
        """A selection made in the 3D view or the model tree by whoever acts from outside
        (with the real mouse). The session only says what to pick and checks the result."""
        arg = S.underlying(element).split(":", 1)[1]
        info = {"changed": False, "error": None, "done": False, "on_plan": element in self.expert()}
        expected = None if arg == "Clear" else self.resolve_selection(arg)
        if arg != "Clear" and expected is None:
            info["error"] = f"nothing to select for {arg}"
            return {"error": info["error"], "info": info}
        self.recent.append(element)
        if arg != "Clear":
            Gui.Selection.clearSelection()
            pump(2)

        def finish():
            yield
            got = self.gui_selection()
            want = [] if expected is None else [(expected[0], sorted(expected[1]))]
            if got != want:
                info["error"] = f"{element}: picked {got}, expected {want}"
                info["picked"] = got
                if not self.pick_fallback:
                    self.sel_refs, self.meta.selection = [], []
                    self._sync_gui()
                    return
                info["corrected"] = True
            self.sel_refs, self.meta.selection = ([], []) if expected is None else ([expected], [arg])
            self._sync_gui()

        gen = finish()
        next(gen)
        self._external = (gen, info)
        return {"kind": "pick", "target": arg}

    def gui_selection(self) -> list:
        """The GUI's selection as [(object, sorted sub-elements)], sub-elements of the
        Body's visible feature attributed to that feature."""
        out = {}
        for sx in Gui.Selection.getSelectionEx(self.doc.Name if self.doc is not None else ""):
            obj, subs = sx.ObjectName, list(sx.SubElementNames)
            if self.body is not None and obj == self.body.Name and subs and "." in subs[0]:
                obj = subs[0].split(".")[0]
                subs = [x.split(".", 1)[1] for x in subs]
            out.setdefault(obj, set()).update(subs)
        return sorted((k, sorted(v)) for k, v in out.items())

    def end_external(self) -> dict:
        gen, info = self._external
        self._external = None
        try:
            self._finish(gen)
        except ActionError as exc:
            info["error"] = str(exc)
        self.settle()
        return info

    # -- dialogs ---------------------------------------------------------------

    def _open_dialog(self, command: str, info: dict) -> None:
        if not Gui.isCommandActive(command):
            raise ActionError(f"{command} is not active")
        on_plan = command in self.ui_command_expert()
        prof = self.open_profile() if command in PROFILE_FEATURES else None
        goal_ref = prof.goal_ref if prof is not None else progress(self.goal, self.meta)
        f = self.intent_for(goal_ref)
        names_before = {o.Name for o in self.doc.Objects}
        pending = Pending(command, on_plan, goal_ref, S.targets(command, f, self.goal.scale),
                          copy.deepcopy(self.meta), list(self.sel_refs), self.doc.UndoCount,
                          profile=prof.name if prof is not None else None)
        self.settle()
        action = Gui.Command.get(command).getAction()[0]  # the toolbar button's QAction
        wait_until(action.isEnabled, timeout=2.0)  # FreeCAD refreshes enabled states on a timer
        with ModalGuard(pending.messages):
            yield {"kind": "command", "command": command, "_action": action}
            wait_until(lambda: bool(Gui.Control.activeDialog()), timeout=3.0)
        if not Gui.Control.activeDialog():
            created = [o.Name for o in self.doc.Objects if o.Name not in names_before]
            for _ in range(self.doc.UndoCount - pending.undo_before):
                self.doc.undo()
            raise ActionError(f"{command} opened no task dialog (created {created}; {pending.messages})")
        pending.feature = next((o.Name for o in self.doc.Objects
                                if o.Name not in names_before and o.TypeId in SOLID_FEATURE_TYPES), None)
        self.pending = pending
        if self.quiet_dialogs:
            for name in QUIET_CHECKBOXES:
                w = self._widget(name)
                if w is not None and w.isEnabled() and w.isChecked():
                    with ModalGuard(pending.messages):
                        w.click()
        # Fail loudly on a dialog that is not the one the spec describes (e.g.
        # Pad's sketch picker when two unused sketches exist): a field that is
        # silently missing would let OK commit FreeCAD's default.
        missing = [fl.name for fl in S.DIALOG_FIELDS.get(command, []) if self._widget(fl.name) is None]
        if missing or pending.feature is None:
            try:
                gen = self._reject({})
                self._actuate(next(gen))
                self._finish(gen)
            except ActionError:
                self._close_dialog()
            raise ActionError(f"{command} opened an unexpected dialog (feature {pending.feature}, missing {missing})")
        info["changed"] = True

    def _edit_field(self, element: str) -> None:
        p = self.pending
        name = S.field_name(element)
        fl = next((x for x in S.DIALOG_FIELDS.get(p.command, []) if x.name == name), None)
        w = self._widget(name)
        if fl is None or w is None or not w.isEnabled():
            raise ActionError(f"{element}: no such enabled field")
        with ModalGuard(p.messages):
            if fl.kind == "number":
                value = p.targets.get(name)
                if value is None:
                    value = S.fallback_number(fl.unit, self.goal.scale)
                value = float(value)
                if fl.unit == "count" and not 1 <= value <= MAX_COUNT:
                    raise ActionError(f"refusing count {value} (allowed 1..{MAX_COUNT})")
                text = str(int(round(value))) if fl.unit == "count" else f"{value:g}"
                yield {"kind": "number", "field": name, "value": value, "unit": fl.unit, "text": text}
                pump(3)
                w = self._widget(name)
                got = self._read_number(w)
                if got is None or not S.value_matches(round(value) if fl.unit == "count" else value, got):
                    raise ActionError(f"{element}: the field shows {got!r}, not {text}")
                raw = w.property("rawValue")  # Gui::QuantitySpinBox: the value it would apply
                if isinstance(raw, float) and fl.unit != "count" and not S.value_matches(value, raw):
                    raise ActionError(f"{element}: the field shows {got!r} but holds {raw!r}")
            elif fl.kind == "choice":
                option = S.option_text(element)
                options = [w.itemText(i) for i in range(w.count())]
                if option not in options:
                    raise ActionError(f"{element}: no such option")
                yield {"kind": "choice", "field": name, "option": option, "options": options,
                       "current": w.currentText()}
                wait_until(lambda: (self._widget(name) is not None
                                    and self._widget(name).currentText() == option), timeout=2.0)
                now = self._widget(name)
                if now is None or now.currentText() != option:
                    raise ActionError(f"{element}: the dropdown shows {None if now is None else now.currentText()!r}")
            else:
                yield {"kind": "toggle", "field": name}
                pump(3)

    @staticmethod
    def _read_number(w) -> float | None:
        m = _NUMBER.search(w.text())
        return float(m.group(0).replace(",", ".")) if m else None

    def _set_number(self, w, value: float, unit: str) -> None:
        """Type `value` into a spin box like a user would and check what the
        widget made of it. Never call setValue on FreeCAD's spin boxes:
        Gui::UIntSpinBox stores unsigned counts shifted into Qt's signed range
        (shown 2 = stored -2147483646), so setValue(6) means ~2.1 billion
        occurrences, which the live preview then tries to build."""
        if unit == "count" and not 1 <= value <= MAX_COUNT:
            raise ActionError(f"refusing count {value} (allowed 1..{MAX_COUNT})")
        old = w.text()
        text = str(int(round(value))) if unit == "count" else f"{value:g}"
        if unit == "mm":
            text += " mm"
        elif unit == "deg":
            text += " °"
        line = w.lineEdit()
        line.setText(text)
        w.interpretText()  # parses the text like Enter would, without triggering the dialog's OK
        if hasattr(w, "editingFinished"):
            w.editingFinished.emit()
        got = self._read_number(w)
        if got is None or not S.value_matches(round(value) if unit == "count" else value, got):
            line.setText(old)
            w.interpretText()
            raise ActionError(f"typed {text!r} but the field shows {w.text()!r}")

    def _accept(self, info: dict) -> None:
        p = self.pending
        fields = self.ui_fields()
        for name, fld in fields.items():  # guard rail: never commit an absurd count
            if fld.get("unit") == "count" and (fld.get("value") is None or fld["value"] > MAX_COUNT):
                raise ActionError(f"refusing OK: {name} = {fld.get('value')}")
        fields_ok = fields_on_plan({"fields": fields}, p.targets)
        button = self._button("OK")
        if button is None:
            raise ActionError("no OK button")
        with ModalGuard(p.messages):
            yield {"kind": "button", "button": "OK"}
            wait_until(lambda: not Gui.Control.activeDialog(), timeout=3.0)
        if Gui.Control.activeDialog():  # FreeCAD refused: the dialog stays open
            p.on_plan = False  # only Cancel gets out of this now
            raise ActionError(f"OK refused: {p.messages[-1:] or 'dialog still open'}")
        self.pending = None
        feat = self.doc.getObject(p.feature) if p.feature else None
        if feat is not None:
            self.meta.objects[feat.Name] = ObjMeta(feat.Name, "feature", goal_ref=p.goal_ref, command=p.command)
            profile = getattr(feat, "Profile", None)
            if p.command in PROFILE_FEATURES and profile:
                om = self.meta.objects.get(profile[0].Name)
                if om is not None:
                    om.consumed_by = feat.Name
        self._consume_selection()
        n_tx = max(1, self.doc.UndoCount - p.undo_before)
        self._push_undo(UndoEntry(p.before, p.sel_before, doc_tx=True, n_tx=n_tx))
        used = getattr(feat, "Profile", None) if feat is not None else None
        profile_ok = p.profile is None or (bool(used) and used[0].Name == p.profile)
        on_plan = p.on_plan and fields_ok and profile_ok and feat is not None
        if not on_plan:
            self.meta.dirty += 1
        self._refresh_validity()
        self._sync_gui()
        info["changed"] = True

    def _reject(self, info: dict) -> None:
        p = self.pending
        button = self._button("Cancel")
        if button is None:
            raise ActionError("no Cancel button")
        with ModalGuard(p.messages):
            yield {"kind": "button", "button": "Cancel"}
            wait_until(lambda: not Gui.Control.activeDialog(), timeout=3.0)
        if Gui.Control.activeDialog():
            raise ActionError("Cancel did not close the dialog")
        self.pending = None
        extra = self.doc.UndoCount - p.undo_before  # Cancel should leave no transaction behind
        for _ in range(max(0, extra)):
            self.doc.undo()
        if p.feature and self.doc.getObject(p.feature) is not None:
            self.doc.removeObject(p.feature)
        self.doc.recompute()
        self.meta, self.sel_refs = p.before, p.sel_before
        self._refresh_validity()
        self._sync_gui()
        info["changed"] = True


class _KeyLogger(QtCore.QObject):
    def __init__(self, log: list) -> None:
        super().__init__()
        self.log = log

    def eventFilter(self, obj, ev):  # noqa: N802 (Qt API)
        if ev.type() in (QtCore.QEvent.KeyPress, QtCore.QEvent.ShortcutOverride):
            kind = "press" if ev.type() == QtCore.QEvent.KeyPress else "shortcut?"
            self.log.append(f"{kind} {ev.text()!r} key={ev.key()} -> {obj.metaObject().className()}#{obj.objectName()}")
            del self.log[:-50]
        return False

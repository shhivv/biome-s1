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
import re
import time
from dataclasses import dataclass, field

import FreeCAD as App
import FreeCADGui as Gui
from PySide import QtCore, QtWidgets

from ..actions import PROFILE_FEATURES, SOLID_FEATURE_TYPES
from ..expert import Meta, ObjMeta, progress
from ..goals import StartSpec
from ..runtime.gui_session import GuiSession
from ..runtime.session import UNDO_LIMIT, ActionError, UndoEntry
from ..schema import Goal, State
from . import spec as S
from .teacher import command_elements, dialog_elements, dialog_expert, fields_on_plan

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
    def __init__(self) -> None:
        super().__init__()
        self.pending: Pending | None = None
        self.messages: list[str] = []

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._close_dialog()
        super().close()

    def reset(self, goal: Goal, start: StartSpec | None = None) -> None:
        self._close_dialog()
        super().reset(goal, start)

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
        pump(2)
        return ok

    def _absorb_gui_transactions(self) -> None:
        """The GUI sometimes commits a transaction of its own after an action
        (leaving a sketch it re-entered after an Undo commits "Sketch
        recompute"). Fold such extras into the latest undo entry so one Undo
        still reverts exactly one step."""
        if self.doc is None or not self.undo_stack:
            return
        tx = sum(e.n_tx for e in self.undo_stack if e.doc_tx)
        extra = self.doc.UndoCount - tx
        if extra <= 0 or self.doc.UndoCount >= UNDO_LIMIT:  # FreeCAD caps its history; counts stop being exact
            return
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
            st.ui = {"dialog": self.pending.command, "fields": self.ui_fields()}
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
            if role == "cmd":
                self._open_dialog(command, info)
            elif self.pending is None:
                raise ActionError(f"{element}: no task dialog is open")
            elif role in ("set", "opt", "toggle"):
                self._edit_field(element)
            elif element == S.OK:
                self._accept(info)
            elif element == S.CANCEL:
                self._reject(info)
            else:
                raise ActionError(f"unknown element {element}")
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
            action.trigger()
            wait_until(lambda: bool(Gui.Control.activeDialog()), timeout=2.0)
        if not Gui.Control.activeDialog():
            created = [o.Name for o in self.doc.Objects if o.Name not in names_before]
            for _ in range(self.doc.UndoCount - pending.undo_before):
                self.doc.undo()
            raise ActionError(f"{command} opened no task dialog (created {created}; {pending.messages})")
        pending.feature = next((o.Name for o in self.doc.Objects
                                if o.Name not in names_before and o.TypeId in SOLID_FEATURE_TYPES), None)
        self.pending = pending
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
                self._set_number(w, float(value), fl.unit)
            elif fl.kind == "choice":
                idx = w.findText(S.option_text(element))
                if idx < 0:
                    raise ActionError(f"{element}: no such option")
                w.setCurrentIndex(idx)
            else:
                w.click()

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
            button.click()
            wait_until(lambda: not Gui.Control.activeDialog(), timeout=2.0)
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
            button.click()
            wait_until(lambda: not Gui.Control.activeDialog(), timeout=2.0)
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

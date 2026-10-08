"""Read FreeCAD's interface from the operating system's accessibility tree (macOS).

Mesa-S1's runtime normally reads the interface from inside FreeCAD (the Qt
widget tree). This module reads the same things from *outside* the app,
through the macOS accessibility API, the layer screen readers use:

- toolbar commands: each button's `AXHelp` is FreeCAD's command name; buttons
  that group several commands (FreeCAD 1.1's dropdown buttons) report the
  group's name instead and show the current action's name in their
  description, so those are matched by that name (`GROUPED_ACTIONS`);
- task-dialog fields: the last part of `AXIdentifier` is the widget's
  objectName (`lengthEdit`, `changeMode`, `checkBoxReversed`, ...), the same
  names Mesa-S1's element ids use; values come from the title (dropdowns),
  the value (check boxes) or the text (number fields);
- OK / Cancel.

Not available from accessibility, so still taken from the session: the
document state, the open dialog's command, a dropdown's other entries
(visible only while its menu is open), and the 3D view.

Lessons from FreeCAD 1.1 (scripts/mesa_ax.py):
- reads must not descend into the model tree: touching its rows makes Qt crash
  later, when FreeCAD rebuilds the tree;
- toolbar enabled states are only as fresh as FreeCAD's last refresh
  (`UiSession.refresh_commands` keeps them current);
- the GUI enables more commands than make sense (e.g. Fillet with no body), so
  the runtime's catalogue rules still decide which enabled commands are options.

Requires `pip install -e ".[ax]"` (pyobjc) and Accessibility permission for the
process running it (System Settings > Privacy & Security > Accessibility).
"""

from __future__ import annotations

import re
import time

from ..schema import State
from . import spec as S
from .teacher import dialog_elements

GROUPED_ACTIONS = {"New Sketch": "PartDesign_NewSketch"}  # displayed name -> command, for grouped toolbar buttons
# Never descended into: the model tree's rows (reading them crashes Qt later), and the menu bar
# (once FreeCAD has been in front its menus are populated, and a path through them leads back to
# the application element: the walk would loop).
SKIP_ROLES = ("AXOutline", "AXTable", "AXBrowser", "AXList", "AXApplication", "AXMenuBar", "AXMenuBarItem",
              "AXMenu", "AXMenuItem")
FIELD_ROLES = ("AXMenuButton", "AXCheckBox", "AXIncrementor", "AXTextField", "AXPopUpButton", "AXComboBox")
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def _ax():
    import ApplicationServices as AS  # pyobjc; macOS only

    return AS


class AccessibilityReader:
    """Snapshot of the parts of FreeCAD's accessibility tree Mesa-S1 uses."""

    def __init__(self, pid: int) -> None:
        self.AS = _ax()
        if not self.AS.AXIsProcessTrusted():
            raise PermissionError("grant Accessibility permission to this terminal / process")
        self.app = self.AS.AXUIElementCreateApplication(pid)

    def _attr(self, el, name):
        err, v = self.AS.AXUIElementCopyAttributeValue(el, name, None)
        return v if err == 0 else None

    def _text(self, el) -> str | None:
        n = self._attr(el, "AXNumberOfCharacters")
        if not n:
            return None
        rng = self.AS.AXValueCreate(self.AS.kAXValueCFRangeType, self.AS.CFRangeMake(0, int(n)))
        err, v = self.AS.AXUIElementCopyParameterizedAttributeValue(el, "AXStringForRange", rng, None)
        return str(v) if err == 0 and v is not None else None

    def read(self, wait: float = 10.0) -> dict:
        t0 = time.time()
        windows = self._attr(self.app, "AXWindows") or []
        while not windows and time.time() - t0 < wait:  # the window can take a moment to register
            time.sleep(0.2)
            windows = self._attr(self.app, "AXWindows") or []
        out = {"commands": {}, "fields": {}, "dialog_buttons": {}}
        stack = list(windows)
        while stack:
            el = stack.pop()
            role = str(self._attr(el, "AXRole") or "")
            ident = str(self._attr(el, "AXIdentifier") or "")
            in_dialog = ".Tasks." in ident
            help_ = self._attr(el, "AXHelp")
            if role in ("AXButton", "AXMenuButton") and help_ and not in_dialog:
                enabled = bool(self._attr(el, "AXEnabled"))
                out["commands"][str(help_)] = out["commands"].get(str(help_), False) or enabled
                m = re.search(r"<b>([^<]+)</b>", str(self._attr(el, "AXDescription") or ""))
                if role == "AXMenuButton" and m and m.group(1).strip() in GROUPED_ACTIONS:
                    out["commands"][GROUPED_ACTIONS[m.group(1).strip()]] = True  # grouped: the rules decide
            if in_dialog:
                title = self._attr(el, "AXTitle")
                if role == "AXButton" and str(title) in ("OK", "Cancel"):
                    out["dialog_buttons"][str(title)] = bool(self._attr(el, "AXEnabled"))
                elif role in FIELD_ROLES:
                    value = self._attr(el, "AXValue")
                    out["fields"][ident.split(".")[-1]] = {
                        "role": role, "title": None if title is None else str(title),
                        "value": value if isinstance(value, (bool, int, float)) or value is None else str(value),
                        "text": self._text(el) if role in ("AXIncrementor", "AXTextField") else None,
                        "enabled": bool(self._attr(el, "AXEnabled"))}
            if role in SKIP_ROLES or "Tree" in ident:  # never touch the model tree's rows (crashes Qt later)
                continue
            stack.extend(self._attr(el, "AXChildren") or [])
        out["seconds"] = round(time.time() - t0, 3)
        return out


def accessibility_view(snapshot: dict, state: State, actions: list[str]) -> tuple[list[str], dict | None, list[str]]:
    """Mesa-S1's inputs rebuilt from an accessibility snapshot: (element list, State.ui, disagreements).

    `state` / `actions` are the session's view; from them only the document state,
    the open dialog's command and feature, dropdown option lists, parameter targets
    and the elements accessibility cannot see (canvas, workbench, Done) are used."""
    notes: list[str] = []
    if state.ui and state.ui.get("dialog"):
        fields = {}
        for fl in S.DIALOG_FIELDS.get(state.ui["dialog"], []):
            f, qt = snapshot["fields"].get(fl.name), state.ui["fields"].get(fl.name)
            if f is None:
                if qt is not None:
                    notes.append(f"missing field {fl.name}")
                continue
            d: dict = {"kind": fl.kind, "enabled": f["enabled"]}
            if fl.kind == "number":
                m = _NUMBER.search(f.get("text") or "")
                d.update(value=float(m.group(0).replace(",", ".")) if m else None, unit=fl.unit,
                         target=(qt or {}).get("target"))
            elif fl.kind == "choice":
                d.update(value=f.get("title"), options=(qt or {}).get("options", []))
            else:
                d["value"] = bool(f.get("value"))
            if qt is not None and qt.get("value") != d["value"]:
                notes.append(f"value {fl.name}: session={qt.get('value')!r} accessibility={d['value']!r}")
            fields[fl.name] = d
        if "OK" not in snapshot["dialog_buttons"]:
            notes.append("OK/Cancel not found")
        ui = {**state.ui, "fields": fields}
        return dialog_elements(ui), ui, notes
    enabled = {c for c, on in snapshot["commands"].items() if on}
    elements = [a for a in actions if S.role(a) != "cmd" or S.underlying(a) in enabled]
    lost = [a for a in actions if a not in elements]
    if lost:
        notes.append(f"enabled per session but not per accessibility: {lost}")
    return elements, state.ui, notes


# ---------------------------------------------------------------------------------------------
# Acting from outside the app
#
# What works on FreeCAD 1.1 (found by probing the Pad and Pocket dialogs):
# - toolbar commands, check boxes, OK / Cancel: the accessibility "press" action;
# - number fields: focus them through accessibility, then select-all, paste the text (cmd+V) and
#   Tab, as key events posted to the FreeCAD process (setting their AXValue is accepted but ignored;
#   typed characters sometimes arrive without their text and are dropped, shortcuts don't).
#   Lengths and angles (Gui::QuantitySpinBox) only take typed text when editing finishes,
#   which in the background needs a step: up then down after typing.
# - dropdowns: unreliable. "press" sometimes opens the popup, but posted arrow keys + Return
#   reach it only some of the time (while FreeCAD is in the background, and even in front),
#   and the dropdown never takes keyboard focus while closed. The driver leaves dropdown
#   choices to the session by default; the code below is kept for --ax-dropdowns.
#   Don't read the popup's rows through accessibility: Qt crashes later when FreeCAD rebuilds
#   the panel. Marking a row selected does not commit, and posted mouse clicks are ignored.
# Never post Escape (it cancels the task dialog) or Return outside a popup (it presses the
# dialog's default button, OK).
# Posted events go to the process, not the screen: FreeCAD stays in the background and the
# user's mouse and keyboard are untouched.

_KEYCODES = {"a": 0, "0": 29, "1": 18, "2": 19, "3": 20, "4": 21, "5": 23, "6": 22, "7": 26, "8": 28, "9": 25,
             ".": 47, "-": 27, "v": 9, "tab": 48, "return": 36, "down": 125, "up": 126}


class AccessibilityActuator:
    """Performs UiSession.begin_external() requests through accessibility + posted key events."""

    def __init__(self, reader: AccessibilityReader, pid: int) -> None:
        self.r = reader
        self.AS = reader.AS
        import Quartz  # pyobjc; macOS only

        self.Q = Quartz
        self.pid = pid
        self.mouse = None  # set (a screen.Mouse) when the app is in front: fields are clicked, keys are real

    def _find(self, pred, skip_dialog: bool = False):
        stack = list(self.r._attr(self.r.app, "AXWindows") or [])
        while stack:
            el = stack.pop()
            if pred(el):
                return el
            role = str(self.r._attr(el, "AXRole") or "")
            ident = str(self.r._attr(el, "AXIdentifier") or "")
            if role in SKIP_ROLES or "Tree" in ident or (skip_dialog and ".Tasks." in ident):
                continue
            stack.extend(self.r._attr(el, "AXChildren") or [])
        return None

    def _field(self, name: str):
        return self._find(lambda e: str(self.r._attr(e, "AXIdentifier") or "").endswith("." + name))

    def _key(self, name: str, command: bool = False) -> None:
        for down in (True, False):
            ev = self.Q.CGEventCreateKeyboardEvent(None, _KEYCODES[name], down)
            if len(name) == 1 and not command:  # attach the character: macOS doesn't always derive it
                self.Q.CGEventKeyboardSetUnicodeString(ev, 1, name)
            if command:
                self.Q.CGEventSetFlags(ev, self.Q.kCGEventFlagMaskCommand)
            if self.mouse is not None:  # in front: real key events, to the focused field
                self.Q.CGEventPost(self.Q.kCGHIDEventTap, ev)
            else:
                self.Q.CGEventPostToPid(self.pid, ev)
            time.sleep(0.02)

    def _paste(self, text: str) -> None:
        """Enter text with cmd+V. Posted key events don't reliably carry their character (Qt then
        sees an empty keystroke and ignores it), while shortcuts work by key code. The clipboard is
        restored afterwards, every item and type."""
        import AppKit

        pb = AppKit.NSPasteboard.generalPasteboard()
        saved = [{t: it.dataForType_(t) for t in it.types()} for it in (pb.pasteboardItems() or [])]
        try:
            pb.clearContents()
            pb.setString_forType_(text, AppKit.NSPasteboardTypeString)
            self._key("v", command=True)
            time.sleep(0.15)
        finally:
            pb.clearContents()
            items = []
            for d in saved:
                item = AppKit.NSPasteboardItem.alloc().init()
                for t, data in d.items():
                    if data is not None:
                        item.setData_forType_(data, t)
                items.append(item)
            if items:
                pb.writeObjects_(items)

    def _press(self, el, what: str) -> None:
        if el is None:
            raise RuntimeError(f"accessibility: {what} not found")
        err = self.AS.AXUIElementPerformAction(el, "AXPress")
        if err != 0:
            raise RuntimeError(f"accessibility: pressing {what} failed ({err})")

    def perform(self, req: dict) -> None:
        kind = req["kind"]
        if kind == "command":
            cmd = req["command"]
            shown = {c: name for name, c in GROUPED_ACTIONS.items()}.get(cmd)

            def is_button(e) -> bool:
                if str(self.r._attr(e, "AXRole")) not in ("AXButton", "AXMenuButton"):
                    return False
                if str(self.r._attr(e, "AXHelp") or "") == cmd:
                    return True
                if shown is None:  # a grouped button shows its current action's name in bold
                    return False
                m = re.search(r"<b>([^<]+)</b>", str(self.r._attr(e, "AXDescription") or ""))
                return bool(m) and m.group(1).strip() == shown

            el = self._find(is_button, skip_dialog=True)
            self._press(el, f"toolbar button {cmd}")
        elif kind == "button":
            el = self._find(lambda e: str(self.r._attr(e, "AXRole")) == "AXButton"
                            and str(self.r._attr(e, "AXTitle")) == req["button"]
                            and ".Tasks." in str(self.r._attr(e, "AXIdentifier") or ""))
            self._press(el, f"dialog button {req['button']}")
        elif kind == "toggle":
            self._press(self._field(req["field"]), f"check box {req['field']}")
        elif kind == "number":
            el = self._field(req["field"])
            if el is None:
                raise RuntimeError(f"accessibility: field {req['field']} not found")
            # Focus through accessibility, never by clicking where the field claims to be: in a
            # scrolled task panel that spot can be another widget (e.g. the Python console).
            self.AS.AXUIElementSetAttributeValue(el, "AXFocused", True)
            time.sleep(0.15)
            self._key("a", command=True)
            self._paste(req["text"])
            # Spin boxes keep entered text pending until editing finishes, which needs a real focus
            # change (not available in the background, unreliable in front) or Return (which would
            # press the dialog's OK). Stepping commits the pending text: up, then down. The session
            # checks the value afterwards (and refuses counts above its cap).
            self._key("up")
            self._key("down")
            self._key("tab")
            time.sleep(0.2)
        elif kind == "choice":
            el = self._field(req["field"])
            if el is None:
                raise RuntimeError(f"accessibility: dropdown {req['field']} not found")
            import threading

            def popup_open() -> bool:  # the popup's list appears as a child of the dropdown (its rows are not read)
                return any(str(self.r._attr(c, "AXRole")) == "AXList" for c in self.r._attr(el, "AXChildren") or [])

            threading.Thread(target=lambda: self.AS.AXUIElementPerformAction(el, "AXPress"), daemon=True).start()
            deadline = time.time() + 3.0
            while not popup_open() and time.time() < deadline:
                time.sleep(0.05)
            if not popup_open():
                raise RuntimeError(f"accessibility: the {req['field']} popup did not open")
            time.sleep(0.15)
            # The entry order comes from the session (req["options"]); the popup's rows are never read:
            # touching them through accessibility makes Qt crash when FreeCAD rebuilds the panel afterwards.
            options = req["options"]
            if req["option"] not in options:
                raise RuntimeError(f"accessibility: {req['option']!r} not among {options}")
            delta = options.index(req["option"]) - (options.index(req["current"]) if req["current"] in options else 0)
            for _ in range(abs(delta)):
                self._key("down" if delta > 0 else "up")
                time.sleep(0.04)
            self._key("return")  # inside the open popup: picks the highlighted entry
            deadline = time.time() + 2.0
            while popup_open() and time.time() < deadline:
                time.sleep(0.05)
            if popup_open():
                raise RuntimeError(f"accessibility: the {req['field']} popup stayed open")
        else:
            raise RuntimeError(f"unknown request {kind}")

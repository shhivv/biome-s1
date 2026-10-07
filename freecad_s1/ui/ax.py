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
SKIP_ROLES = ("AXOutline", "AXTable", "AXBrowser", "AXList")
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

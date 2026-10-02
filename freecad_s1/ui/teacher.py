"""UI-level teacher and element enumeration.

The command-level expert (`expert.expert_actions`) still decides *what* to
build next. This module turns its answer into interface elements:

- no task dialog open: each acceptable command maps to its element
  (`cmd:PartDesign_Pad`, `canvas:Select:Face+Z`, `wb:...`, `Done`);
- a task dialog open: if the dialog belongs to an on-plan command, every
  field that does not hold its target yet is acceptable (any order), then
  OK; if it belongs to an off-plan command, Cancel.

The live dialog is described by a plain dict (`State.ui`), so this module
runs anywhere:

    {"dialog": "PartDesign_Pad",
     "fields": {"lengthEdit": {"kind": "number", "unit": "mm", "value": 10.0, "enabled": true},
                "changeMode": {"kind": "choice", "value": "Dimension", "options": [...], "enabled": true},
                "checkBoxReversed": {"kind": "check", "value": false, "enabled": true}}}

Pure stdlib.
"""

from __future__ import annotations

from .spec import CANCEL, EXCLUDED_OPTIONS, OK, to_ui, value_matches


def allowed_options(options: list[str]) -> list[str]:
    return [o for o in options if not any(x in o for x in EXCLUDED_OPTIONS)]


def dialog_elements(ui: dict) -> list[str]:
    """Elements offered while a task dialog is open."""
    out = []
    for name, fld in ui.get("fields", {}).items():
        if not fld.get("enabled", True):
            continue
        kind = fld["kind"]
        if kind == "number":
            out.append(f"set:{name}")
        elif kind == "choice":
            out += [f"opt:{name}={o}" for o in allowed_options(fld.get("options", []))]
        elif kind == "check":
            out.append(f"toggle:{name}")
    return out + [OK, CANCEL]


def dialog_expert(ui: dict, targets: dict[str, object], on_plan: bool) -> list[str]:
    """Acceptable elements while a dialog is open."""
    if not on_plan:
        return [CANCEL]
    need = []
    fields = ui.get("fields", {})
    for name, target in targets.items():
        fld = fields.get(name)
        if fld is None or not fld.get("enabled", True) or value_matches(target, fld.get("value")):
            continue
        kind = fld["kind"]
        if kind == "number":
            need.append(f"set:{name}")
        elif kind == "choice":
            if target in allowed_options(fld.get("options", [])):
                need.append(f"opt:{name}={target}")
        elif kind == "check":
            need.append(f"toggle:{name}")
    return need or [OK]


def fields_on_plan(ui: dict, targets: dict[str, object]) -> bool:
    """Every exposed field holds its target (what OK will commit)."""
    fields = ui.get("fields", {})
    return all(value_matches(t, fields[n].get("value")) for n, t in targets.items()
               if n in fields and fields[n].get("enabled", True))


def command_elements(actions: list[str]) -> list[str]:
    return list(dict.fromkeys(to_ui(a) for a in actions))


def command_expert(acceptable: list[str]) -> list[str]:
    return command_elements(acceptable)

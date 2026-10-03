"""UI element ids, task-dialog field specs and their targets.

An element id names one thing the model can act on:

    cmd:PartDesign_Pad            toolbar / menu command (a QAction)
    wb:PartDesignWorkbench        an entry of the workbench selector
    set:lengthEdit                type a value into a numeric field
    opt:changeMode=Through all    pick one entry of a combo box
    toggle:checkBoxReversed       click a check box
    click:OK / click:Cancel       the task dialog's buttons
    canvas:Select:Face+Z          an interaction in the 3D view or the sketch
    canvas:Sketcher_CreateCircle  editor, which the widget tree cannot see
    Done                          terminal

The model picks the element; the value typed into a `set:` field comes from
the parameter stage (`targets`), like numeric command arguments do in the
command-level model. Combo entries and check boxes are discrete, so the
model chooses them itself.

Each task dialog exposes only the fields listed in `DIALOG_FIELDS`, and every
listed field has a target, so the teacher can always say whether the dialog
is filled in as the plan requires. Field names are the widgets'
`objectName`s in FreeCAD 1.1 (dump them with scripts/probe_ui_dialogs.py);
`UiSession` refuses a dialog in which any listed field is missing.

Pure stdlib: no FreeCAD import.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..actions import CATALOGUE
from ..expert import expert_plan_length
from ..runtime import params as P
from ..schema import GoalFeature

ROLES = ["<none>", "cmd", "wb", "set", "opt", "toggle", "click", "canvas", "done"]

OK = "click:OK"
CANCEL = "click:Cancel"


@dataclass(frozen=True)
class Field:
    name: str  # widget objectName
    kind: str  # "number" | "choice" | "check"
    unit: str = ""  # numbers: "mm" | "deg" | "count"


def _f(name: str, kind: str, unit: str = "") -> Field:
    return Field(name, kind, unit)


_PROFILE = [_f("sidesMode", "choice"), _f("changeMode", "choice"), _f("lengthEdit", "number", "mm"),
            _f("checkBoxReversed", "check")]
_REVOLVE = [_f("changeMode", "choice"), _f("axis", "choice"), _f("revolveAngle", "number", "deg"),
            _f("checkBoxMidplane", "check"), _f("checkBoxReversed", "check")]

DIALOG_FIELDS: dict[str, list[Field]] = {
    "PartDesign_Pad": _PROFILE,
    "PartDesign_Pocket": _PROFILE,
    "PartDesign_Revolution": _REVOLVE,
    "PartDesign_Groove": _REVOLVE,
    "PartDesign_Hole": [_f("DepthType", "choice"), _f("Diameter", "number", "mm"), _f("Depth", "number", "mm"),
                        _f("ThreadType", "choice"), _f("HoleCutType", "choice"), _f("Reversed", "check")],
    "PartDesign_Fillet": [_f("filletRadius", "number", "mm"), _f("checkBoxUseAllEdges", "check")],
    "PartDesign_Chamfer": [_f("chamferType", "choice"), _f("chamferSize", "number", "mm"),
                           _f("checkBoxUseAllEdges", "check")],
    "PartDesign_Thickness": [_f("Value", "number", "mm"), _f("modeComboBox", "choice"), _f("joinComboBox", "choice"),
                             _f("checkIntersection", "check"), _f("checkReverse", "check")],
    "PartDesign_Draft": [],
    "PartDesign_Mirrored": [_f("comboPlane", "choice")],
    "PartDesign_LinearPattern": [_f("comboDirection", "choice"), _f("comboMode", "choice"),
                                 _f("spinExtent", "number", "mm"), _f("spinOccurrences", "number", "count"),
                                 _f("enableCheckbox", "check")],
    "PartDesign_PolarPattern": [_f("comboDirection", "choice"), _f("comboMode", "choice"),
                                _f("spinExtent", "number", "deg"), _f("spinOccurrences", "number", "count")],
}

# Commands that open a task dialog in the GUI; everything else in the
# catalogue completes without one.
DIALOG_COMMANDS = set(DIALOG_FIELDS)

# Combo entries that need a pick in the 3D view (or extra fields we do not
# expose); never offered.
EXCLUDED_OPTIONS = ("reference", "Up to face", "Up to shape", "Custom direction", "Two sided", "Two angles",
                    "Two distances", "Distance and angle")

# Command-level actions that happen in the 3D view or the sketch editor.
CANVAS_CATEGORIES = {"select", "sketch_geometry", "sketch_constraint"}


def is_canvas(action: str) -> bool:
    spec = CATALOGUE.get(action)
    return spec is not None and (spec.category in CANVAS_CATEGORIES or action == "Sketcher_ToggleConstruction")


def to_ui(action: str) -> str:
    """Command-level action id -> UI element id."""
    if action == "Done":
        return "Done"
    if action.startswith("Std_Workbench:"):
        return "wb:" + action.split(":", 1)[1]
    if is_canvas(action):
        return "canvas:" + action
    return "cmd:" + action


def role(element: str) -> str:
    if element == "Done":
        return "done"
    head = element.split(":", 1)[0]
    return head if head in ROLES else "<none>"


def underlying(element: str) -> str | None:
    """The command-level action an element stands for (None for dialog widgets)."""
    r = role(element)
    if r == "done":
        return "Done"
    if r == "wb":
        return "Std_Workbench:" + element[3:]
    if r in ("cmd", "canvas"):
        return element.split(":", 1)[1]
    return None


def field_name(element: str) -> str:
    """`set:lengthEdit` / `opt:changeMode=Dimension` / `toggle:checkBoxReversed` -> widget objectName."""
    return element.split(":", 1)[1].split("=", 1)[0]


def option_text(element: str) -> str:
    return element.split("=", 1)[1]


def targets(command: str, f: GoalFeature, scale: float) -> dict[str, object]:
    """Field values the plan requires for `command` built from intent `f`.
    None means "any value" (e.g. the pocket length when it cuts through all).
    Defaults are spelled out so off-plan clicks are detectable, and they match
    what the headless executors build (runtime/session.py)."""
    if command in ("PartDesign_Pad", "PartDesign_Pocket"):
        if command == "PartDesign_Pad":
            mode, length = "Dimension", P.pad_length(f, scale)
        else:
            typ, length = P.pocket_spec(f, scale)
            mode = "Dimension" if typ == "Length" else "Through all"
            length = length if typ == "Length" else None
        return {"sidesMode": "One sided", "changeMode": mode, "lengthEdit": length, "checkBoxReversed": False}
    if command in ("PartDesign_Revolution", "PartDesign_Groove"):
        return {"changeMode": "Angle", "axis": "Vertical sketch axis", "revolveAngle": 360.0,
                "checkBoxMidplane": False, "checkBoxReversed": False}
    if command == "PartDesign_Hole":
        return {"DepthType": "Through all", "Diameter": P.hole_diameter(f, scale), "Depth": None,
                "ThreadType": "None", "HoleCutType": "None", "Reversed": False}
    if command == "PartDesign_Fillet":
        return {"filletRadius": P.dressup_value(f, command), "checkBoxUseAllEdges": False}
    if command == "PartDesign_Chamfer":
        return {"chamferType": "Equal distance", "chamferSize": P.dressup_value(f, command),
                "checkBoxUseAllEdges": False}
    if command == "PartDesign_Thickness":
        return {"Value": P.dressup_value(f, command), "modeComboBox": "Skin", "joinComboBox": "Arc",
                "checkIntersection": False, "checkReverse": True}  # = Reversed, the headless default
    if command == "PartDesign_Mirrored":
        return {"comboPlane": "Base YZ-plane"}
    if command == "PartDesign_LinearPattern":
        spec = P.pattern_spec(f, command, scale)
        return {"comboDirection": "Base X-axis", "comboMode": "Extent", "spinExtent": spec["length"],
                "spinOccurrences": spec["n"], "enableCheckbox": False}
    if command == "PartDesign_PolarPattern":
        spec = P.pattern_spec(f, command, scale)
        return {"comboDirection": "Base Z-axis", "comboMode": "Extent", "spinExtent": 360.0,
                "spinOccurrences": spec["n"]}
    return {}


def ui_step_budget(goal, start) -> int:
    """Step budget for a UI episode: the command-level plan plus, per intent,
    a task dialog's field edits and OK (at most ~5), doubled + 6 like
    runtime.episode.step_budget."""
    n = expert_plan_length(goal, start.doc_open, start.workbench, start.body) + 5 * len(goal.features)
    return 2 * n + 6


def fallback_number(unit: str, scale: float) -> float:
    """Value typed into a numeric field the plan does not care about."""
    return {"deg": 180.0, "count": 3.0}.get(unit, round(max(1.0, scale * 0.2), 1))


def value_matches(target: object, value: object) -> bool:
    if target is None:
        return True
    if isinstance(target, bool) or isinstance(target, str):
        return value == target
    try:
        return abs(float(value) - float(target)) <= 1e-3 * max(1.0, abs(float(target)))
    except (TypeError, ValueError):
        return False


# Words in UI element ids beyond the command catalogue's, for the word-piece
# vocabulary of UI models (model.featurize).
UI_ID_EXAMPLES = (
    ["set:" + fl.name for fields in DIALOG_FIELDS.values() for fl in fields if fl.kind == "number"]
    + ["toggle:" + fl.name for fields in DIALOG_FIELDS.values() for fl in fields if fl.kind == "check"]
    + [f"opt:{k}={v}" for c in DIALOG_FIELDS for k, v in targets(c, GoalFeature("<unk>", {}), 1.0).items()
       if isinstance(v, str)]
    + ["opt:x=Through all", "opt:x=To first", "opt:x=To last", "opt:x=Horizontal sketch axis",
       "opt:x=Normal sketch axis", "opt:x=Base Y-axis", "opt:x=Base XY-plane", "opt:x=Base XZ-plane",
       "opt:x=Spacing", "opt:x=Pipe", "opt:x=Recto verso", "opt:x=Intersection", "opt:x=Counterbore",
       "opt:x=Countersink", "opt:x=Counterdrill", "opt:x=ISO metric regular", "opt:x=Symmetric",
       OK, CANCEL, "cmd:x", "wb:x", "canvas:x"]
)

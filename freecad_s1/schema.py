"""Structured FreeCAD state, goal and action records plus the fixed vocabularies.

Everything here is plain stdlib so it runs both inside FreeCAD's bundled
Python (runtime, data generation) and in the torch process (featurization).
States cross the process boundary as JSON dicts produced by `asdict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Vocabularies. Index 0 is always the unknown/padding bucket.
# ---------------------------------------------------------------------------

WORKBENCHES = [
    "<unk>",
    "NoneWorkbench",
    "StartWorkbench",
    "PartWorkbench",
    "PartDesignWorkbench",
    "SketcherWorkbench",
]

NODE_TYPES = [
    "<unk>",
    "PartDesign::Body",
    "Sketcher::SketchObject",
    "PartDesign::Pad",
    "PartDesign::Pocket",
    "PartDesign::Revolution",
    "PartDesign::Groove",
    "PartDesign::Hole",
    "PartDesign::Fillet",
    "PartDesign::Chamfer",
    "PartDesign::Draft",
    "PartDesign::Thickness",
    "PartDesign::Mirrored",
    "PartDesign::LinearPattern",
    "PartDesign::PolarPattern",
    "Part::Box",
    "Part::Cylinder",
]

GEOMETRY_KINDS = ["line", "circle", "arc", "point", "other"]

CONSTRAINT_KINDS = [
    "Coincident",
    "Horizontal",
    "Vertical",
    "DistanceX",
    "DistanceY",
    "Distance",
    "Radius",
    "Diameter",
    "Equal",
    "PointOnObject",
    "Symmetric",
    "Lock",
    "Other",
]

SELECTION_KINDS = ["<none>", "plane", "face", "edges", "feature", "sketch", "other"]

# Numeric per-node attributes. Keys in LENGTH_KEYS are divided by the goal
# scale during featurization; everything else is used as-is (already O(1)).
NODE_NUM_KEYS = [
    "tip", "in_edit", "valid", "visible", "consumed", "active_body",
    "n_geo", "n_construction", "n_constraints", "dof", "fully_constrained", "closed",
    "support_nx", "support_ny", "support_nz", "support_offset", "on_face",
    "sk_w", "sk_h", "sk_cx", "sk_cy",
    "length", "through_all", "reversed", "angle", "radius", "occurrences", "n_refs",
    "volume_ratio", "n_faces",
]
LENGTH_KEYS = {"support_offset", "sk_w", "sk_h", "sk_cx", "sk_cy", "length", "radius"}

GOAL_KINDS = [
    "<unk>",
    "base_box", "base_cyl", "base_hex", "base_ring",
    "boss_cyl", "boss_box", "hole", "hole_std", "pocket_rect",
    "polar_pattern", "linear_pattern", "mirror",
    "fillet_top", "fillet_vertical", "chamfer_top", "shell",
]
GOAL_PARAM_KEYS = ["w", "d", "h", "r", "ri", "ro", "x", "y", "depth", "n", "length", "size", "t"]
GOAL_LENGTH_KEYS = {"w", "d", "h", "r", "ri", "ro", "x", "y", "depth", "length", "size", "t"}

RECENT_ACTIONS = 8


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """One object in the feature tree, in document (= build) order."""

    name: str
    type: str
    parent: int = -1  # index of the owning node (the Body), -1 for top level
    depth: int = 0
    num: dict[str, float] = field(default_factory=dict)
    geo: dict[str, int] = field(default_factory=dict)  # sketch geometry kind counts
    cons: dict[str, int] = field(default_factory=dict)  # sketch constraint kind counts


@dataclass
class SelItem:
    kind: str  # one of SELECTION_KINDS
    object: str
    object_type: str
    subs: list[str] = field(default_factory=list)
    normal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset: float = 0.0  # signed distance of the face/plane from origin along normal
    count: int = 0  # number of sub-elements


@dataclass
class ShapeInfo:
    valid: bool = False
    volume: float = 0.0
    area: float = 0.0
    bbox: tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_faces: int = 0
    n_edges: int = 0
    n_solids: int = 0
    face_dirs: list[str] = field(default_factory=list)  # e.g. ["+Z", "-Z", "+X"]


@dataclass
class State:
    doc_open: bool = False
    workbench: str = "StartWorkbench"
    edit: str | None = None  # name of the object in edit mode (a sketch)
    has_body: bool = False
    undo_available: bool = False
    tree: list[Node] = field(default_factory=list)
    selection: list[SelItem] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)  # last RECENT_ACTIONS action ids
    events: list[str] = field(default_factory=list)  # recent observer events (debug/telemetry)
    shape: ShapeInfo = field(default_factory=ShapeInfo)
    ui: dict[str, Any] | None = None  # open task dialog and its fields (UI-level sessions only, see ui/teacher.py)
    # Per recent action (aligned with `recent`): 1.0 for an OK pressed while every number field held its target
    # (the agent saw that on screen when it pressed OK; afterwards the values are no longer shown).
    recent_flags: list[float] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> "State":
        return State(
            doc_open=d["doc_open"],
            workbench=d["workbench"],
            edit=d.get("edit"),
            has_body=d.get("has_body", False),
            undo_available=d.get("undo_available", False),
            tree=[Node(**n) for n in d.get("tree", [])],
            selection=[SelItem(**{**s, "normal": tuple(s["normal"])}) for s in d.get("selection", [])],
            recent=list(d.get("recent", [])),
            recent_flags=list(d.get("recent_flags", [])),
            events=list(d.get("events", [])),
            shape=ShapeInfo(**{**d["shape"], "bbox": tuple(d["shape"]["bbox"])}),
            ui=d.get("ui"),
        )


@dataclass
class GoalFeature:
    kind: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class Goal:
    """What to build: an ordered list of feature intents plus global target
    descriptors measured on the target solid. `scale` normalizes lengths."""

    features: list[GoalFeature]
    level: int = 1
    scale: float = 1.0
    target: ShapeInfo = field(default_factory=ShapeInfo)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Goal":
        t = d.get("target") or {}
        return Goal(
            features=[GoalFeature(**f) for f in d["features"]],
            level=d.get("level", 1),
            scale=d.get("scale", 1.0),
            target=ShapeInfo(**{**t, "bbox": tuple(t.get("bbox", (0, 0, 0)))}) if t else ShapeInfo(),
        )

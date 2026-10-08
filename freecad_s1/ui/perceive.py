"""Mesa-S1's screen state (freecad_s1/screen_state.py) rebuilt from what is on
screen, without FreeCAD's API.

- The model tree comes from OCR of a screenshot (`hands.Hands.tree_rows`).
  FreeCAD's tree nests a sketch under the feature that consumed it (and keeps
  that branch collapsed), while the session lists objects in creation order,
  all under the Body. Every profile feature (Pad, Pocket, Revolution, Groove,
  Hole) consumes exactly one sketch, created just before it, so the hidden
  sketch rows are put back in that order.
- Mode (workbench, sketch in edit, undo available, the open dialog and its
  fields) comes from accessibility (`ax.AccessibilityReader`).
- The selection is what the executor picked (it read the element and its 3D
  point back from the status bar); recent actions are the agent's own history.
"""
from __future__ import annotations

import re

from ..schema import Node, SelItem, State

# Tree label -> object type. FreeCAD names objects after their type (Pad, Pad001, ...).
TYPE_OF = {
    "Body": "PartDesign::Body", "Sketch": "Sketcher::SketchObject", "Pad": "PartDesign::Pad",
    "Pocket": "PartDesign::Pocket", "Revolution": "PartDesign::Revolution", "Groove": "PartDesign::Groove",
    "Hole": "PartDesign::Hole", "Fillet": "PartDesign::Fillet", "Chamfer": "PartDesign::Chamfer",
    "Draft": "PartDesign::Draft", "Thickness": "PartDesign::Thickness", "Mirrored": "PartDesign::Mirrored",
    "LinearPattern": "PartDesign::LinearPattern", "PolarPattern": "PartDesign::PolarPattern",
}
PROFILE_TYPES = {"PartDesign::Pad", "PartDesign::Pocket", "PartDesign::Revolution", "PartDesign::Groove",
                 "PartDesign::Hole"}
SOLID_TYPES = set(TYPE_OF.values()) - {"PartDesign::Body", "Sketcher::SketchObject"}
_LABEL = re.compile(r"(?:^|[^A-Za-z0-9_])(" + "|".join(sorted(TYPE_OF, key=len, reverse=True)) + r")(\d{3})?(?![A-Za-z0-9_])")


def tree_objects(rows) -> list[tuple[str, str, float]]:
    """(label, type, x) for the object rows of an OCR'd model tree, top to bottom. `rows` are
    screen.Text items; origin features, the document row and anything unrecognised are skipped."""
    out = []
    for t in sorted(rows, key=lambda r: r.y):
        m = _LABEL.search(t.text)
        if m:
            label = m.group(1) + (m.group(2) or "")
            out.append((label, TYPE_OF[m.group(1)], t.x))  # the row's left edge: deeper rows start further right
    return out


def tree_state(objects: list[tuple[str, str, float]], edit: str | None = None) -> list[Node]:
    """State.tree (creation order, everything under the Body) from the visible tree rows.

    A sketch row indented under a feature row belongs to it; collapsed branches hide those rows,
    so each profile feature without a visible sketch child gets one put back before it."""
    nodes: list[Node] = []
    body = -1
    pending_sketch: list[tuple[str, float]] = []  # sketch rows seen nested under the next feature
    feat_x = None
    for label, typ, x in objects:
        if typ == "PartDesign::Body":
            nodes.append(Node(name=label, type=typ, parent=-1, depth=0, num={"active_body": 1.0, "visible": 1.0}))
            body = len(nodes) - 1
            continue
        if typ == "Sketcher::SketchObject" and feat_x is not None and x > feat_x + 10:
            # nested under the feature row just above it: it was created before that feature
            nodes.insert(len(nodes) - 1, Node(name=label, type=typ, parent=body, depth=1,
                                              num={"consumed": 1.0, "visible": 0.0}))
            nodes[-1].num["has_profile"] = 1.0
            continue
        node = Node(name=label, type=typ, parent=body, depth=1, num={"visible": 1.0})
        if typ == "Sketcher::SketchObject":
            node.num.update({"consumed": 0.0, "in_edit": float(label == edit)})
            feat_x = None
        else:
            feat_x = x
        nodes.append(node)
    out: list[Node] = []
    for n in nodes:  # put back the sketches of collapsed profile features
        if n.type in PROFILE_TYPES and not n.num.pop("has_profile", 0.0):
            out.append(Node(name="Sketch?", type="Sketcher::SketchObject", parent=n.parent, depth=1,
                            num={"consumed": 1.0, "visible": 0.0, "in_edit": 0.0}))
        n.num.pop("has_profile", None)
        out.append(n)
    solids = [i for i, n in enumerate(out) if n.type in SOLID_TYPES]
    for i, n in enumerate(out):
        n.num["tip"] = float(bool(solids) and i == solids[-1])
        n.num.setdefault("in_edit", 0.0)
    for i in solids[:-1]:  # PartDesign shows only the tip; earlier solids are hidden
        out[i].num["visible"] = 0.0
    return out


def perceived_state(tree_rows, ax_snapshot: dict, workbench: str, edit_open: bool, undo: bool,
                    selection: list[SelItem], recent: list[str], ui: dict | None) -> State:
    """The screen state from the screen: tree OCR rows, accessibility facts, the executor's memory."""
    objects = tree_objects(tree_rows)
    sketches = [label for label, typ, _ in objects if typ == "Sketcher::SketchObject"]
    edit = sketches[-1] if (edit_open and sketches) else None
    tree = tree_state(objects, edit)
    return State(doc_open=True, workbench=workbench, edit=edit, has_body=any(n.type == "PartDesign::Body" for n in tree),
                 undo_available=undo, tree=tree, selection=list(selection), recent=list(recent), ui=ui)

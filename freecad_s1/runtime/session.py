"""FreeCAD session: snapshot state, enumerate actions, execute actions.

Must run inside FreeCAD's Python (``FreeCAD``, ``Part``, ``Sketcher``
importable). `HeadlessSession` works in FreeCADCmd / the bundled interpreter
with no GUI: workbench, selection and edit mode — which live in FreeCADGui
normally — are tracked by the session. `gui_session.GuiSession` reads them
from the real GUI instead.

Every document-changing action runs in its own FreeCAD transaction, so
``Std_Undo`` maps 1:1 onto ``doc.undo()``. Actions that turn out to be
no-ops (e.g. a constraint with nothing to apply to) abort their transaction.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass

import FreeCAD as App
import Part
import Sketcher

from ..actions import CATALOGUE, SOLID_FEATURE_TYPES, enumerate_actions
from ..expert import Meta, ObjMeta, expert_actions, progress
from ..goals import StartSpec
from ..schema import CONSTRAINT_KINDS, Goal, Node, SelItem, ShapeInfo, State, RECENT_ACTIONS
from . import params as P

V = App.Vector
AXES = {"X": V(1, 0, 0), "Y": V(0, 1, 0), "Z": V(0, 0, 1)}
PLANE_ROLES = {"XY": "XY_Plane", "XZ": "XZ_Plane", "YZ": "YZ_Plane"}

# FreeCAD "Lock" = point DistanceX + DistanceY; counted as one "Lock" per point.
CONSTRAINT_SLOTS = {
    "Sketcher_ConstrainDistanceX": ("dx", {"rect", "line"}),
    "Sketcher_ConstrainDistanceY": ("dy", {"rect", "line"}),
    "Sketcher_ConstrainDiameter": ("size", {"circle", "hex"}),
    "Sketcher_ConstrainRadius": ("size", {"circle", "hex"}),
    "Sketcher_ConstrainLock": ("lock", {"rect", "circle", "hex", "line", "point"}),
    "Sketcher_ConstrainHorizontal": ("orient", {"hex", "line"}),
    "Sketcher_ConstrainVertical": ("orient", {"line"}),
}
GEO_PARAMS = {"LineSegment": 4, "Circle": 3, "ArcOfCircle": 5, "Point": 2}
CONSTRAINT_DOF = {"Coincident": 2, "Symmetric": 2, "Block": 3}


# FreeCAD keeps 20 undo transactions unless the GUI raises the limit; the
# limit is not settable from Python, so older entries become permanent.
UNDO_LIMIT = 20


class ActionError(Exception):
    pass


@dataclass
class UndoEntry:
    meta: Meta
    sel_refs: list
    doc_tx: bool
    n_tx: int = 1  # FreeCAD transactions to undo (a GUI task dialog may commit more than one)


class _Observer:
    """Document observer feeding the `events` stream (App.addDocumentObserver)."""

    def __init__(self, sink: deque):
        self.sink = sink

    def slotCreatedObject(self, obj):  # noqa: N802 (FreeCAD callback names)
        self.sink.append(f"created:{obj.TypeId}")

    def slotDeletedObject(self, obj):  # noqa: N802
        self.sink.append(f"deleted:{obj.TypeId}")

    def slotUndoDocument(self, doc):  # noqa: N802
        self.sink.append("undo")

    def slotRecomputedDocument(self, doc):  # noqa: N802
        self.sink.append("recomputed")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _is_plane(face) -> bool:
    return face.Surface.__class__.__name__ == "Plane"


def _face_normal(face) -> V:
    u0, u1, v0, v1 = face.ParameterRange
    return face.normalAt((u0 + u1) / 2, (v0 + v1) / 2)


def planar_faces(shape, axis: V) -> list[tuple[int, object]]:
    out = []
    for i, f in enumerate(shape.Faces):
        if _is_plane(f) and _face_normal(f).dot(axis) > 0.999:
            out.append((i, f))
    return out


def pick_face(shape, direction: str):
    """Largest planar face whose outward normal is `direction` (e.g. "+Z")."""
    axis = AXES[direction[1]] * (1 if direction[0] == "+" else -1)
    faces = planar_faces(shape, axis)
    if not faces:
        return None
    return max(faces, key=lambda t: (round(t[1].Area, 6), t[1].CenterOfMass.dot(axis)))


def edge_names(shape, edges) -> list[str]:
    out = []
    for i, e in enumerate(shape.Edges):
        if any(e.isSame(x) for x in edges):
            out.append(f"Edge{i + 1}")
    return out


def line_direction(edge):
    """Direction of a straight edge, or None. Some OCC edges (e.g. from Draft
    features) have curve types FreeCAD cannot wrap, and `.Curve` raises."""
    try:
        c = edge.Curve
    except TypeError:
        return None
    return c.Direction if c.__class__.__name__ == "Line" else None


def vertical_edges(shape) -> list[str]:
    out = []
    for i, e in enumerate(shape.Edges):
        d = line_direction(e)
        if d is not None and abs(abs(d.z) - 1) < 1e-6:
            out.append(f"Edge{i + 1}")
    return out


def shape_info(shape) -> ShapeInfo:
    if shape is None or shape.isNull() or not shape.Solids:
        return ShapeInfo()
    bb = shape.BoundBox
    dirs = []
    for d in ("+Z", "-Z", "+X", "-X", "+Y", "-Y"):
        if planar_faces(shape, AXES[d[1]] * (1 if d[0] == "+" else -1)):
            dirs.append(d)
    if vertical_edges(shape):
        dirs.append("|Z")
    return ShapeInfo(
        valid=shape.isValid(),
        volume=shape.Volume,
        area=shape.Area,
        bbox=(bb.XLength, bb.YLength, bb.ZLength),
        n_faces=len(shape.Faces),
        n_edges=len(shape.Edges),
        n_solids=len(shape.Solids),
        face_dirs=dirs,
    )


def iou(a, b) -> float:
    """Volumetric IoU of two solids (exact, via OCC booleans)."""
    if a is None or b is None or a.isNull() or b.isNull() or not a.Solids or not b.Solids:
        return 0.0
    try:
        inter = a.common(b).Volume
    except Exception:  # boolean failure
        return 0.0
    union = a.Volume + b.Volume - inter
    return float(inter / union) if union > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class HeadlessSession:
    _counter = 0
    undo_limit = UNDO_LIMIT  # FreeCAD's history length for this session's documents

    def __init__(self) -> None:
        self.doc = None
        self.goal: Goal | None = None
        self.meta = Meta()
        self.sel_refs: list[tuple[str, list[str], str]] = []  # (object name, subs, select arg)
        self.undo_stack: list[UndoEntry] = []
        self.recent: deque[str] = deque(maxlen=RECENT_ACTIONS)
        self.events: deque[str] = deque(maxlen=16)
        self.done = False
        self._observer = _Observer(self.events)
        App.addDocumentObserver(self._observer)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self.doc is not None:
            try:
                App.closeDocument(self.doc.Name)
            except Exception:
                pass
            self.doc = None

    def shutdown(self) -> None:
        self.close()
        try:
            App.removeDocumentObserver(self._observer)
        except Exception:
            pass

    def reset(self, goal: Goal, start: StartSpec | None = None) -> None:
        start = start or StartSpec()
        self.close()
        self.goal = goal
        self.meta = Meta(workbench=start.workbench)
        self.sel_refs = []
        self.undo_stack = []
        self.recent.clear()
        self.done = False
        if start.doc_open:
            self._new_document()
            if start.body:
                self._make_body()
        self.events.clear()

    def _new_document(self) -> None:
        HeadlessSession._counter += 1
        self.doc = App.newDocument(f"S1Doc{HeadlessSession._counter}")
        self.doc.UndoMode = 1
        self.meta.doc_open = True

    def _make_body(self) -> str:
        body = self.doc.addObject("PartDesign::Body", "Body")
        if self.meta.body is None:
            self.meta.body = body.Name
            self.meta.objects[body.Name] = ObjMeta(body.Name, "body", command="PartDesign_Body")
        else:
            self.meta.objects[body.Name] = ObjMeta(body.Name, "other", command="PartDesign_Body")
        return body.Name

    # -- accessors ----------------------------------------------------------

    @property
    def body(self):
        return self.doc.getObject(self.meta.body) if (self.doc and self.meta.body) else None

    def tip(self):
        b = self.body
        return b.Tip if b is not None else None

    def solid(self):
        t = self.tip()
        if t is None or not hasattr(t, "Shape") or t.Shape.isNull() or not t.Shape.Solids:
            return None
        return t.Shape

    def origin_feature(self, role: str):
        return next(o for o in self.body.Origin.OriginFeatures if o.Role == role)

    def workbench(self) -> str:
        return self.meta.workbench

    def edit_object(self) -> str | None:
        return self.meta.edit

    # -- public API ----------------------------------------------------------

    def state(self) -> State:
        return snapshot(self)

    def valid_actions(self, state: State | None = None) -> list[str]:
        return enumerate_actions(state or self.state())

    def expert(self) -> list[str]:
        return self.command_expert()

    def command_expert(self) -> list[str]:
        """Acceptable command-level actions; empty when the state is
        unrecoverable (the off-plan change is older than FreeCAD's undo
        history). Subclasses that act on other interfaces override `expert`."""
        acts = expert_actions(self.goal, self.meta)
        if acts == ["Std_Undo"] and not self.undo_stack:
            return []
        return acts

    def progress(self) -> int:
        """Index of the goal intent currently being worked on (== number of
        intents when everything is built). Auxiliary label, never an input."""
        return progress(self.goal, self.meta)

    def step(self, action: str) -> dict:
        """Execute `action`. Returns {"changed", "error", "done", "on_plan"}."""
        on_plan = action in self.command_expert()
        info = {"changed": False, "error": None, "done": False, "on_plan": on_plan}
        spec = CATALOGUE.get(action)
        if spec is None:
            info["error"] = f"unknown action {action}"
            return info
        self.recent.append(action)
        try:
            if action == "Done":
                self.done = info["done"] = True
            elif action == "Std_New":
                self._new_document()
                info["changed"] = True
            elif action.startswith("Std_Workbench:"):
                self.meta.workbench = action.split(":", 1)[1]
            elif action == "Std_ViewFitAll":
                pass
            elif action == "Std_Undo":
                info["changed"] = self._undo()
            elif action.startswith("Select:"):
                self._select(action.split(":", 1)[1])
            elif action == "Sketcher_LeaveSketch":
                before = copy.deepcopy(self.meta)
                self.meta.edit = None
                self._push_undo(UndoEntry(before, list(self.sel_refs), doc_tx=False))
                if not on_plan:
                    self.meta.dirty += 1
                info["changed"] = True
            else:
                info["changed"] = self._transact(action, on_plan)
        except ActionError as exc:
            info["error"] = str(exc)
        return info

    # -- transactions / undo ----------------------------------------------

    def _transact(self, action: str, on_plan: bool) -> bool:
        # Decide no-ops before opening a transaction: aborting an empty
        # transaction inside sketch edits desyncs FreeCAD's undo stack.
        precheck = PRECHECKS.get(action)
        if precheck is not None and not precheck(self, action):
            return False
        before = copy.deepcopy(self.meta)
        sel_before = list(self.sel_refs)
        self.doc.openTransaction(action)
        try:
            changed = EXECUTORS[action](self, action)
            if changed:
                self.doc.recompute()
        except Exception as exc:
            self.doc.abortTransaction()
            self.meta, self.sel_refs = before, sel_before
            raise ActionError(f"{action} failed: {exc}") from exc
        if not changed:
            self.doc.abortTransaction()
            self.meta, self.sel_refs = before, sel_before
            return False
        self.doc.commitTransaction()
        self._push_undo(UndoEntry(before, sel_before, doc_tx=True))
        if not on_plan:
            self.meta.dirty += 1
        self._refresh_validity()
        return True

    def _push_undo(self, entry: UndoEntry) -> None:
        self.undo_stack.append(entry)
        while sum(e.n_tx for e in self.undo_stack if e.doc_tx) > self.undo_limit:
            while not self.undo_stack.pop(0).doc_tx:
                pass

    def _undo(self) -> bool:
        if not self.undo_stack:
            return False
        entry = self.undo_stack.pop()
        if entry.doc_tx:
            for _ in range(entry.n_tx):
                self.doc.undo()
            self.doc.recompute()
        workbench = self.meta.workbench
        self.meta = entry.meta
        self.meta.workbench = workbench
        self.sel_refs = entry.sel_refs
        self._refresh_validity()
        return True

    def _refresh_validity(self) -> None:
        for name, om in self.meta.objects.items():
            obj = self.doc.getObject(name)
            om.valid = bool(obj is not None and obj.isValid())
            if om.valid and om.role == "feature":
                shp = obj.Shape
                om.valid = (not shp.isNull()) and len(shp.Solids) == 1 and shp.isValid() and shp.Volume > 1e-6

    # -- selection -----------------------------------------------------------

    def _select(self, arg: str) -> None:
        if arg == "Clear":
            self.sel_refs, self.meta.selection = [], []
            return
        ref = self.resolve_selection(arg)
        if ref is None:
            raise ActionError(f"nothing to select for {arg}")
        self.sel_refs, self.meta.selection = [ref], [arg]

    def resolve_selection(self, arg: str):
        if arg.startswith("Plane:"):
            if self.body is None:
                return None
            return (self.origin_feature(PLANE_ROLES[arg[6:]]).Name, [], arg)
        tip = self.tip()
        if arg == "Tip":
            return (tip.Name, [], arg) if tip is not None and tip.TypeId in SOLID_FEATURE_TYPES else None
        shape = self.solid()
        if shape is None:
            return None
        if arg.startswith("Face"):
            hit = pick_face(shape, arg[4:])
            return (tip.Name, [f"Face{hit[0] + 1}"], arg) if hit else None
        if arg == "Edges@Face+Z":
            hit = pick_face(shape, "+Z")
            return (tip.Name, edge_names(shape, hit[1].OuterWire.Edges), arg) if hit else None
        if arg == "Edges|Z":
            names = vertical_edges(shape)
            return (tip.Name, names, arg) if names else None
        return None

    def _consume_selection(self) -> None:
        self.sel_refs, self.meta.selection = [], []

    # -- executor helpers -----------------------------------------------------

    def intent_for(self, goal_ref: int | None = None):
        i = progress(self.goal, self.meta) if goal_ref is None or goal_ref < 0 else goal_ref
        return P.intent(self.goal, i)

    def add_feature(self, type_id: str, command: str, goal_ref: int):
        feat = self.body.newObject(type_id, command.split("_", 1)[1])
        self.meta.objects[feat.Name] = ObjMeta(feat.Name, "feature", goal_ref=goal_ref, command=command)
        return feat

    def finish_feature(self, feat) -> None:
        self.body.Tip = feat
        self._consume_selection()

    def open_profile(self):
        for name in reversed(list(self.meta.objects)):
            om = self.meta.objects[name]
            if om.role == "sketch" and om.consumed_by is None and om.geometry and self.meta.edit != name:
                return om
        return None


# ---------------------------------------------------------------------------
# Executors: action id -> fn(session, action) -> changed
# ---------------------------------------------------------------------------


def _need(cond, msg):
    if not cond:
        raise ActionError(msg)


def ex_body(s: HeadlessSession, a: str) -> bool:
    s._make_body()
    return True


def ex_new_sketch(s: HeadlessSession, a: str) -> bool:
    _need(s.body is not None and len(s.sel_refs) == 1, "select one plane or face first")
    obj_name, subs, arg = s.sel_refs[0]
    _need(arg.startswith(("Plane:", "Face")), "selection is not a plane or face")
    goal_ref = progress(s.goal, s.meta)
    sk = s.body.newObject("Sketcher::SketchObject", "Sketch")
    sk.AttachmentSupport = [(s.doc.getObject(obj_name), subs[0] if subs else "")]
    sk.MapMode = "FlatFace"
    s.doc.recompute()
    s.meta.objects[sk.Name] = ObjMeta(sk.Name, "sketch", goal_ref=goal_ref, command=a)
    s.meta.edit = sk.Name
    s._consume_selection()
    return True


def _edit_sketch(s: HeadlessSession):
    _need(s.meta.edit is not None, "not editing a sketch")
    return s.doc.getObject(s.meta.edit), s.meta.objects[s.meta.edit]


def _local(sk, om: ObjMeta, s: HeadlessSession, x: float, y: float) -> V:
    """Map intent coordinates to sketch-local coordinates. Sketches on body
    origin planes use intent coordinates directly; sketches on faces take
    global XY and project into the face's frame."""
    support = sk.AttachmentSupport[0][0] if sk.AttachmentSupport else None
    if support is not None and support.TypeId.startswith("App::"):
        return V(x, y, 0)
    z = sk.Placement.Base.z
    p = sk.Placement.inverse().multVec(V(x, y, z))
    return V(p.x, p.y, 0)


def ex_create(s: HeadlessSession, a: str) -> bool:
    sk, om = _edit_sketch(s)
    f = s.intent_for(om.goal_ref)
    cx, cy, sx, sy = P.profile_box(f, s.goal.scale)
    c = _local(sk, om, s, cx, cy)
    n0 = sk.GeometryCount
    if a == "Sketcher_CreateRectangle":
        x0, y0 = c.x - sx / 2, c.y - sy / 2
        pts = [V(x0, y0, 0), V(x0 + sx, y0, 0), V(x0 + sx, y0 + sy, 0), V(x0, y0 + sy, 0)]
        ids = sk.addGeometry([Part.LineSegment(pts[i], pts[(i + 1) % 4]) for i in range(4)], False)
        cons = [Sketcher.Constraint("Coincident", ids[i], 2, ids[(i + 1) % 4], 1) for i in range(4)]
        cons += [Sketcher.Constraint("Horizontal", ids[0]), Sketcher.Constraint("Horizontal", ids[2]),
                 Sketcher.Constraint("Vertical", ids[1]), Sketcher.Constraint("Vertical", ids[3])]
        sk.addConstraint(cons)
        group = {"kind": "rect", "geo": list(ids)}
    elif a == "Sketcher_CreateCircle":
        gid = sk.addGeometry(Part.Circle(c, V(0, 0, 1), min(sx, sy) / 2), False)
        group = {"kind": "circle", "geo": [gid]}
    elif a == "Sketcher_CreateHexagon":
        import ProfileLib.RegularPolygon as RP

        r = min(sx, sy) / 2
        ang = math.radians(240)  # first edge horizontal
        RP.makeRegularPolygon(sk, 6, c, c + V(r * math.cos(ang), r * math.sin(ang), 0), False)
        group = {"kind": "hex", "geo": list(range(n0, sk.GeometryCount))}
    elif a == "Sketcher_CreateLine":
        gid = sk.addGeometry(Part.LineSegment(c, c + V(sx / 2, sy / 3, 0)), False)
        group = {"kind": "line", "geo": [gid]}
    elif a == "Sketcher_CreatePoint":
        gid = sk.addGeometry(Part.Point(c), False)
        group = {"kind": "point", "geo": [gid]}
    else:
        raise ActionError(a)
    group["applied"] = []
    om.geometry.append(a)
    om.groups.append(group)
    return True


def _constraint_target(s: HeadlessSession, a: str):
    om = s.meta.objects.get(s.meta.edit) if s.meta.edit else None
    if om is None:
        return None
    slot, kinds = CONSTRAINT_SLOTS[a]
    return next((g for g in om.groups if g["kind"] in kinds and slot not in g["applied"]), None)


def ex_constrain(s: HeadlessSession, a: str) -> bool:
    sk, om = _edit_sketch(s)
    slot, _ = CONSTRAINT_SLOTS[a]
    group = _constraint_target(s, a)
    if group is None:
        return False  # nothing this constraint applies to: FreeCAD would just refuse
    geo = group["geo"]
    kind = group["kind"]
    cons = []
    if slot in ("dx", "dy"):
        edge = geo[0] if (slot == "dx" or kind == "line") else geo[1]
        g = sk.Geometry[edge]
        if slot == "dx":
            cons.append(Sketcher.Constraint("DistanceX", edge, 1, edge, 2, g.EndPoint.x - g.StartPoint.x))
        else:
            cons.append(Sketcher.Constraint("DistanceY", edge, 1, edge, 2, g.EndPoint.y - g.StartPoint.y))
    elif slot == "size":
        circ = geo[-1]
        r = sk.Geometry[circ].Radius
        cons.append(Sketcher.Constraint("Diameter", circ, 2 * r) if a.endswith("Diameter")
                    else Sketcher.Constraint("Radius", circ, r))
    elif slot == "lock":
        gid, pos = {"rect": (geo[0], 1), "line": (geo[0], 1), "point": (geo[0], 1)}.get(kind, (geo[-1], 3))
        p = sk.getPoint(gid, pos)
        cons += [Sketcher.Constraint("DistanceX", gid, pos, p.x), Sketcher.Constraint("DistanceY", gid, pos, p.y)]
    elif slot == "orient":
        cons.append(Sketcher.Constraint("Horizontal" if a.endswith("Horizontal") else "Vertical", geo[0]))
    sk.addConstraint(cons)
    if sk.solve() != 0:
        raise ActionError(f"{a}: solver reports conflicting/redundant constraints")
    group["applied"].append(slot)
    om.constraints.append(a)
    return True


def ex_toggle_construction(s: HeadlessSession, a: str) -> bool:
    sk, om = _edit_sketch(s)
    _need(sk.GeometryCount > 0, "no geometry")
    sk.toggleConstruction(sk.GeometryCount - 1)
    om.geometry.append(a)
    return True


def ex_profile_feature(s: HeadlessSession, a: str) -> bool:
    prof = s.open_profile()
    _need(prof is not None, "no unused sketch to use as profile")
    sk = s.doc.getObject(prof.name)
    f = s.intent_for(prof.goal_ref)
    scale = s.goal.scale
    type_id = {"PartDesign_Pad": "PartDesign::Pad", "PartDesign_Pocket": "PartDesign::Pocket",
               "PartDesign_Revolution": "PartDesign::Revolution", "PartDesign_Groove": "PartDesign::Groove",
               "PartDesign_Hole": "PartDesign::Hole"}[a]
    if a in ("PartDesign_Pocket", "PartDesign_Groove", "PartDesign_Hole"):
        _need(s.solid() is not None, "subtractive feature needs a solid")
    feat = s.add_feature(type_id, a, prof.goal_ref)
    feat.Profile = sk
    if a == "PartDesign_Pad":
        feat.Length = P.pad_length(f, scale)
    elif a == "PartDesign_Pocket":
        typ, length = P.pocket_spec(f, scale)
        feat.Type = typ
        if typ == "Length":
            feat.Length = length
    elif a in ("PartDesign_Revolution", "PartDesign_Groove"):
        feat.ReferenceAxis = (sk, ["V_Axis"])
        feat.Angle = 360
    elif a == "PartDesign_Hole":
        feat.Diameter = P.hole_diameter(f, scale)
        feat.DepthType = "ThroughAll"
    prof.consumed_by = feat.Name
    s.finish_feature(feat)
    return True


def ex_dressup(s: HeadlessSession, a: str) -> bool:
    _need(len(s.sel_refs) == 1 and s.sel_refs[0][1], "select edges/faces first")
    obj_name, subs, _ = s.sel_refs[0]
    goal_ref = progress(s.goal, s.meta)
    f = s.intent_for(goal_ref)
    type_id = {"PartDesign_Fillet": "PartDesign::Fillet", "PartDesign_Chamfer": "PartDesign::Chamfer",
               "PartDesign_Thickness": "PartDesign::Thickness", "PartDesign_Draft": "PartDesign::Draft"}[a]
    feat = s.add_feature(type_id, a, goal_ref)
    feat.Base = (s.doc.getObject(obj_name), subs)
    value = P.dressup_value(f, a)
    if a == "PartDesign_Fillet":
        feat.Radius = value
    elif a == "PartDesign_Chamfer":
        feat.Size = value
    elif a == "PartDesign_Thickness":
        feat.Value = value
    else:
        feat.Angle = value
    s.finish_feature(feat)
    return True


def ex_pattern(s: HeadlessSession, a: str) -> bool:
    _need(len(s.sel_refs) == 1 and s.sel_refs[0][2] == "Tip", "select a feature first")
    original = s.doc.getObject(s.sel_refs[0][0])
    goal_ref = progress(s.goal, s.meta)
    f = s.intent_for(goal_ref)
    spec = P.pattern_spec(f, a, s.goal.scale)
    type_id = {"PartDesign_Mirrored": "PartDesign::Mirrored", "PartDesign_LinearPattern": "PartDesign::LinearPattern",
               "PartDesign_PolarPattern": "PartDesign::PolarPattern"}[a]
    feat = s.add_feature(type_id, a, goal_ref)
    feat.Originals = [original]
    if a == "PartDesign_Mirrored":
        feat.MirrorPlane = (s.origin_feature("YZ_Plane"), [""])
    elif a == "PartDesign_LinearPattern":
        feat.Direction = (s.origin_feature("X_Axis"), [""])
        feat.Length = spec["length"]
        feat.Occurrences = spec["n"]
    else:
        feat.Axis = (s.origin_feature("Z_Axis"), [""])
        feat.Angle = 360
        feat.Occurrences = spec["n"]
    s.finish_feature(feat)
    return True


def ex_part_primitive(s: HeadlessSession, a: str) -> bool:
    obj = s.doc.addObject("Part::Box" if a == "Part_Box" else "Part::Cylinder", a.split("_", 1)[1])
    s.meta.objects[obj.Name] = ObjMeta(obj.Name, "other", command=a)
    return True


EXECUTORS = {
    "PartDesign_Body": ex_body,
    "PartDesign_NewSketch": ex_new_sketch,
    **{a: ex_create for a in ("Sketcher_CreateRectangle", "Sketcher_CreateCircle", "Sketcher_CreateHexagon",
                              "Sketcher_CreateLine", "Sketcher_CreatePoint")},
    **{a: ex_constrain for a in CONSTRAINT_SLOTS},
    "Sketcher_ToggleConstruction": ex_toggle_construction,
    **{a: ex_profile_feature for a in ("PartDesign_Pad", "PartDesign_Pocket", "PartDesign_Revolution",
                                       "PartDesign_Groove", "PartDesign_Hole")},
    **{a: ex_dressup for a in ("PartDesign_Fillet", "PartDesign_Chamfer", "PartDesign_Thickness", "PartDesign_Draft")},
    **{a: ex_pattern for a in ("PartDesign_Mirrored", "PartDesign_LinearPattern", "PartDesign_PolarPattern")},
    "Part_Box": ex_part_primitive,
    "Part_Cylinder": ex_part_primitive,
}

PRECHECKS = {a: (lambda s, a: _constraint_target(s, a) is not None) for a in CONSTRAINT_SLOTS}


# ---------------------------------------------------------------------------
# Snapshot: FreeCAD document -> State
# ---------------------------------------------------------------------------

TREE_TYPES = {"PartDesign::Body", "Sketcher::SketchObject", "Part::Box", "Part::Cylinder"} | SOLID_FEATURE_TYPES


def _sketch_numbers(sk) -> tuple[dict, dict, dict]:
    geo = {"line": 0, "circle": 0, "arc": 0, "point": 0, "other": 0}
    n_params = 0
    xs, ys = [], []
    n_constr = 0
    for i, g in enumerate(sk.Geometry):
        t = g.__class__.__name__
        n_params += GEO_PARAMS.get(t, 4)
        if sk.getConstruction(i):
            n_constr += 1
        key = {"LineSegment": "line", "Circle": "circle", "ArcOfCircle": "arc", "Point": "point"}.get(t, "other")
        geo[key] += 1
        if t == "LineSegment":
            xs += [g.StartPoint.x, g.EndPoint.x]
            ys += [g.StartPoint.y, g.EndPoint.y]
        elif t in ("Circle", "ArcOfCircle"):
            xs += [g.Center.x - g.Radius, g.Center.x + g.Radius]
            ys += [g.Center.y - g.Radius, g.Center.y + g.Radius]
        elif t == "Point":
            xs.append(g.X)
            ys.append(g.Y)
    cons = {k: 0 for k in CONSTRAINT_KINDS}
    removed = 0
    for c in sk.Constraints:
        t = c.Type
        point_lock = t in ("DistanceX", "DistanceY") and c.Second < -1999  # single-point form
        if point_lock:
            if t == "DistanceX":
                cons["Lock"] = cons.get("Lock", 0) + 1
        else:
            cons[t if t in cons else "Other"] += 1
        removed += CONSTRAINT_DOF.get(t, 1)
    wires = sk.Shape.Wires if not sk.Shape.isNull() else []
    closed = bool(wires) and all(w.isClosed() for w in wires)
    num = {
        "n_geo": len(sk.Geometry) / 10.0,
        "n_construction": n_constr / 10.0,
        "n_constraints": len(sk.Constraints) / 20.0,
        "dof": max(0, n_params - removed) / 10.0,
        "fully_constrained": float(bool(sk.FullyConstrained) and len(sk.Geometry) > 0),
        "closed": float(closed),
        "sk_w": (max(xs) - min(xs)) if xs else 0.0,
        "sk_h": (max(ys) - min(ys)) if ys else 0.0,
        "sk_cx": ((max(xs) + min(xs)) / 2) if xs else 0.0,
        "sk_cy": ((max(ys) + min(ys)) / 2) if ys else 0.0,
    }
    n = sk.Placement.Rotation.multVec(V(0, 0, 1))
    num.update(support_nx=n.x, support_ny=n.y, support_nz=n.z, support_offset=sk.Placement.Base.dot(n))
    support = sk.AttachmentSupport[0][0] if sk.AttachmentSupport else None
    num["on_face"] = float(support is not None and not support.TypeId.startswith("App::"))
    return num, geo, cons


def _feature_numbers(obj, scale: float) -> dict:
    num = {}
    t = obj.TypeId
    if hasattr(obj, "Length") and t in ("PartDesign::Pad", "PartDesign::Pocket", "PartDesign::LinearPattern"):
        num["length"] = float(obj.Length)
    if hasattr(obj, "Type") and t in ("PartDesign::Pad", "PartDesign::Pocket"):
        num["through_all"] = float(str(obj.Type) not in ("Length", "TwoLengths"))
    if t == "PartDesign::Hole":
        num["through_all"] = float(str(obj.DepthType) == "ThroughAll")
        num["radius"] = float(obj.Diameter) / 2
    if hasattr(obj, "Reversed") and t in ("PartDesign::Pad", "PartDesign::Pocket"):
        num["reversed"] = float(bool(obj.Reversed))
    if t in ("PartDesign::Revolution", "PartDesign::Groove", "PartDesign::PolarPattern"):
        num["angle"] = float(obj.Angle) / 360.0
    if t == "PartDesign::Fillet":
        num["radius"] = float(obj.Radius)
    if t == "PartDesign::Chamfer":
        num["radius"] = float(obj.Size)
    if t == "PartDesign::Thickness":
        num["radius"] = float(obj.Value)
    if t in ("PartDesign::LinearPattern", "PartDesign::PolarPattern"):
        num["occurrences"] = float(obj.Occurrences) / 10.0
    if hasattr(obj, "Base") and t in ("PartDesign::Fillet", "PartDesign::Chamfer", "PartDesign::Thickness", "PartDesign::Draft"):
        num["n_refs"] = len(obj.Base[1]) / 10.0 if obj.Base else 0.0
    if hasattr(obj, "Originals"):
        num["n_refs"] = len(obj.Originals) / 10.0
    shp = getattr(obj, "Shape", None)
    if shp is not None and not shp.isNull() and shp.Solids:
        num["volume_ratio"] = shp.Volume / max(scale, 1e-6) ** 3
        num["n_faces"] = len(shp.Faces) / 50.0
    return num


def describe_selection(obj, subs: list[str]) -> SelItem:
    """Describe one selected object (+ sub-elements) the same way for the
    headless session and the live GUI, so the model sees identical inputs."""
    subs = list(subs)
    item = SelItem(kind="other", object=obj.Name, object_type=obj.TypeId, subs=subs, count=len(subs))
    if obj.TypeId == "App::Plane" and not subs:
        n = obj.Placement.Rotation.multVec(V(0, 0, 1))
        item.kind, item.normal = "plane", (n.x, n.y, n.z)
    elif not subs and obj.TypeId in SOLID_FEATURE_TYPES:
        item.kind = "feature"
    elif not subs and obj.TypeId == "Sketcher::SketchObject":
        item.kind = "sketch"
    elif subs and subs[0].startswith("Face"):
        face = obj.Shape.getElement(subs[0])
        n = _face_normal(face)
        item.kind, item.normal, item.offset = "face", (n.x, n.y, n.z), face.CenterOfMass.dot(n)
    elif subs and subs[0].startswith("Edge"):
        item.kind = "edges"
        d = V(0, 0, 0)
        for sub in subs:
            e = obj.Shape.getElement(sub)
            ld = line_direction(e)
            if ld is not None:
                d = d + V(abs(ld.x), abs(ld.y), abs(ld.z))
        if d.Length > 0:
            d.normalize()
        item.normal = (d.x, d.y, d.z)
        item.offset = sum(obj.Shape.getElement(x).CenterOfMass.z for x in subs) / len(subs)
    return item


def selection_items(s) -> list[SelItem]:
    items = []
    for obj_name, subs, _arg in s.sel_refs:
        obj = s.doc.getObject(obj_name)
        if obj is not None:
            items.append(describe_selection(obj, subs))
    return items


def snapshot(s) -> State:
    """Extract the structured state. `s` is a HeadlessSession or GuiSession."""
    scale = s.goal.scale if s.goal else 1.0
    st = State(doc_open=s.doc is not None, workbench=s.workbench(), edit=s.edit_object(),
               undo_available=bool(s.undo_stack), recent=list(s.recent), events=list(s.events)[-8:])
    if s.doc is None:
        return st
    body = s.body
    st.has_body = body is not None
    tip = body.Tip if body is not None else None
    target_vol = s.goal.target.volume if (s.goal and s.goal.target.volume > 0) else scale ** 3
    index: dict[str, int] = {}
    for obj in s.doc.Objects:
        if obj.TypeId not in TREE_TYPES:
            continue
        parent = -1
        owner = obj.getParentGeoFeatureGroup() if obj.TypeId != "PartDesign::Body" else None
        if owner is not None and owner.Name in index:
            parent = index[owner.Name]
        node = Node(name=obj.Name, type=obj.TypeId, parent=parent, depth=0 if parent < 0 else 1)
        num = {"tip": float(obj == tip), "in_edit": float(obj.Name == st.edit), "valid": float(obj.isValid()),
               "visible": float(bool(obj.Visibility)), "active_body": float(body is not None and obj == body)}
        if obj.TypeId == "Sketcher::SketchObject":
            sk_num, node.geo, node.cons = _sketch_numbers(obj)
            num.update(sk_num)
            num["consumed"] = float(any(getattr(o, "Profile", None) and o.Profile[0] == obj for o in obj.InList))
        elif obj.TypeId != "PartDesign::Body":
            num.update(_feature_numbers(obj, scale))
            if "volume_ratio" in num:
                num["volume_ratio"] = num["volume_ratio"] * scale ** 3 / target_vol
        node.num = num
        index[obj.Name] = len(st.tree)
        st.tree.append(node)
    st.selection = selection_items(s)
    st.shape = shape_info(s.solid()) if body is not None else ShapeInfo()
    return st

"""State/goal/action -> token arrays (numpy), and batch collation (torch).

The state is encoded as a *sequence of typed tokens*, not a flat vector:

    [CLS] [GLOBAL] [NODE x tree order ...] [SEL ...] [RECENT x recency ...]
    [GOAL_GLOBAL] [GOAL_FEAT x goal order ...]

Every token carries a segment id, a primary categorical id (node type,
selection kind, action id, goal kind, workbench), a secondary categorical id
(tree depth / selected object type), a position (tree index, recency, goal
order) and a segment-specific numeric vector. Feature-tree nodes keep their
document order and parent depth so the encoder sees the tree as a sequence.

Candidate actions are encoded independently of the state: catalogue id,
category, scope, word pieces of the command name, and an argument vector.

UI-level models (`ui=True`, see freecad_s1/ui) act on interface elements
(`cmd:PartDesign_Pad`, `set:lengthEdit`, `opt:changeMode=Through all`, ...).
An element is encoded through the command it stands for (if any), the word
pieces of its full id, its role, and a small vector of live widget values
from `State.ui` (the number in a field, whether a combo entry is current or a
check box is checked). Recent-action tokens carry the element's role.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..actions import (ACTION_IDS, CATALOGUE, CATEGORIES, SCOPES, SOLID_FEATURE_TYPES, WORD_VOCAB, action_vector,
                       action_words)
from ..ui import spec as UIS
from ..schema import (
    CONSTRAINT_KINDS, GEOMETRY_KINDS, GOAL_KINDS, GOAL_LENGTH_KEYS, GOAL_PARAM_KEYS, LENGTH_KEYS,
    NODE_NUM_KEYS, NODE_TYPES, SELECTION_KINDS, WORKBENCHES, Goal, State,
)

SEG_CLS, SEG_GLOBAL, SEG_NODE, SEG_SEL, SEG_RECENT, SEG_GOAL_GLOBAL, SEG_GOAL = range(7)
N_SEGMENTS = 7
NUM_DIM = len(NODE_NUM_KEYS) + len(GEOMETRY_KINDS) + len(CONSTRAINT_KINDS)
MAX_POS = 128  # position-embedding table; randomized positions sample from this range
MAX_ORD = 32  # ordinal-embedding table (0 = no ordinal)
MAX_NODES = 40
MAX_SEL = 4
MAX_GOAL = 24
MAX_WORDS = 8
ACT_VEC_DIM = 6
UI_VEC_DIM = 6
UI_ROLES = UIS.ROLES
_ROLE = {r: i for i, r in enumerate(UI_ROLES)}
UI_FIELDS = sorted({f.name for fields in UIS.DIALOG_FIELDS.values() for f in fields})  # one-hot in recent tokens
_FIELD = {n: i for i, n in enumerate(UI_FIELDS)}
UI_WORD_VOCAB = WORD_VOCAB + sorted({w for e in UIS.UI_ID_EXAMPLES for w in action_words(e)} - set(WORD_VOCAB))
_UI_WORD = {w: i for i, w in enumerate(UI_WORD_VOCAB)}

# Primary-id vocab per segment, laid out in one shared embedding table.
_A_SIZES = [1, len(WORKBENCHES), len(NODE_TYPES), len(SELECTION_KINDS), len(ACTION_IDS), 1, len(GOAL_KINDS)]
A_OFFSETS = np.cumsum([0] + _A_SIZES[:-1]).tolist()
A_VOCAB = int(sum(_A_SIZES))
B_VOCAB = len(NODE_TYPES) + 4  # depth buckets share the table with object types

_NODE_TYPE = {t: i for i, t in enumerate(NODE_TYPES)}
_WB = {w: i for i, w in enumerate(WORKBENCHES)}
_SEL = {k: i for i, k in enumerate(SELECTION_KINDS)}
_ACT = {a: i for i, a in enumerate(ACTION_IDS)}
_GOAL = {k: i for i, k in enumerate(GOAL_KINDS)}
_CAT = {c: i for i, c in enumerate(CATEGORIES)}
_SCOPE = {c: i for i, c in enumerate(SCOPES)}
_WORD = {w: i for i, w in enumerate(WORD_VOCAB)}
FACE_DIRS = ["+Z", "-Z", "+X", "-X", "+Y", "-Y", "|Z"]


@dataclass
class Tokens:
    seg: np.ndarray  # (L,) int8
    a: np.ndarray  # (L,) int16  (already offset into A_VOCAB)
    b: np.ndarray  # (L,) int16
    pos: np.ndarray  # (L,) int8
    num: np.ndarray  # (L, NUM_DIM) float16
    ord: np.ndarray  # (L,) int8: coupled ordinal, 0 = none (see encode_state)


def _clip(x: float, lim: float = 8.0) -> float:
    return float(max(-lim, min(lim, x)))


def encode_state(state: State, goal: Goal, invariant: bool = False, sentinel: bool = False, ui: bool = False,
                 ui_recent: bool = False) -> Tokens:
    """`invariant` (H5): drop numeric features whose magnitude grows with the
    length of the build (tree size, number of intents, intents remaining) and
    express face/edge counts relative to the target, so longer goals do not
    push numeric inputs outside the training range.

    `sentinel`: append an END intent token (kind id 0, ordinal n+1) after the
    goal list, so "everything is built" is represented like any other intent
    (used by the modular architecture's pointer).

    `ui`: recent actions are UI element ids; their tokens get the element's
    role (an extra `b` range after B_VOCAB, only in UI models). With
    `ui_recent`, a recent dialog edit also says which field it touched
    (one-hot over UI_FIELDS), so the model can tell what is filled in."""
    scale = goal.scale if goal.scale > 0 else 1.0
    tgt_vol = goal.target.volume if goal.target.volume > 0 else scale ** 3
    rows: list[tuple[int, int, int, int, list[float]]] = []
    ords: list[int] = []

    rows.append((SEG_CLS, 0, 0, 0, []))

    sh = state.shape
    t = goal.target
    if invariant:
        faces = _clip(sh.n_faces / max(t.n_faces, 1), 4.0)
        edges = _clip(sh.n_edges / max(t.n_edges, 1), 4.0)
        tree_len = 0.0
    else:
        faces, edges, tree_len = sh.n_faces / 50, sh.n_edges / 100, len(state.tree) / 10
    g = [float(state.doc_open), float(state.has_body), float(state.edit is not None), float(state.undo_available),
         len(state.selection) / 4, float(sh.valid), _clip(sh.volume / tgt_vol),
         *(_clip(x / scale) for x in sh.bbox), faces, edges, float(sh.n_solids),
         *(float(d in sh.face_dirs) for d in FACE_DIRS), tree_len,
         float(bool(state.ui and state.ui.get("dialog")))]  # a task dialog is open (always 0 for command-level states)
    rows.append((SEG_GLOBAL, _WB.get(state.workbench, 0), 0, 0, g))

    # Coupled ordinals (cf. position coupling / index hints): goal intent k and
    # the tree objects that would realize it share ordinal k+1. For a tree
    # node the ordinal is 1 + the number of solid features before it, so the
    # k-th feature and the sketch it consumes line up with intent k. Computed
    # from observable state only.
    node_ords = []
    n_solid = 0
    for node in state.tree[:MAX_NODES]:
        if node.type in SOLID_FEATURE_TYPES:
            node_ords.append(min(n_solid + 1, MAX_ORD - 1))
            n_solid += 1
        elif node.type == "Sketcher::SketchObject":
            node_ords.append(min(n_solid + 1, MAX_ORD - 1))
        else:
            node_ords.append(0)

    for i, node in enumerate(state.tree[:MAX_NODES]):
        num = [_clip(node.num.get(k, 0.0) / (scale if k in LENGTH_KEYS else 1.0)) for k in NODE_NUM_KEYS]
        num += [node.geo.get(k, 0) / 10 for k in GEOMETRY_KINDS]
        num += [node.cons.get(k, 0) / 10 for k in CONSTRAINT_KINDS]
        if invariant and "n_faces" in node.num:
            num[NODE_NUM_KEYS.index("n_faces")] = _clip(node.num["n_faces"] * 50 / max(t.n_faces, 1), 4.0)
        b = len(NODE_TYPES) + min(node.depth, 3)
        rows.append((SEG_NODE, _NODE_TYPE.get(node.type, 0), b, min(i, MAX_POS - 1), num))
        ords.append(len(rows) - 1)

    for i, sel in enumerate(state.selection[:MAX_SEL]):
        num = [*sel.normal, _clip(sel.offset / scale), sel.count / 10]
        rows.append((SEG_SEL, _SEL.get(sel.kind, _SEL["other"]), _NODE_TYPE.get(sel.object_type, 0), i, num))

    for i, act in enumerate(reversed(state.recent)):
        if ui:
            r = UIS.role(act)
            num = []
            if ui_recent and r in ("set", "opt", "toggle") and UIS.field_name(act) in _FIELD:
                num = [0.0] * len(UI_FIELDS)
                num[_FIELD[UIS.field_name(act)]] = 1.0
            rows.append((SEG_RECENT, _ACT.get(UIS.underlying(act) or act, 0), B_VOCAB + _ROLE.get(r, 0), i, num))
        else:
            rows.append((SEG_RECENT, _ACT.get(act, 0), 0, i, []))

    if invariant:
        gg = [*(_clip(x / scale) for x in t.bbox), _clip(t.volume / scale ** 3)]
    else:
        gg = [*(_clip(x / scale) for x in t.bbox), _clip(t.volume / scale ** 3), t.n_faces / 50, t.n_edges / 100,
              len(goal.features) / 10]
    rows.append((SEG_GOAL_GLOBAL, 0, 0, 0, gg))
    goal_rows: list[int] = []

    for i, f in enumerate(goal.features[:MAX_GOAL]):
        num = []
        for k in GOAL_PARAM_KEYS:
            v = f.params.get(k)
            num.append(0.0 if v is None else _clip(v / scale if k in GOAL_LENGTH_KEYS else v / 10))
        num += [float(k in f.params) for k in GOAL_PARAM_KEYS]
        # Relative order, so "which intent is next / is this the last one"
        # does not hinge on absolute position embeddings seen in training.
        n_goal = len(goal.features)
        if sentinel:  # modular mode: order comes from the coupled ordinal only, so an
            num += [0.0, 0.0, 0.0]  # intent's features never depend on the goal's length
        else:
            num += [i / max(n_goal - 1, 1), float(i == n_goal - 1), 0.0 if invariant else (n_goal - 1 - i) / 10]
        rows.append((SEG_GOAL, _GOAL.get(f.kind, 0), 0, min(i, MAX_POS - 1), num))
        goal_rows.append(len(rows) - 1)
    if sentinel:
        end_num = [0.0] * (2 * len(GOAL_PARAM_KEYS)) + [1.0, 1.0, 0.0]
        rows.append((SEG_GOAL, 0, 0, min(len(goal_rows), MAX_POS - 1), end_num))
        goal_rows.append(len(rows) - 1)

    n = len(rows)
    ordinal = np.zeros(n, np.int8)
    for row, o in zip(ords, node_ords):
        ordinal[row] = o
    for k, row in enumerate(goal_rows):
        ordinal[row] = min(k + 1, MAX_ORD - 1)
    seg = np.zeros(n, np.int8)
    a = np.zeros(n, np.int16)
    b = np.zeros(n, np.int16)
    pos = np.zeros(n, np.int8)
    num = np.zeros((n, NUM_DIM), np.float16)
    for j, (s, ai, bi, p, v) in enumerate(rows):
        seg[j], a[j], b[j], pos[j] = s, A_OFFSETS[s] + ai, bi, p
        if v:
            num[j, : len(v)] = v
    return Tokens(seg, a, b, pos, num, ordinal)


_ACTION_CACHE: dict[tuple[str, bool], tuple[int, int, int, list[int], list[float], int]] = {}


def encode_action(action_id: str, ui: bool = False) -> tuple[int, int, int, list[int], list[float], int]:
    """(id, category, scope, word ids[MAX_WORDS], arg vector, role) for one
    action. Unknown commands (e.g. from a live GUI) keep their word pieces.
    UI element ids are encoded through the command they stand for."""
    hit = _ACTION_CACHE.get((action_id, ui))
    if hit is not None:
        return hit
    base = (UIS.underlying(action_id) or action_id) if ui else action_id
    spec = CATALOGUE.get(base)
    vocab = _UI_WORD if ui else _WORD
    words = [vocab.get(w, 1) for w in action_words(action_id)][:MAX_WORDS]
    words += [0] * (MAX_WORDS - len(words))
    role = _ROLE.get(UIS.role(action_id), 0) if ui else 0
    out = (_ACT.get(base, 0), _CAT.get(spec.category, 0) if spec else 0,
           _SCOPE.get(spec.scope, 0) if spec else 0, words, action_vector(base), role)
    _ACTION_CACHE[(action_id, ui)] = out
    return out


def ui_vector(element: str, ui_state: dict | None, scale: float) -> list[float]:
    """Live widget values for a dialog element: [mm value / scale, degrees /
    360, count / 10, checked, current option, is a dialog widget]."""
    vec = [0.0] * UI_VEC_DIM
    role = UIS.role(element)
    if role not in ("set", "opt", "toggle", "click"):
        return vec
    vec[5] = 1.0
    fld = (ui_state or {}).get("fields", {}).get(UIS.field_name(element)) if role != "click" else None
    if fld is None:
        return vec
    value = fld.get("value")
    if role == "set" and isinstance(value, (int, float)):
        unit = fld.get("unit")
        if unit == "deg":
            vec[1] = _clip(value / 360.0)
        elif unit == "count":
            vec[2] = _clip(value / 10.0)
        else:
            vec[0] = _clip(value / scale)
    elif role == "toggle":
        vec[3] = float(bool(value))
    elif role == "opt":
        vec[4] = float(value == UIS.option_text(element))
    return vec


def encode_actions(actions: list[str], ui: bool = False, ui_state: dict | None = None,
                   scale: float = 1.0) -> dict[str, np.ndarray]:
    enc = [encode_action(a, ui) for a in actions]
    out = {
        "id": np.array([e[0] for e in enc], np.int16),
        "cat": np.array([e[1] for e in enc], np.int8),
        "scope": np.array([e[2] for e in enc], np.int8),
        "words": np.array([e[3] for e in enc], np.int16).reshape(len(enc), MAX_WORDS),
        "vec": np.array([e[4] for e in enc], np.float32).reshape(len(enc), ACT_VEC_DIM),
    }
    if ui:
        out["role"] = np.array([e[5] for e in enc], np.int8)
        out["ui"] = np.array([ui_vector(a, ui_state, scale) for a in actions], np.float32).reshape(len(enc), UI_VEC_DIM)
    return out


@dataclass
class Example:
    tokens: Tokens
    actions: dict[str, np.ndarray]
    target: np.ndarray  # (N,) bool, acceptable actions
    progress: int = -1  # index of the goal intent being worked on (== #intents when done); -1 unknown


def make_example(state: State, goal: Goal, actions: list[str], acceptable: list[str] | None = None,
                 progress: int | None = None, invariant: bool = False, sentinel: bool = False,
                 ui: bool = False, ui_recent: bool = False) -> Example:
    acc = set(acceptable or [])
    scale = goal.scale if goal.scale > 0 else 1.0
    return Example(encode_state(state, goal, invariant, sentinel, ui, ui_recent),
                   encode_actions(actions, ui, state.ui, scale),
                   np.array([a in acc for a in actions], dtype=bool), -1 if progress is None else int(progress))


def collate(examples: list[Example]):
    """Pad a list of examples into a dict of torch tensors."""
    import torch

    bsz = len(examples)
    L = max(len(e.tokens.seg) for e in examples)
    N = max(len(e.target) for e in examples)
    seg = np.zeros((bsz, L), np.int64)
    a = np.zeros((bsz, L), np.int64)
    b = np.zeros((bsz, L), np.int64)
    pos = np.zeros((bsz, L), np.int64)
    num = np.zeros((bsz, L, NUM_DIM), np.float32)
    ordinal = np.zeros((bsz, L), np.int64)
    tmask = np.zeros((bsz, L), bool)
    # Progress label as a pointer target over token slots: the k-th goal
    # token, or column L (the END slot) once every intent is built.
    progress = np.full(bsz, -1, np.int64)
    aid = np.zeros((bsz, N), np.int64)
    cat = np.zeros((bsz, N), np.int64)
    scope = np.zeros((bsz, N), np.int64)
    words = np.zeros((bsz, N, MAX_WORDS), np.int64)
    vec = np.zeros((bsz, N, ACT_VEC_DIM), np.float32)
    role = np.zeros((bsz, N), np.int64)
    uivec = np.zeros((bsz, N, UI_VEC_DIM), np.float32)
    amask = np.zeros((bsz, N), bool)
    target = np.zeros((bsz, N), bool)
    for i, e in enumerate(examples):
        t, n = len(e.tokens.seg), len(e.target)
        seg[i, :t], a[i, :t], b[i, :t], pos[i, :t] = e.tokens.seg, e.tokens.a, e.tokens.b, e.tokens.pos
        num[i, :t] = e.tokens.num
        ordinal[i, :t] = e.tokens.ord
        tmask[i, :t] = True
        if e.progress >= 0:
            goal_slots = np.flatnonzero(e.tokens.seg == SEG_GOAL)
            progress[i] = goal_slots[e.progress] if e.progress < len(goal_slots) else L
        aid[i, :n], cat[i, :n], scope[i, :n] = e.actions["id"], e.actions["cat"], e.actions["scope"]
        words[i, :n], vec[i, :n] = e.actions["words"], e.actions["vec"]
        if "role" in e.actions:
            role[i, :n], uivec[i, :n] = e.actions["role"], e.actions["ui"]
        amask[i, :n] = True
        target[i, :n] = e.target
    to = torch.from_numpy
    return {
        "seg": to(seg), "a": to(a), "b": to(b), "pos": to(pos), "num": to(num), "token_mask": to(tmask),
        "ord": to(ordinal), "progress": to(progress),
        "act_id": to(aid), "act_cat": to(cat), "act_scope": to(scope), "act_words": to(words),
        "act_vec": to(vec), "act_role": to(role), "act_ui": to(uivec), "action_mask": to(amask), "target": to(target),
    }

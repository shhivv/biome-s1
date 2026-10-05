"""Goal specifications, per-feature command recipes and a curriculum sampler.

A goal is an ordered list of feature intents (what a System-2 planner would
hand down: "a 40x30x10 plate, a Ø6 through-hole at (10, 0), polar x6, fillet
the top edges r=1"). Each intent expands into a short burst of low-level
commands described by its `Recipe`; turning intents into command sequences —
and knowing where in that sequence the session currently is — is the
System-1 model's job.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

from .schema import Goal, GoalFeature

RECT_CONSTRAINTS = ("Sketcher_ConstrainDistanceX", "Sketcher_ConstrainDistanceY", "Sketcher_ConstrainLock")
CIRCLE_CONSTRAINTS = ("Sketcher_ConstrainDiameter", "Sketcher_ConstrainLock")
HEX_CONSTRAINTS = ("Sketcher_ConstrainDiameter", "Sketcher_ConstrainLock", "Sketcher_ConstrainHorizontal")


@dataclass(frozen=True)
class Recipe:
    feature: str  # the PartDesign command that completes this intent
    support: str | None = None  # Select arg for the sketch support (sketch-based intents)
    geometry: str | None = None  # Sketcher create command
    constraints: tuple[str, ...] = ()  # constraint commands, any order
    select: str | None = None  # Select arg required right before `feature` (non-sketch intents)

    @property
    def sketched(self) -> bool:
        return self.geometry is not None


RECIPES: dict[str, Recipe] = {
    "base_box": Recipe("PartDesign_Pad", "Plane:XY", "Sketcher_CreateRectangle", RECT_CONSTRAINTS),
    "base_cyl": Recipe("PartDesign_Pad", "Plane:XY", "Sketcher_CreateCircle", CIRCLE_CONSTRAINTS),
    "base_hex": Recipe("PartDesign_Pad", "Plane:XY", "Sketcher_CreateHexagon", HEX_CONSTRAINTS),
    "base_ring": Recipe("PartDesign_Revolution", "Plane:XZ", "Sketcher_CreateRectangle", RECT_CONSTRAINTS),
    "boss_cyl": Recipe("PartDesign_Pad", "Face+Z", "Sketcher_CreateCircle", CIRCLE_CONSTRAINTS),
    "boss_box": Recipe("PartDesign_Pad", "Face+Z", "Sketcher_CreateRectangle", RECT_CONSTRAINTS),
    "hole": Recipe("PartDesign_Pocket", "Face+Z", "Sketcher_CreateCircle", CIRCLE_CONSTRAINTS),
    "hole_std": Recipe("PartDesign_Hole", "Face+Z", "Sketcher_CreateCircle", CIRCLE_CONSTRAINTS),
    "pocket_rect": Recipe("PartDesign_Pocket", "Face+Z", "Sketcher_CreateRectangle", RECT_CONSTRAINTS),
    "polar_pattern": Recipe("PartDesign_PolarPattern", select="Tip"),
    "linear_pattern": Recipe("PartDesign_LinearPattern", select="Tip"),
    "mirror": Recipe("PartDesign_Mirrored", select="Tip"),
    "fillet_top": Recipe("PartDesign_Fillet", select="Edges@Face+Z"),
    "fillet_vertical": Recipe("PartDesign_Fillet", select="Edges|Z"),
    "chamfer_top": Recipe("PartDesign_Chamfer", select="Edges@Face+Z"),
    "shell": Recipe("PartDesign_Thickness", select="Face+Z"),
}


@dataclass
class StartSpec:
    """Initial GUI/document condition for an episode."""

    doc_open: bool = True
    workbench: str = "PartDesignWorkbench"
    body: bool = False


def _u(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 1)


def _base(rng: random.Random, kinds: list[str]) -> GoalFeature:
    kind = rng.choice(kinds)
    if kind == "base_box":
        return GoalFeature(kind, {"w": _u(rng, 20, 80), "d": _u(rng, 20, 80), "h": _u(rng, 5, 30)})
    if kind in ("base_cyl", "base_hex"):
        return GoalFeature(kind, {"r": _u(rng, 12, 40), "h": _u(rng, 5, 30)})
    ri = _u(rng, 5, 25)
    return GoalFeature(kind, {"ri": ri, "ro": round(ri + _u(rng, 4, 15), 1), "h": _u(rng, 4, 25)})


def _half_extents(base: GoalFeature) -> tuple[float, float]:
    """Half-size of the region on the base top face where features may sit."""
    p = base.params
    if base.kind == "base_box":
        return p["w"] / 2, p["d"] / 2
    return p["r"] * 0.6, p["r"] * 0.6  # square inscribed-ish in circle / hexagon


def _min_dim(base: GoalFeature) -> float:
    p = base.params
    if base.kind == "base_box":
        return min(p["w"], p["d"], p["h"])
    if base.kind == "base_ring":
        return min(p["ro"] - p["ri"], p["h"])
    return min(2 * p["r"], p["h"])


def _on_top(rng: random.Random, kind: str, base: GoalFeature, x_sign: float = 0.0) -> GoalFeature:
    hx, hy = _half_extents(base)
    h = base.params["h"]
    if kind in ("boss_cyl", "hole", "hole_std"):
        r = _u(rng, 1.5, max(1.6, min(hx, hy) * 0.35))
        x = _u(rng, -(hx - r) * 0.8, (hx - r) * 0.8)
        if x_sign:
            x = _u(rng, r + 1, max(r + 1.1, (hx - r) * 0.8)) * x_sign
        y = _u(rng, -(hy - r) * 0.8, (hy - r) * 0.8)
        params = {"r": r, "x": x, "y": y}
        if kind == "boss_cyl":
            params["h"] = _u(rng, 2, 20)
        return GoalFeature(kind, params)
    # boss_box / pocket_rect
    w = _u(rng, 2, max(2.1, hx * 0.8))
    d = _u(rng, 2, max(2.1, hy * 0.8))
    x = _u(rng, -(hx - w / 2) * 0.8, (hx - w / 2) * 0.8)
    if x_sign:
        x = _u(rng, w / 2 + 1, max(w / 2 + 1.1, (hx - w / 2) * 0.8)) * x_sign
    y = _u(rng, -(hy - d / 2) * 0.8, (hy - d / 2) * 0.8)
    params = {"w": w, "d": d, "x": x, "y": y}
    if kind == "boss_box":
        params["h"] = _u(rng, 2, 20)
    else:
        params["depth"] = _u(rng, 1, h * 0.7)
    return GoalFeature(kind, params)


def _dressup(rng: random.Random, kind: str, base: GoalFeature) -> GoalFeature:
    m = _min_dim(base)
    if kind in ("fillet_top", "fillet_vertical"):
        return GoalFeature(kind, {"r": _u(rng, 0.5, max(0.6, m * 0.2))})
    if kind == "chamfer_top":
        return GoalFeature(kind, {"size": _u(rng, 0.5, max(0.6, m * 0.2))})
    return GoalFeature("shell", {"t": _u(rng, 0.8, max(0.9, m * 0.15))})


def _scale(base: GoalFeature) -> float:
    p = base.params
    if base.kind == "base_box":
        return max(p["w"], p["d"], p["h"])
    if base.kind == "base_ring":
        return max(2 * p["ro"], p["h"])
    return max(2 * p["r"], p["h"])


def sample_goal(level: int, rng: random.Random) -> Goal:
    """Level 1: one base feature. Level 2: base + one secondary feature.
    Level 3: base + 2-4 secondary features incl. patterns and mirrors.
    Level 4 (held out of training): boss + two cut groups + dressup, >= 6 features."""
    if level <= 1:
        base = _base(rng, ["base_box", "base_cyl", "base_hex", "base_ring"])
        return Goal([base], level=1, scale=_scale(base))

    base = _base(rng, ["base_box", "base_box", "base_cyl", "base_hex", "base_ring"] if level == 2 else
                 ["base_box", "base_box", "base_cyl", "base_hex"])
    feats = [base]
    if level == 2:
        if base.kind == "base_ring":
            feats.append(_dressup(rng, rng.choice(["fillet_top", "chamfer_top"]), base))
        else:
            kinds = ["boss_cyl", "boss_box", "hole", "hole_std", "pocket_rect", "fillet_top", "chamfer_top", "shell"]
            if base.kind == "base_box":
                kinds.append("fillet_vertical")
            k = rng.choice(kinds)
            feats.append(_on_top(rng, k, base) if k in ("boss_cyl", "boss_box", "hole", "hole_std", "pocket_rect")
                         else _dressup(rng, k, base))
        return Goal(feats, level=2, scale=_scale(base))

    # Level 3: 0-1 boss and 1-2 cut groups in random order (<= 4 secondary
    # features), then maybe one dressup. Randomized structure forces the policy
    # to read the goal instead of memorizing templates. Level 4 is held out of
    # training: boss + two cut groups + dressup, always >= 6 features, i.e.
    # longer than anything in the training distribution.
    for _ in range(50):
        goal = _composite(rng, base, level)
        n = len(goal.features)
        if level <= 3 and n <= 5 or level == 4 and n >= 6 or level == 5 and n >= 8 or level >= 6 and n >= 2 * level - 1:
            return goal
    return goal


def _composite(rng: random.Random, base: GoalFeature, level: int) -> Goal:
    components: list[tuple[list[GoalFeature], bool]] = []
    if level >= 4 or rng.random() < 0.5:
        components.append(([_on_top(rng, rng.choice(["boss_cyl", "boss_box"]), base)], False))
    # cut groups: level 4 -> 2, 5 -> 3, 6 -> 4 (11 intents), 7 -> 5 (13), 8 -> 6 (15), 9 -> 7 (17)
    for _ in range(level - 2 if level >= 4 else 2 if rng.random() < 0.35 else 1):
        components.append(_cut_group(rng, base))
    if level <= 3:
        rng.shuffle(components)
    feats = [base]
    has_cut = False
    for group, cut in components:
        if level <= 3 and len(feats) - 1 + len(group) > 4:
            continue
        feats += group
        has_cut = has_cut or cut
    if level >= 4 or (len(feats) < 5 and rng.random() < 0.7):
        options = ["fillet_top", "chamfer_top"] + (["fillet_vertical"] if base.kind == "base_box" else [])
        if not has_cut:
            options.append("shell")
        feats.append(_dressup(rng, rng.choice(options), base))
    return Goal(feats, level=level, scale=_scale(base))


def _cut_group(rng: random.Random, base: GoalFeature) -> tuple[list[GoalFeature], bool]:
    """A hole/pocket, optionally patterned or mirrored (or a mirrored boss)."""
    pattern_choice = rng.random()
    if base.kind in ("base_cyl", "base_hex") and pattern_choice < 0.5:
        rho = base.params["r"] * _u(rng, 0.45, 0.65)
        n = rng.randint(3, 8)
        r = _u(rng, 1.0, max(1.1, min(rho * math.sin(math.pi / n) * 0.6, base.params["r"] * 0.15)))
        return [GoalFeature(rng.choice(["hole", "hole_std"]), {"r": r, "x": round(rho, 1), "y": 0.0}),
                GoalFeature("polar_pattern", {"n": float(n)})], True
    if base.kind == "base_box" and pattern_choice < 0.35:
        w, d = base.params["w"], base.params["d"]
        n = rng.randint(2, 4)
        r = _u(rng, 1.0, max(1.1, min(d * 0.12, w / (4 * n))))
        x0 = round(-w / 2 + w * 0.2, 1)
        length = round(w * 0.6, 1)
        return [GoalFeature(rng.choice(["hole", "hole_std"]), {"r": r, "x": x0, "y": _u(rng, -d * 0.2, d * 0.2)}),
                GoalFeature("linear_pattern", {"n": float(n), "length": length})], True
    if pattern_choice < 0.75:
        k = rng.choice(["hole", "hole_std", "pocket_rect", "boss_cyl"])
        return [_on_top(rng, k, base, x_sign=1.0), GoalFeature("mirror", {})], k != "boss_cyl"
    k = rng.choice(["hole", "hole_std", "pocket_rect"])
    return [_on_top(rng, k, base)], True


# ---------------------------------------------------------------------------
# Generalization splits. Every feature kind occurs in training; the rules
# below name combinations and adjacent pairs that never do. The first two are
# excluded from training by filtering; the last two are pairs the training
# sampler cannot produce at all (it only mirrors hole/hole_std/pocket_rect/
# boss_cyl and only patterns hole/hole_std).
# ---------------------------------------------------------------------------

PATTERNS = {"polar_pattern", "linear_pattern"}


def heldout_composition(goal: Goal) -> str | None:
    """Name of the held-out composition rule `goal` matches, if any."""
    kinds = [f.kind for f in goal.features]
    if "boss_box" in kinds and PATTERNS & set(kinds):
        return "pattern+boss_box"  # patterns seen with boss_cyl, boss_box seen without patterns
    for a, b in zip(kinds, kinds[1:]):
        if a == "hole_std" and b == "mirror":
            return "mirrored_hole_std"  # mirror seen on hole/pocket/boss, hole_std seen patterned
        if a == "boss_box" and b == "mirror":
            return "mirrored_boss_box"  # suite comp2
        if a == "pocket_rect" and b in PATTERNS:
            return "patterned_pocket_rect"  # suite comp3
    return None


SPLITS = ("train", "iid", "comp", "comp2", "comp3", "len")


def goal_in_split(goal: Goal, split: str) -> bool:
    """train/iid: levels 1-3 without held-out compositions; comp: level-3
    goals matching a held-out rule; len: level-4 (6-7 features) goals without
    held-out compositions."""
    rule = heldout_composition(goal)
    if split in ("train", "iid"):
        return goal.level <= 3 and rule is None
    if split == "comp":
        return goal.level == 3 and rule is not None
    if split == "len":
        return goal.level >= 4 and rule is None
    raise ValueError(split)


def _mirrored_boss_box_goal(rng: random.Random) -> Goal:
    """Confirmation split "comp2": boss_box followed by mirror, a pair the
    training sampler never produces (_cut_group only mirrors hole, hole_std,
    pocket_rect and boss_cyl). Optional plain hole and dressup; <= 5 intents."""
    base = _base(rng, ["base_box", "base_box", "base_cyl", "base_hex"])
    feats = [base]
    if rng.random() < 0.5:
        feats.append(_on_top(rng, rng.choice(["hole", "pocket_rect"]), base))
    feats += [_on_top(rng, "boss_box", base, x_sign=1.0), GoalFeature("mirror", {})]
    if len(feats) < 5 and rng.random() < 0.7:
        feats.append(_dressup(rng, rng.choice(["fillet_top", "chamfer_top"]), base))
    return Goal(feats, level=3, scale=_scale(base))


def _patterned_pocket_goal(rng: random.Random) -> Goal:
    """Suite "comp3": a rectangular pocket followed by a polar pattern, a pair
    the training sampler never produces. Optional boss_cyl before and dressup
    after; <= 5 intents."""
    base = _base(rng, ["base_cyl", "base_hex"])
    r = base.params["r"]
    feats = [base]
    if rng.random() < 0.5:
        feats.append(_on_top(rng, "boss_cyl", base))
    n = rng.randint(3, 8)
    rho = round(r * _u(rng, 0.45, 0.65), 1)
    side = max(1.0, min(2 * rho * math.sin(math.pi / n) * 0.5, r * 0.25))
    feats.append(GoalFeature("pocket_rect", {"w": _u(rng, 1.0, side), "d": _u(rng, 1.0, side), "x": rho, "y": 0.0,
                                             "depth": _u(rng, 1.0, base.params["h"] * 0.6)}))
    feats.append(GoalFeature("polar_pattern", {"n": float(n)}))
    if len(feats) < 5 and rng.random() < 0.7:
        feats.append(_dressup(rng, rng.choice(["fillet_top", "chamfer_top"]), base))
    return Goal(feats, level=3, scale=_scale(base))


# FreeCAD's task-dialog defaults, per goal parameter. Round numbers like these
# are common in real parts, and when the goal asks for one the dialog field
# already holds it: the right move is to click OK without typing. Training
# goals snap parameters to these values with probability S1_DEFAULT_SNAP
# (training split only; evaluation suites are unchanged).
DIALOG_DEFAULTS = {
    "base_box": {"h": 10.0}, "base_cyl": {"h": 10.0}, "base_hex": {"h": 10.0},
    "boss_cyl": {"h": 10.0}, "boss_box": {"h": 10.0}, "pocket_rect": {"depth": 5.0}, "hole_std": {"r": 3.0},
    "fillet_top": {"r": 1.0}, "fillet_vertical": {"r": 1.0}, "chamfer_top": {"size": 1.0}, "shell": {"t": 1.0},
    "linear_pattern": {"n": 2.0},
}
DEFAULT_SNAP = float(os.environ.get("S1_DEFAULT_SNAP", "0"))


def snap_to_defaults(goal: Goal, rng: random.Random, p: float) -> Goal:
    for f in goal.features:
        for key, value in DIALOG_DEFAULTS.get(f.kind, {}).items():
            if key in f.params and rng.random() < p:
                f.params[key] = value
    return goal


def sample_split_goal(split: str, level: int, rng: random.Random, tries: int = 2000) -> Goal:
    if split == "comp2":
        return _mirrored_boss_box_goal(rng)
    if split == "comp3":
        return _patterned_pocket_goal(rng)
    if split == "defaults":  # held-out test: in-distribution goals with many parameters at dialog defaults
        return snap_to_defaults(sample_split_goal("iid", level, rng, tries), rng, 0.6)
    for _ in range(tries):
        goal = sample_goal(level, rng)
        if goal_in_split(goal, split):
            if split == "train" and DEFAULT_SNAP > 0:
                snap_to_defaults(goal, rng, DEFAULT_SNAP)
            return goal
    raise ValueError(f"could not sample a {split} goal at level {level}")


def sample_start(rng: random.Random, level: int) -> StartSpec:
    r = rng.random()
    if r < 0.25:
        return StartSpec(doc_open=False, workbench=rng.choice(["StartWorkbench", "PartWorkbench", "PartDesignWorkbench"]))
    if r < 0.45 and level >= 2:
        return StartSpec(doc_open=True, workbench="PartDesignWorkbench", body=True)
    return StartSpec(doc_open=True, workbench=rng.choice(["StartWorkbench", "PartWorkbench", "PartDesignWorkbench", "PartDesignWorkbench"]))

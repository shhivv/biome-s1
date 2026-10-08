"""The part of a `State` that can be read off the screen.

A model that only ever sees `screen_state(state)` can run on what perception
recovers from FreeCAD's window (model tree, toolbars, task panel, the
executor's own picks) without FreeCAD's API. Kept:

- mode: document open, workbench, sketch in edit, body exists, undo enabled,
  the open dialog and its fields (accessibility)
- the model tree: object types, order and nesting (OCR of the tree), and per
  object what the tree shows: in edit, visible (greyed out or not), consumed
  (nested under its feature), active body (bold), tip (the last solid shown)
- the selection: kind, object type, count of sub-elements, and normal/offset
  (the executor picked it and read its point back from the status bar)
- recent actions (the agent's own history)

Dropped (not shown in the window without opening extra panels):
- per-object geometry and parameters (sketch size and support, constraint
  counts and degrees of freedom, lengths, radii, occurrences, volumes, ...)
- sketch geometry / constraint kind counts
- whole-part shape statistics (volume, area, bounding box, face and edge
  counts, face directions, validity)
- object validity (shown only as an error overlay icon, not read yet)
"""
from __future__ import annotations

import dataclasses

from .schema import ShapeInfo, State

SCREEN_NODE_KEYS = ("tip", "in_edit", "visible", "consumed", "active_body")


def screen_state(state: State) -> State:
    tree = [dataclasses.replace(n, num={k: v for k, v in n.num.items() if k in SCREEN_NODE_KEYS}, geo={}, cons={})
            for n in state.tree]
    return dataclasses.replace(state, tree=tree, shape=ShapeInfo())


def ok_on_target(ui: dict | None) -> float:
    """1.0 if pressing OK now commits on-target values: every enabled number field with a target
    (the parameter stage's value, shown next to what the field holds) already holds it."""
    from .ui.spec import value_matches

    if not ui or not ui.get("dialog"):
        return 0.0
    for f in ui.get("fields", {}).values():
        if f.get("kind") == "number" and f.get("enabled", True) and f.get("target") is not None:
            if f.get("value") is None or not value_matches(f["target"], f["value"]):
                return 0.0
    return 1.0


class OkMemory:
    """Keeps `State.recent_flags` for one episode. Call `before(state)` with the state an action is
    chosen in and `after(new_state)` with the state it led to. An action that was refused without
    entering the history (e.g. a command while a dialog is open) adds no flag."""

    def __init__(self) -> None:
        self.flags: list[float] = []
        self._pending = 0.0
        self._recent: list[str] | None = None

    def before(self, state: State) -> None:
        self._pending = ok_on_target(state.ui)
        self._recent = list(state.recent)
        state.recent_flags = self.window(state)

    def after(self, new_state: State | None) -> None:
        if new_state is None:
            return
        if self._recent is not None and new_state.recent and new_state.recent != self._recent:
            self.flags.append(self._pending if new_state.recent[-1] == "click:OK" else 0.0)
        new_state.recent_flags = self.window(new_state)

    def window(self, state: State) -> list[float]:
        n = len(state.recent)
        out = self.flags[-n:] if n else []
        return [0.0] * (n - len(out)) + out

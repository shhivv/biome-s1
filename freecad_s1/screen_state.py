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

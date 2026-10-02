"""UI-level Taiga-S1: the model acts on interface elements (toolbar commands,
task-panel fields, OK/Cancel) read from FreeCAD's live Qt widget tree,
instead of on headless command ids.

`spec` and `teacher` are pure stdlib (usable from the torch process);
`session`, `server` and the shard entry run inside the FreeCAD GUI.
"""

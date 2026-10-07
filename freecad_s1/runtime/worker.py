"""JSON-lines RPC server around a HeadlessSession; runs in FreeCAD's Python.

The torch process (which cannot import FreeCAD) drives FreeCAD through this
worker. Requests arrive one per line on stdin; each response is one line on
stdout prefixed with ``@@S1 `` (FreeCAD may print its own noise).

Ops:
  {"op": "reset", "level": L, "seed": S, "split": "iid"}  sample a feasible goal + start
  {"op": "reset", "goal": {...}, "start": {...}}  use a given goal
  {"op": "step", "action": "PartDesign_Pad", "reward": bool}
  {"op": "score"}
  {"op": "quit"}
Every reset/step response carries state, valid actions and the expert's
acceptable set (for DAgger labels and on-policy agreement metrics).
"""

from __future__ import annotations

import json
import random
import sys
import traceback
from dataclasses import asdict

from ..goals import StartSpec
from ..schema import Goal
from ..goals import sample_start
from .episode import build_target, new_episode, sample_feasible_goal, score, step_budget
from .fcenv import PROTOCOL_PREFIX as PREFIX
from .session import HeadlessSession, iou


class Worker:
    def __init__(self, session=None, target_session=None, budget_fn=None) -> None:
        """`session` is driven by the policy; targets are built in
        `target_session` when given (the GUI server keeps target construction
        in a hidden headless session so it never shows up in the window).
        `budget_fn(goal, start)` overrides the step budget (UI sessions)."""
        self.session = session or HeadlessSession()
        self.target_session = target_session
        self.budget_fn = budget_fn or step_budget
        self.target = None
        self.last_iou = 0.0

    def observe(self) -> dict:
        state = self.session.state()
        return {"state": state.to_json(), "actions": self.session.valid_actions(state), "expert": self.session.expert(),
                "progress": self.session.progress()}

    def handle(self, req: dict) -> dict:
        op = req["op"]
        s = self.session
        if op == "reset":
            builder = self.target_session or s
            if "goal" in req:
                goal = Goal.from_json(req["goal"])
                self.target = build_target(builder, goal)
                start = StartSpec(**req.get("start", {}))
                s.reset(goal, start)
            elif self.target_session is not None:
                rng = random.Random(int(req["seed"]))
                goal, self.target = sample_feasible_goal(builder, int(req["level"]), rng,
                                                         split=req.get("split", "train"))
                start = sample_start(rng, int(req["level"]))
                s.reset(goal, start)
            else:
                goal, start, self.target = new_episode(s, int(req["level"]), random.Random(int(req["seed"])),
                                                       split=req.get("split", "train"))
            self.last_iou = 0.0
            return {"goal": goal.to_json(), "start": asdict(start), "budget": self.budget_fn(goal, start), **self.observe()}
        if op == "step":
            info = s.step(req["action"])
            out = {"info": info, **self.observe()}
            if req.get("reward"):
                value = iou(s.solid(), self.target)
                out["iou"], out["delta_iou"] = value, value - self.last_iou
                self.last_iou = value
            return out
        if op == "ext_begin":  # an action the client performs from outside the app (UiSession.begin_external)
            return {"request": s.begin_external(req["action"])}
        if op == "observe":
            return self.observe()
        if op == "ext_end":
            return {"info": s.end_external(), **self.observe()}
        if op == "score":
            return score(s, self.target)
        if op == "save":  # save the document (and, in the GUI, a screenshot)
            out = {}
            if s.doc is not None and req.get("fcstd"):
                s.doc.saveAs(req["fcstd"])
                out["fcstd"] = req["fcstd"]
            if req.get("png"):
                import FreeCADGui as Gui

                view = Gui.getDocument(s.doc.Name).ActiveView
                if req.get("camera"):  # fixed camera, e.g. for step-by-step frames
                    view.setCamera(req["camera"])
                else:
                    view.viewIsometric()
                    view.fitAll()
                view.saveImage(req["png"], int(req.get("width", 1400)), int(req.get("height", 1000)),
                               req.get("background", "Current"))  # e.g. "Transparent"
                out["png"] = req["png"]
                out["camera"] = view.getCamera()
            return out
        raise ValueError(f"unknown op {op}")

    def serve(self) -> None:
        out = sys.stdout
        sys.stdout = sys.stderr  # keep stray prints off the protocol channel
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            if req.get("op") == "quit":
                break
            try:
                resp = {"ok": True, **self.handle(req)}
            except Exception as exc:  # report, keep serving
                resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
            out.write(PREFIX + json.dumps(resp, separators=(",", ":")) + "\n")
            out.flush()
        self.session.shutdown()


if __name__ == "__main__":
    Worker().serve()

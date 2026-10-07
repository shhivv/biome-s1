"""Run Mesa-S1 with its view of FreeCAD's interface read from the macOS accessibility tree.

Starts a visible FreeCAD (background, no focus) under the watchdog serving a UI
session, and at every step rebuilds the model's element list and dialog-field
values from the OS accessibility tree (see freecad_s1/ui/ax.py for what is and
isn't read from it). Actions are applied by the session. Reports success and how often the accessibility view agrees with
the session's own view.

    pip install -e ".[ax]"     # pyobjc; grant Accessibility permission to your terminal
    python scripts/mesa_ax.py --episodes 15
    python scripts/mesa_ax.py --name flange        # a showcase part
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse
import json
import re
import shutil
import socket
import subprocess
import tempfile
import time

from freecad_s1.model.net import from_pretrained, load_checkpoint
from freecad_s1.rollout import Policy
from freecad_s1.runtime.fcenv import PROTOCOL_PREFIX
from freecad_s1.schema import Goal, State
from freecad_s1.ui.ax import AccessibilityActuator, AccessibilityReader, accessibility_view
from freecad_s1.ui.launch import find_pids, launch_gui_script

ROOT = Path(__file__).resolve().parents[1]


class Server:
    def __init__(self, port: int, wait: float = 180.0) -> None:
        deadline = time.time() + wait
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=600)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(1)
        self.file = self.sock.makefile("r")

    def call(self, req: dict) -> dict:
        self.sock.sendall((json.dumps(req) + "\n").encode())
        resp = json.loads(self.file.readline()[len(PROTOCOL_PREFIX):])
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error"))
        return resp


def launch(port: int, mem_gb: float):
    tmp = Path(tempfile.mkdtemp(prefix="s1-mesa-ax-"))
    script = tmp / "mesa_ax_server.FCMacro"  # unique path: the watchdog's handle on this FreeCAD
    shutil.copy(ROOT / "scripts" / "ui_server.FCMacro", script)
    proc = launch_gui_script(script, {"S1_PORT": str(port)}, log=tmp / "freecad.log", mem_limit_gb=mem_gb,
                             timeout=6 * 3600, hidden=False)  # accessibility needs a real (non-hidden) window
    srv = Server(port)
    pid = next(p for p in find_pids(str(script))
               if "bin/freecad" in subprocess.run(["ps", "-o", "command=", "-p", str(p)], capture_output=True,
                                                  text=True).stdout.lower())
    reader = AccessibilityReader(pid)
    return proc, srv, reader, AccessibilityActuator(reader, pid)


def step(srv: Server, actuator: AccessibilityActuator, action: str, stats: dict) -> dict:
    """Perform `action` from outside: the session prepares and books it, accessibility performs it."""
    req = srv.call({"op": "ext_begin", "action": action})["request"]
    if req.get("internal"):  # workbench, canvas, commands without a dialog, Done
        return srv.call({"op": "step", "action": action})
    if "error" in req:
        return {"info": req["info"], **srv.call({"op": "observe"})}
    try:
        actuator.perform(req)
        stats["acted_via_accessibility"] += 1
    except RuntimeError as exc:
        key = re.sub(r"\d+(\.\d+)?", "#", str(exc))[:90]
        stats["actuation_errors"][key] = stats["actuation_errors"].get(key, 0) + 1
    return srv.call({"op": "ext_end"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="release/mesa-s1")
    ap.add_argument("--episodes", type=int, default=15, help="sampled goals (ignored with --name)")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--split", default="iid")
    ap.add_argument("--seed", type=int, default=1_300_900)
    ap.add_argument("--goals", default="scripts/showcase_goals.json")
    ap.add_argument("--name", help="a showcase goal from --goals instead of sampled ones")
    ap.add_argument("--port", type=int, default=8897)
    ap.add_argument("--mem-gb", type=float, default=2.5)
    ap.add_argument("--out", default="runs/mesa_ax.json")
    ap.add_argument("--log", help="append each action here before performing it (locates a crash)")
    ap.add_argument("--act", action="store_true",
                    help="also perform dialog actions (toolbar commands, fields, dropdowns, check boxes, OK/Cancel) "
                         "through accessibility + posted key events, not by the session")
    args = ap.parse_args()

    model = load_checkpoint(args.model) if args.model.endswith(".pt") else from_pretrained(args.model)
    policy = Policy(model, "cpu")
    if args.name:
        goal = json.loads((ROOT / args.goals).read_text())[args.name]
        jobs = [{"op": "reset", "goal": goal, "start": {"doc_open": True, "workbench": "PartDesignWorkbench"}}]
    else:
        jobs = [{"op": "reset", "level": args.level, "split": args.split, "seed": args.seed + k}
                for k in range(args.episodes)]
    proc, srv, reader, actuator = launch(args.port, args.mem_gb)
    stats = {"steps": 0, "same_elements": 0, "same_decision": 0, "read_ms": [], "notes": {}, "episodes": [],
             "acted_via_accessibility": 0, "actuation_errors": {}}
    try:
        for k, job in enumerate(jobs):
            try:
                r = srv.call(job)
                goal = Goal.from_json(r["goal"])
                state, actions = State.from_json(r["state"]), r["actions"]
                done, steps = False, 0
                while steps < r["budget"]:
                    snap = reader.read()
                    elements, ui, notes = accessibility_view(snap, state, actions)
                    stats["read_ms"].append(1000 * snap["seconds"])
                    for n in notes:
                        key = re.sub(r"\d+(\.\d+)?", "#", n)[:90]
                        stats["notes"][key] = stats["notes"].get(key, 0) + 1
                    seen = State.from_json({**state.to_json(), "ui": ui})
                    action = policy.act([seen], [goal], [elements])[0]
                    stats["steps"] += 1
                    stats["same_elements"] += int(set(elements) == set(actions))
                    stats["same_decision"] += int(action == policy.act([state], [goal], [actions])[0])
                    if args.log:
                        with open(args.log, "a") as fh:
                            fh.write(f"episode {k} step {steps}: {action}\n")
                    s = step(srv, actuator, action, stats) if args.act else srv.call({"op": "step", "action": action})
                    if args.log and s["info"].get("error"):
                        with open(args.log, "a") as fh:
                            fh.write(f"    -> error: {s['info']['error']}\n")
                    steps += 1
                    if s["info"]["done"]:
                        done = True
                        break
                    state, actions = State.from_json(s["state"]), s["actions"]
                sc = srv.call({"op": "score"})
                ep = {"episode": k, "success": bool(done and sc["match"]), "iou": round(sc["iou"], 3), "steps": steps,
                      "features": [f.kind for f in goal.features]}
            except (RuntimeError, ValueError, OSError) as exc:  # FreeCAD died: note it and restart
                ep = {"episode": k, "success": False, "crashed": repr(exc)[:80]}
                proc.kill("crashed")
                time.sleep(2)
                proc, srv, reader, actuator = launch(args.port, args.mem_gb)
            stats["episodes"].append(ep)
            print(json.dumps(ep), flush=True)
        try:
            srv.call({"op": "exit"})
        except Exception:
            pass
    finally:
        proc.kill("done")
    ms = sorted(stats["read_ms"])
    report = {"episodes": len(stats["episodes"]), "success": sum(e["success"] for e in stats["episodes"]),
              "crashes": sum(1 for e in stats["episodes"] if "crashed" in e), "steps": stats["steps"],
              "same_element_list": stats["same_elements"], "same_decision_as_session_view": stats["same_decision"],
              "accessibility_read_ms_median": round(ms[len(ms) // 2]) if ms else None,
              "acted_via_accessibility": stats["acted_via_accessibility"],
              "actuation_errors": stats["actuation_errors"],
              "disagreements": dict(sorted(stats["notes"].items(), key=lambda kv: -kv[1])[:10]),
              "detail": stats["episodes"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "detail"}, indent=1))


if __name__ == "__main__":
    main()

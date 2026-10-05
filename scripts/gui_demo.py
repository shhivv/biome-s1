"""Drive a live FreeCAD GUI with the model.

1. Launch FreeCAD with the server macro:
     FREECAD_S1_REPO=$PWD /Applications/FreeCAD.app/Contents/MacOS/FreeCAD scripts/freecad_gui_server.FCMacro
2. python scripts/gui_demo.py --model release/hf --level 3 --split iid --seed 7

Mesa-S1 (UI level): launch FreeCAD with FREECAD_S1_UI=1 as well, then
   python scripts/gui_demo.py --model release/mesa-s1 --level 3 --split iid --seed 7
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse
import json
import socket
import time
from pathlib import Path

from freecad_s1.model.net import from_pretrained, load_checkpoint
from freecad_s1.rollout import Policy
from freecad_s1.runtime.fcenv import PROTOCOL_PREFIX
from freecad_s1.schema import Goal, State


class SocketEnv:
    def __init__(self, port: int, wait: float = 120.0) -> None:
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
        line = self.file.readline()
        resp = json.loads(line[len(PROTOCOL_PREFIX):])
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error") + "\n" + resp.get("trace", ""))
        return resp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="release/hf")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--split", default="iid")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between steps (for watching)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default="runs/gui_demo")
    ap.add_argument("--goals", help="JSON file of named goals ({name: goal}); use with --name")
    ap.add_argument("--name", help="which goal from --goals to build")
    ap.add_argument("--frames-dir", help="save a frame after every step (needs --camera-from)")
    ap.add_argument("--camera-from", help="JSON file with a saved camera (from a previous run's .camera.json)")
    ap.add_argument("--background", default="Current", help='screenshot background, e.g. "Transparent"')
    args = ap.parse_args()

    model = load_checkpoint(args.model) if args.model.endswith(".pt") else from_pretrained(args.model)
    policy = Policy(model, "cpu")
    env = SocketEnv(args.port)
    if args.goals:
        spec = json.loads(Path(args.goals).read_text())[args.name]
        r = env.call({"op": "reset", "goal": spec, "start": {"doc_open": True, "workbench": "PartDesignWorkbench"}})
    else:
        r = env.call({"op": "reset", "level": args.level, "split": args.split, "seed": args.seed})
    goal = Goal.from_json(r["goal"])
    print("goal:", " -> ".join(f"{f.kind}{json.dumps(f.params)}" for f in goal.features))
    print("start:", r["start"], "budget:", r["budget"])
    state, actions, expert = State.from_json(r["state"]), r["actions"], r["expert"]
    agree = steps = 0
    done = False
    t0 = time.time()
    while steps < r["budget"]:
        t = time.perf_counter()
        action = policy.act([state], [goal], [actions])[0]
        ms = (time.perf_counter() - t) * 1000
        ok = action in expert
        agree += ok
        steps += 1
        print(f"{steps:3d}  {action:34s} {'✓' if ok else '✗ expert: ' + expert[0]}  ({ms:.1f} ms, {len(actions)} valid)")
        s = env.call({"op": "step", "action": action})
        if args.frames_dir:
            cam = json.loads(Path(args.camera_from).read_text())["camera"]
            fdir = Path(args.frames_dir).resolve()
            fdir.mkdir(parents=True, exist_ok=True)
            env.call({"op": "save", "png": str(fdir / f"step_{steps:03d}.png"), "camera": cam,
                      "background": "Transparent", "width": 1200, "height": 1200})
            with open(fdir / "steps.jsonl", "a") as fh:
                fh.write(json.dumps({"step": steps, "action": action}) + "\n")
        if s["info"]["error"]:
            print("      error:", s["info"]["error"])
        if s["info"]["done"] or not s["expert"]:
            done = s["info"]["done"]
            break
        state, actions, expert = State.from_json(s["state"]), s["actions"], s["expert"]
        time.sleep(args.delay)
    sc = env.call({"op": "score"})
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tag = args.name if args.goals else f"L{args.level}_{args.split}_{args.seed}"
    saved = env.call({"op": "save", "fcstd": str(out / f"{tag}.FCStd"), "png": str(out / f"{tag}.png"),
                      "background": args.background, "width": 1600, "height": 1600})
    success = done and sc["match"]
    print(f"\nresult: {'SUCCESS' if success else 'FAIL'}  IoU {sc['iou']:.4f}  done {done}  steps {steps}  "
          f"agreement {agree}/{steps}  wall {time.time() - t0:.1f}s")
    print("saved:", saved.get("fcstd"), saved.get("png"))
    if saved.get("camera"):
        (out / f"{tag}.camera.json").write_text(json.dumps({"camera": saved["camera"]}))


if __name__ == "__main__":
    main()

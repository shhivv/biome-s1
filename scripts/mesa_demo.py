"""Record Mesa-S1 operating FreeCAD's interface, with every action highlighted.

Starts a visible FreeCAD (in the background, without taking focus) under the
watchdog, serving a DemoUiSession (freecad_s1/ui/demo.py): before each action
the toolbar button, dialog field, dropdown, check box or OK/Cancel button the
model chose is highlighted, numbers are typed character by character, and a
caption names the action and the model's decision time. The main window is
recorded to frames and encoded into one MP4 per part.

    python scripts/mesa_demo.py --name flange
    python scripts/mesa_demo.py --all                    # every showcase part
    python scripts/mesa_demo.py --level 3 --seed 7       # a sampled goal instead

Output: runs/mesa_demo/<part>.mp4 (+ frames/, .FCStd, .png). Needs ffmpeg.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse
import json
import shutil
import socket
import subprocess
import tempfile
import time

from freecad_s1.model.net import from_pretrained, load_checkpoint
from freecad_s1.rollout import Policy
from freecad_s1.runtime.fcenv import PROTOCOL_PREFIX
from freecad_s1.schema import Goal, State
from freecad_s1.ui.launch import launch_gui_script

PRETTY = {"base_box": "Plate", "base_cyl": "Disc", "base_hex": "Hex prism", "base_ring": "Ring", "boss_cyl": "Boss",
          "boss_box": "Block", "hole": "Hole", "hole_std": "Hole", "pocket_rect": "Pocket",
          "polar_pattern": "Polar pattern", "linear_pattern": "Linear pattern", "mirror": "Mirror",
          "fillet_top": "Fillet", "fillet_vertical": "Fillet", "chamfer_top": "Chamfer", "shell": "Shell"}


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
            raise RuntimeError(resp.get("error", "") + "\n" + resp.get("trace", ""))
        return resp


def run_part(srv: Server, policy: Policy, name: str, reset: dict, out: Path, fps: int, max_steps: int) -> dict:
    r = srv.call(reset)
    goal = Goal.from_json(r["goal"])
    frames = out / "frames" / name
    shutil.rmtree(frames, ignore_errors=True)
    srv.call({"op": "record_start", "dir": str(frames), "fps": fps})
    plan = "  →  ".join(PRETTY.get(f.kind, f.kind) for f in goal.features)
    srv.call({"op": "demo_wait", "caption": f"Goal: {plan}", "seconds": 2.5})
    state, actions = State.from_json(r["state"]), r["actions"]
    steps, done, decide_ms = 0, False, []
    while steps < min(r["budget"], max_steps):
        t = time.perf_counter()
        action = policy.act([state], [goal], [actions])[0]
        ms = (time.perf_counter() - t) * 1000
        decide_ms.append(ms)
        srv.call({"op": "demo_note", "text": f"·  {len(actions)} options  ·  decided in {ms:.1f} ms"})
        s = srv.call({"op": "step", "action": action})
        steps += 1
        print(f"{steps:3d}  {action:40s} {ms:5.1f} ms" + (f"   error: {s['info']['error']}" if s["info"]["error"] else ""))
        if s["info"]["done"]:
            done = True
            break
        state, actions = State.from_json(s["state"]), s["actions"]
    sc = srv.call({"op": "score"})
    ok = done and sc["match"]
    srv.call({"op": "demo_note", "text": ""})
    srv.call({"op": "demo_wait", "caption": f"{'Built' if ok else 'Finished'}: {steps} actions, IoU {sc['iou']:.3f}",
              "seconds": 2.5})
    stop = srv.call({"op": "record_stop"})
    n = stop["frames"]
    (frames / "events.json").write_text(json.dumps(stop["events"]))  # per-action frame ranges, for the teaser cut
    srv.call({"op": "save", "fcstd": str(out / f"{name}.FCStd"), "png": str(out / f"{name}.png"),
              "width": 1600, "height": 1600})
    video = out / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "frame_%06d.jpg"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(video)], check=True)
    return {"part": name, "success": ok, "iou": sc["iou"], "steps": steps, "frames": n, "video": str(video),
            "median_decision_ms": sorted(decide_ms)[len(decide_ms) // 2] if decide_ms else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="release/mesa-s1")
    ap.add_argument("--goals", default="scripts/showcase_goals.json")
    ap.add_argument("--name", default="flange")
    ap.add_argument("--all", action="store_true", help="record every goal in --goals")
    ap.add_argument("--level", type=int, help="record a sampled goal at this level instead")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/mesa_demo")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--port", type=int, default=8870)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--mem-gb", type=float, default=3.0, help="watchdog: kill FreeCAD above this footprint")
    args = ap.parse_args()

    model = load_checkpoint(args.model) if args.model.endswith(".pt") else from_pretrained(args.model)
    policy = Policy(model, "cpu")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.level:
        jobs = [(f"L{args.level}_seed{args.seed}", {"op": "reset", "level": args.level, "split": "iid", "seed": args.seed})]
    else:
        goals = json.loads(Path(args.goals).read_text())
        names = list(goals) if args.all else [args.name]
        start = {"doc_open": True, "workbench": "PartDesignWorkbench"}
        jobs = [(n, {"op": "reset", "goal": goals[n], "start": start}) for n in names]

    tmp = Path(tempfile.mkdtemp(prefix="s1-mesa-demo-"))
    script = tmp / "mesa_demo_server.FCMacro"  # unique path: the watchdog's handle on this FreeCAD
    shutil.copy(Path(__file__).with_name("ui_server.FCMacro"), script)
    proc = launch_gui_script(script, {"S1_PORT": str(args.port), "S1_DEMO": "1"}, log=tmp / "freecad.log",
                             mem_limit_gb=args.mem_gb, timeout=3600, hidden=False)
    results = []
    try:
        srv = Server(args.port)
        srv.call({"op": "demo_window", "width": args.width, "height": args.height})
        for name, reset in jobs:
            print(f"== {name}")
            results.append(run_part(srv, policy, name, reset, out, args.fps, args.max_steps))
            print(json.dumps(results[-1]))
        try:
            srv.call({"op": "exit"})
        except Exception:
            pass
    finally:
        proc.kill("demo finished")
        print(f"FreeCAD peak memory {proc.peak_gb:.2f} GB")
    (out / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Pick the recorded frames for the Mesa-S1 teaser (viz/src/MesaTeaser.tsx).

The recording (scripts/mesa_demo.py) holds every action for ~1 s. The teaser
has ~7.5 s for the whole build, so frames are chosen per action: interface
actions (toolbar buttons, dialog fields, dropdowns, check boxes, OK/Cancel)
get most of the time, interactions in the 3D view and the sketch editor whizz
past. Each action's highlight and "press" are kept.

    python scripts/mesa_teaser_frames.py --part flange

Writes viz/public/mesa/<part>/src_*.jpg (gitignored) and viz/src/mesaFrames.json:
the frame per teaser frame, a smoothed camera (push in on the widget being
clicked), and the real-time numbers for the "slowed down" badge.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse
import json
import shutil
import subprocess

from freecad_s1.ui.spec import role

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = {"canvas": 0.3, "done": 0.8}  # everything else (interface widgets) = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="flange")
    ap.add_argument("--recording", default="runs/mesa_demo")
    ap.add_argument("--frames", type=int, default=225, help="teaser frames for the build (30 fps)")
    ap.add_argument("--width", type=int, default=2400, help="frame width (zoomed shots need resolution)")
    ap.add_argument("--zoom", type=float, default=1.8, help="push-in on interface widgets")
    ap.add_argument("--ease", type=float, default=0.16, help="camera smoothing per frame")
    args = ap.parse_args()

    src = ROOT / args.recording / "frames" / args.part
    events = json.loads((src / "events.json").read_text())
    weights = [WEIGHT.get(role(e["element"]), 1.0) for e in events]
    total = sum(weights)
    alloc = [max(2, round(args.frames * w / total)) for w in weights]
    alloc[-1] += args.frames - sum(alloc)  # exact length; the slack goes to the final action

    picks, labels, captions, notes, targets, cursor_targets, pressed = [], [], [], [], [], [], []
    for e, k in zip(events, alloc):
        lo, hi = e["start"], max(e["start"], e["end"] - 1)
        r = e.get("rect")
        widget = r is not None and not e.get("dashed")
        # push in on interface widgets so the click reads on a phone; ease out for 3D-view steps
        tz = args.zoom if widget else 1.0
        tf = (r[0] + r[2] / 2, r[1] + r[3] / 2) if widget else (0.5, 0.5)
        for j in range(k):  # evenly through highlight -> press -> result
            picks.append(round(lo + (hi - lo) * j / max(k - 1, 1)))
            labels.append(e["element"])
            captions.append(e.get("caption", e["element"]))
            notes.append(e.get("note", "").strip(" \u00b7"))
            targets.append((tz, tf))
            # cursor: where the recording's cursor went for this action (widget centre, or into the 3D view)
            cursor_targets.append(None if r is None else ((r[0] + r[2] * (0.62 if e.get("dashed") else 0.5)),
                                                          (r[1] + r[3] * (0.6 if e.get("dashed") else 0.5))))
            pressed.append(picks[-1] >= e.get("press", hi + 1))
    zoom, focus, z, fx, fy = [], [], 1.0, 0.5, 0.5
    for tz, (tx, ty) in targets:  # smoothed camera
        z += (tz - z) * args.ease
        fx += (tx - fx) * args.ease
        fy += (ty - fy) * args.ease
        zoom.append(round(z, 4))
        focus.append([round(fx, 4), round(fy, 4)])
    # cursor path: glide (ease in-out) to each action's target over its first frames, then rest there
    cursor, cx, cy, i = [], 0.55, 0.6, 0
    for e, k in zip(events, alloc):
        tgt = cursor_targets[i]
        sx, sy = cx, cy
        glide = max(1, min(k - 1, round(k * 0.45)))
        for j in range(k):
            if tgt is not None:
                u = min(1.0, (j + 1) / glide)
                ease = 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2
                cx, cy = sx + (tgt[0] - sx) * ease, sy + (tgt[1] - sy) * ease
            cursor.append([round(cx, 4), round(cy, 4)])
            i += 1
    clicks, i = [], 0  # ripple start per interface click: the first frame at/after the action's press
    for e, k in zip(events, alloc):
        if e.get("rect") is not None and not e.get("dashed"):
            first = next((i + j for j in range(k) if pressed[i + j]), None)
            if first is not None:
                clicks.append(first)
        i += k

    out = ROOT / "viz" / "public" / "mesa" / args.part
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    unique = sorted(set(picks))
    for i in unique:
        subprocess.run(["sips", "-Z", str(args.width), str(src / f"frame_{i:06d}.jpg"), "--out",
                        str(out / f"src_{i:06d}.jpg")], check=True, capture_output=True)
    last = events[-1]["end"]
    final = out / "final.jpg"  # the finished part, for the end card
    subprocess.run(["sips", "-Z", str(args.width), str(src / f"frame_{last:06d}.jpg"), "--out", str(final)],
                   check=True, capture_output=True)
    model_s = sum(e.get("decide_ms", 0) for e in events) / 1000
    run_s = model_s + sum(e.get("exec_ms", 0) for e in events) / 1000
    spec = {"part": args.part, "frames": [f"mesa/{args.part}/src_{i:06d}.jpg" for i in picks], "elements": labels, "captions": captions, "notes": notes,
            "cursor": cursor, "clicks": clicks,
            "zoom": zoom, "focus": focus, "final": f"mesa/{args.part}/final.jpg", "actions": len(events),
            # real time without the demo's pauses: the model's decisions + FreeCAD executing them
            "model_s": round(model_s, 2), "run_s": round(run_s, 1),
            "slowdown": round(args.frames / 30 / run_s, 1) if run_s else None}
    (ROOT / "viz" / "src" / "mesaFrames.json").write_text(json.dumps(spec))
    print(f"{len(picks)} teaser frames from {len(unique)} recorded frames, {len(events)} actions -> {out}")


if __name__ == "__main__":
    main()

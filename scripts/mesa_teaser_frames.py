"""Pick the recorded frames for the Mesa-S1 teaser (viz/src/MesaTeaser.tsx).

The recording (scripts/mesa_demo.py) holds every action for ~1 s. The teaser
has ~7.5 s for the whole build, so frames are chosen per action: interface
actions (toolbar buttons, dialog fields, dropdowns, check boxes, OK/Cancel)
get most of the time, interactions in the 3D view and the sketch editor whizz
past. Each action's highlight and "press" are kept.

    python scripts/mesa_teaser_frames.py --part flange

Writes viz/public/mesa/<part>/f_000.jpg, ... (1920 px wide; gitignored) and
viz/src/mesaFrames.json.
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
    ap.add_argument("--width", type=int, default=1920)
    args = ap.parse_args()

    src = ROOT / args.recording / "frames" / args.part
    events = json.loads((src / "events.json").read_text())
    weights = [WEIGHT.get(role(e["element"]), 1.0) for e in events]
    total = sum(weights)
    alloc = [max(2, round(args.frames * w / total)) for w in weights]
    alloc[-1] += args.frames - sum(alloc)  # exact length; the slack goes to the final action

    picks, labels = [], []
    for e, k in zip(events, alloc):
        lo, hi = e["start"], max(e["start"], e["end"] - 1)
        for j in range(k):  # evenly through highlight -> press -> result
            picks.append(round(lo + (hi - lo) * j / max(k - 1, 1)))
            labels.append(e["element"])

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
    spec = {"part": args.part, "frames": [f"mesa/{args.part}/src_{i:06d}.jpg" for i in picks], "elements": labels,
            "final": f"mesa/{args.part}/final.jpg", "actions": len(events)}
    (ROOT / "viz" / "src" / "mesaFrames.json").write_text(json.dumps(spec))
    print(f"{len(picks)} teaser frames from {len(unique)} recorded frames, {len(events)} actions -> {out}")


if __name__ == "__main__":
    main()

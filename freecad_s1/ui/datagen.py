"""UI-level synthetic data: the same episodes as freecad_s1.datagen, but the
teacher labels interface elements and every step runs through the real
FreeCAD GUI (hidden), via `UiSession`.

    python -m freecad_s1.ui.datagen --out data/ui_train --episodes 400 800 1200 --workers 4

Each worker is a separate hidden FreeCAD GUI running scripts/ui_shard.FCMacro
under the watchdog (freecad_s1/ui/launch.py): it is killed above --mem-gb or
after --timeout seconds, and when this launcher exits. Output is the
datagen format (gzip JSONL; `actions` / `acceptable` are UI element ids and
`state.ui` holds the open dialog), so freecad_s1.data loads it as is; train
with `train_sft --ui`.

A UI step costs ~60 ms against ~3 ms headless, and every worker is a full
GUI (~1 GB), so keep --workers small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runtime.fcenv import REPO_ROOT
from .launch import launch_gui_script

SHARD_SCRIPT = REPO_ROOT / "scripts" / "ui_shard.FCMacro"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs=3, default=[100, 150, 200], metavar=("L1", "L2", "L3"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=3.0, help="kill a worker above this memory footprint")
    ap.add_argument("--timeout", type=float, default=4 * 3600, help="kill a worker after this many seconds")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    per_worker = [[c // args.workers + (1 if w < c % args.workers else 0) for c in args.episodes]
                  for w in range(args.workers)]
    # One script copy per worker: the watchdog finds a worker by its script path.
    procs = []
    for w in range(args.workers):
        script = out / f".shard{w:03d}.FCMacro"
        script.write_text(SHARD_SCRIPT.read_text())
        env = {"S1_OUT_DIR": str(out), "S1_SHARD": str(w), "S1_SEED": str(args.seed),
               "S1_EPISODES": " ".join(map(str, per_worker[w]))}
        procs.append((w, launch_gui_script(script, env, log=out / f"shard{w:03d}.log",
                                           mem_limit_gb=args.mem_gb, timeout=args.timeout)))
    failed = False
    for w, proc in procs:
        proc.wait()
        summary_path = out / f"shard{w:03d}.summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"shard": w, "error": "no summary"}
        summary["peak_gb"] = round(proc.peak_gb, 2)
        if proc.killed:
            summary["killed"] = proc.killed
        failed |= "error" in summary or "killed" in summary
        print(json.dumps(summary))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

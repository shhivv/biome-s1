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

FreeCAD occasionally segfaults inside a task dialog (seen once in ~150
pilot episodes, accepting a Pocket after switching its mode back and forth);
the launcher restarts a crashed worker for the episodes it still owes, and
partial shards stay loadable.

A UI step costs ~60 ms against ~3 ms headless, and every worker is a full
GUI (~1 GB), so keep --workers small.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..runtime.fcenv import REPO_ROOT
from .launch import GuiProcess, launch_gui_script

SHARD_SCRIPT = REPO_ROOT / "scripts" / "ui_shard.FCMacro"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs=3, default=[100, 150, 200], metavar=("L1", "L2", "L3"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=3.0, help="kill a worker above this memory footprint")
    ap.add_argument("--timeout", type=float, default=4 * 3600, help="kill a worker after this many seconds")
    ap.add_argument("--max-restarts", type=int, default=1000,
                    help="restarts per worker (FreeCAD crashes ~1 per 100 episodes; memory-cap kills recycle workers)")
    ap.add_argument("--quiet-dialogs", action="store_true",
                    help="turn off task dialogs' live preview / recompute-on-change (see UiSession)")
    ap.add_argument("--shard-offset", type=int, default=0,
                    help="added to shard ids, so a continuation run into the same --out never overwrites a shard")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    remaining = [[c // args.workers + (1 if w < c % args.workers else 0) for c in args.episodes]
                 for w in range(args.workers)]
    attempts = [0] * args.workers
    running: dict[int, tuple[int, GuiProcess]] = {}
    failed = False

    def start(w: int) -> None:
        # FreeCAD occasionally segfaults inside its own dialogs; a worker that
        # dies is restarted with a fresh shard id (and so fresh episodes) for
        # the episodes it still owes. Partial shards stay usable.
        shard = args.shard_offset + w * 10000 + attempts[w]
        attempts[w] += 1
        script = out / f".shard{shard:05d}.FCMacro"  # unique path: the watchdog's handle on the worker
        script.write_text(SHARD_SCRIPT.read_text())
        env = {"S1_OUT_DIR": str(out), "S1_SHARD": str(shard), "S1_SEED": str(args.seed),
               "S1_EPISODES": " ".join(map(str, remaining[w])),
               **({"S1_QUIET_DIALOGS": "1"} if args.quiet_dialogs else {})}
        running[w] = (shard, launch_gui_script(script, env, log=out / f"shard{shard:03d}.log",
                                               mem_limit_gb=args.mem_gb, timeout=args.timeout))

    for w in range(args.workers):
        start(w)
    while running:
        time.sleep(2)
        for w, (shard, proc) in list(running.items()):
            if proc.proc.poll() is None:
                continue
            proc.wait()
            del running[w]
            summary_path = out / f"shard{shard:03d}.summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"shard": shard}
            summary["peak_gb"] = round(proc.peak_gb, 2)
            if proc.killed:
                summary["killed"] = proc.killed
            done = "episodes" in summary and "error" not in summary and not proc.killed
            if not done:
                started = episodes_started(out / f"shard{shard:03d}.jsonl.gz")
                remaining[w] = [max(0, r - started.get(level, 0)) for level, r in enumerate(remaining[w], start=1)]
                summary.update(crashed=True, episodes_started=started, still_owed=remaining[w])
            print(json.dumps(summary), flush=True)
            if not done and sum(remaining[w]):
                if attempts[w] >= args.max_restarts + 1:
                    failed = True
                    print(json.dumps({"worker": w, "gave_up": True, "owed": remaining[w]}), flush=True)
                else:
                    start(w)
    if failed:
        raise SystemExit(1)


def episodes_started(path: Path) -> dict[int, int]:
    """Episodes per level that a (possibly truncated) shard began."""
    from ..data import read_records

    counts: dict[int, int] = {}
    if not path.exists():
        return counts
    for rec in read_records(str(path)):
        if "goal" in rec:
            level = int(rec["goal"].get("level", 1))
            counts[level] = counts.get(level, 0) + 1
    return counts


if __name__ == "__main__":
    main()

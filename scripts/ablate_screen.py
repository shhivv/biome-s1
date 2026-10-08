"""How much does a model rely on what the screen can't show?

Scores a checkpoint on recorded steps twice: with the full state, and with
`screen_state()` (freecad_s1/screen_state.py) applied. Reports how often the
top choice is acceptable, overall and per label category. Uses the held-out
episodes of the training split (every 20th episode by id).

    python scripts/ablate_screen.py --ckpt release/mesa-s1 --data data/ui_train --shards 80
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse
import json
import zlib
from collections import defaultdict

import torch

from freecad_s1.data import label_category, read_records
from freecad_s1.model.featurize import collate, make_example
from freecad_s1.model.net import from_pretrained, load_checkpoint
from freecad_s1.schema import Goal, State
from freecad_s1.ui import spec as UIS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="release/mesa-s1")
    ap.add_argument("--data", default="data/ui_train")
    ap.add_argument("--shards", type=int, default=80, help="shards to read (spread over the directory)")
    ap.add_argument("--out")
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt) if args.ckpt.endswith(".pt") else from_pretrained(args.ckpt)
    model.eval()
    opts = model.cfg.feature_opts()
    opts.pop("screen", None)
    paths = sorted(Path(args.data).glob("*.jsonl.gz"))
    paths = paths[:: max(1, len(paths) // args.shards)][: args.shards]

    hits = {"full": defaultdict(int), "screen": defaultdict(int)}
    total = defaultdict(int)
    batch = {"full": [], "screen": [], "cat": []}

    def flush():
        if not batch["cat"]:
            return
        for mode in ("full", "screen"):
            exs = batch[mode]
            with torch.no_grad():
                logits, _ = model({k: v for k, v in collate(exs).items()})
            for ex, lg, cat in zip(exs, logits, batch["cat"]):
                n = len(ex.target)
                ok = bool(ex.target[int(lg[:n].argmax())])
                hits[mode][cat] += ok
                hits[mode]["all"] += ok
        for cat in batch["cat"]:
            total[cat] += 1
            total["all"] += 1
        for k in batch:
            batch[k].clear()

    for path in paths:
        goals = {}
        for rec in read_records(str(path)):
            if "goal" in rec:
                goals[rec["ep"]] = Goal.from_json(rec["goal"])
                continue
            if zlib.crc32(rec["ep"].encode()) % 20 != 0:  # held-out episodes only
                continue
            if not set(rec["acceptable"]) & set(rec["actions"]):
                continue
            state = State.from_json(rec["state"])
            goal = goals[rec["ep"]]
            if opts.get("ui_param"):
                UIS.annotate_targets(state.ui, goal, rec.get("progress"))
            for mode in ("full", "screen"):
                batch[mode].append(make_example(state, goal, rec["actions"], rec["acceptable"], rec.get("progress"),
                                                **opts, screen=(mode == "screen")))
            batch["cat"].append(label_category(rec["acceptable"]))
            if len(batch["cat"]) >= 512:
                flush()
    flush()
    report = {cat: {"n": n, "full": round(hits["full"][cat] / n, 4), "screen": round(hits["screen"][cat] / n, 4)}
              for cat, n in sorted(total.items(), key=lambda kv: -kv[1])}
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

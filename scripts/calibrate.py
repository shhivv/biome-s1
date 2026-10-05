"""Fit a softmax temperature so reported action probabilities are calibrated.

Calibration data = on-policy states from fresh seeds across every suite
(iid, held-out compositions, held-out lengths), with 20% injected random
actions so recovery states are represented, each labeled by the expert.
Split by episode: one half fits T (min multi-label NLL), the other half
reports ECE / NLL / confidence before and after. Writes T into config.json.

    python scripts/calibrate.py --model release/hf
    S1_MEM_LIMIT_GB=2.5 python scripts/calibrate.py --model release/mesa-s1 --workers 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from freecad_s1.data import Dataset
from freecad_s1.evaluate import SPLIT_SEED_OFFSET, SUITES
from freecad_s1.model.featurize import collate
from freecad_s1.model.net import from_pretrained
from freecad_s1.rollout import Policy, run_episodes
from freecad_s1.runtime.client import VecEnv

CALIB_SEED_BASE = 7_000_000  # disjoint from test (1e6+), probes (3e6/4e6), RL (5e5+)


@torch.no_grad()
def logits_of(model, ds: Dataset):
    out = []
    for i in range(0, len(ds), 512):
        b = collate(ds.examples[i:i + 512])
        lg, _ = model(b)
        for row, ex in zip(lg, ds.examples[i:i + 512]):
            out.append((row[: len(ex.target)].double(), torch.from_numpy(ex.target)))
    return out


def metrics(pairs, T: float, bins: int = 15) -> dict:
    conf, correct, nll = [], [], []
    for lg, tgt in pairs:
        p = torch.softmax(lg / T, -1)
        top = int(p.argmax())
        conf.append(float(p[top]))
        correct.append(bool(tgt[top]))
        nll.append(-float(torch.log(p[tgt].sum().clamp_min(1e-300))))
    conf, correct = np.array(conf), np.array(correct)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    wrong = conf[~correct]
    return {"n": len(conf), "acc": round(float(correct.mean()), 4), "nll": round(float(np.mean(nll)), 4),
            "ece": round(float(ece), 4), "mean_conf": round(float(conf.mean()), 4),
            "mean_conf_when_wrong": round(float(wrong.mean()), 4) if len(wrong) else None,
            "n_wrong": int((~correct).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="release/hf")
    ap.add_argument("--episodes", type=int, default=32, help="per suite level")
    ap.add_argument("--perturb", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    model = from_pretrained(args.model)
    policy = Policy(model, torch.device("cpu"))
    rng = random.Random(CALIB_SEED_BASE)
    ds = Dataset()
    if model.cfg.ui:  # UI-level model (Mesa-S1): hidden FreeCAD GUIs with crash recovery
        from freecad_s1.ui.env import UiVecEnv

        vec = UiVecEnv(args.workers)
    else:
        vec = VecEnv(args.workers)
    try:
        for suite in ("iid", "comp", "comp2", "len", "len2", "len3"):
            for level, split in SUITES[suite]:
                for s in range(0, args.episodes, args.workers):
                    specs = [{"level": level, "split": split, "seed": CALIB_SEED_BASE + level * 100_000 + s + i
                              + SPLIT_SEED_OFFSET.get(split, 0)} for i in range(args.workers)]
                    run_episodes(policy, vec, specs, collect=ds, rng=rng, perturb=args.perturb)
    finally:
        vec.close()
    eps = sorted(set(ds.ep))
    fit_eps = set(eps[::2])
    fit = ds.subset([i for i, e in enumerate(ds.ep) if e in fit_eps])
    held = ds.subset([i for i, e in enumerate(ds.ep) if e not in fit_eps])
    fit_pairs, held_pairs = logits_of(model, fit), logits_of(model, held)
    grid = np.exp(np.linspace(np.log(0.5), np.log(64), 120))
    best_T = min(grid, key=lambda T: metrics(fit_pairs, T)["nll"])
    report = {"states": len(ds), "episodes": len(eps), "T": round(float(best_T), 3),
              "held_out_T1": metrics(held_pairs, 1.0), "held_out_T": metrics(held_pairs, best_T)}
    print(json.dumps(report, indent=2))
    cfg_path = Path(args.model) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["config"]["temperature"] = round(float(best_T), 3)
    cfg["metadata"]["calibration"] = report
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print("wrote temperature to", cfg_path)


if __name__ == "__main__":
    main()

"""Evaluation harness.

1. Per-step (offline): top-1 accuracy of the model's argmax against the
   expert's acceptable set on held-out labeled states, broken down by
   curriculum level, label category and whether the episode was noisy.
2. Multi-step (online): the model drives live FreeCAD from a fresh goal
   until it emits Done or exhausts the step budget (2x expert length + 6).
   Success = Done issued and the final solid matches the target with
   volumetric IoU >= 0.99. Also reports clean success (no leftover junk),
   step efficiency vs the expert, on-policy agreement with the expert and a
   failure breakdown. Level 4 goals (longer feature chains) never appear in
   training data and test compositional generalization; --perturb injects
   random off-plan actions to test recovery.

    python -m freecad_s1.evaluate --ckpt runs/sft/best.pt --data data/test --episodes 200
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch

from .data import Dataset, load_dataset
from .model.featurize import collate
from .model.net import S1Model, load_checkpoint, select_device

TEST_SEED_BASE = 1_000_000  # online eval seeds; training/DAgger never use this range


def multilabel_nll(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """-log sum_{a in acceptable} p(a): any acceptable action counts as correct."""
    pos = logits.masked_fill(~target, torch.finfo(logits.dtype).min)
    return torch.logsumexp(logits, -1) - torch.logsumexp(pos, -1)


@torch.no_grad()
def offline_metrics(model: S1Model, ds: Dataset, device: torch.device, batch_size: int = 512) -> dict:
    model.eval()
    correct = np.zeros(len(ds), bool)
    nll = np.zeros(len(ds), np.float32)
    for start in range(0, len(ds), batch_size):
        batch = collate(ds.examples[start:start + batch_size])
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(batch)
        pred = logits.argmax(-1)
        hit = batch["target"].gather(1, pred[:, None]).squeeze(1)
        correct[start:start + len(pred)] = hit.cpu().numpy()
        nll[start:start + len(pred)] = multilabel_nll(logits, batch["target"]).cpu().numpy()

    def group(keys) -> dict:
        acc = defaultdict(list)
        for k, c in zip(keys, correct):
            acc[k].append(c)
        return {str(k): {"acc": round(float(np.mean(v)), 4), "n": len(v)} for k, v in sorted(acc.items())}

    return {
        "acc": round(float(correct.mean()), 4),
        "nll": round(float(nll.mean()), 4),
        "n": len(ds),
        "by_level": group(ds.level),
        "by_category": group(ds.category),
        "by_noise": group(["clean" if n == 0 else ("dagger" if n < 0 else "noisy") for n in ds.noise]),
    }


# Seed offsets keep suites that share a level (iid-L3, comp, comp2, comp3) on disjoint seeds.
SPLIT_SEED_OFFSET = {"iid": 0, "comp": 10_000, "comp2": 20_000, "comp3": 30_000, "len": 0, "defaults": 40_000}

SUITES = {
    "iid": [(1, "iid"), (2, "iid"), (3, "iid")],  # training distribution, fresh goals
    "comp": [(3, "comp")],  # held-out feature combinations (goals.heldout_composition)
    "comp2": [(3, "comp2")],  # held-out pair boss_box -> mirror (seen by the model before the M design)
    "comp3": [(3, "comp3")],  # held-out pair pocket_rect -> polar_pattern; evaluated once, on the final model
    "defaults": [(3, "defaults")],  # parameters at FreeCAD's dialog defaults (nothing to type before OK)
    "len": [(4, "len")],  # held-out length: 6-7 intents, training has <= 5
    "len2": [(5, "len")],  # held-out length: 8-9 intents
    "len3": [(6, "len")],  # held-out length: 11 intents (2.2x the training maximum)
    "len4": [(7, "len")],  # stress: 13 intents
    "len5": [(8, "len")],  # stress: 15 intents
    "len6": [(9, "len")],  # stress: 17 intents
}


def online_metrics(model: S1Model, device: torch.device, suites=("iid", "comp", "len"), episodes: int = 100,
                   workers: int = 8, seed_base: int = TEST_SEED_BASE, sample: bool = False,
                   perturb: float = 0.0) -> dict:
    from .rollout import Policy, run_episodes
    from .runtime.client import VecEnv

    policy = Policy(model, device)
    if model.cfg.ui:  # UI-level model: hidden FreeCAD GUIs (keep workers small, ~1 GB each)
        from .ui.env import UiVecEnv

        vec = UiVecEnv(workers)
    else:
        vec = VecEnv(workers)
    results = []
    try:
        for suite in suites:
            for level, split in SUITES[suite]:
                base = seed_base + level * 100_000 + SPLIT_SEED_OFFSET.get(split, 0)
                rng = random.Random(base)  # per-suite perturbation stream: subsets reproduce
                for start in range(0, episodes, workers):
                    n = min(workers, episodes - start)
                    specs = [{"level": level, "split": split, "seed": base + start + i} for i in range(n)]
                    if n < workers:  # keep lockstep simple: idle workers replay a spec, results dropped
                        specs += [specs[0]] * (workers - n)
                    batch = run_episodes(policy, vec, specs, sample=sample, rng=rng, perturb=perturb)[:n]
                    for r in batch:
                        r.level = f"{suite}-L{level}"
                    results += batch
    finally:
        vec.close()
    report = summarize(results)
    if hasattr(vec, "recovered"):  # UI envs: FreeCAD crashes, and how many replay recovered
        report["freecad_crashes"] = {"raw": vec.crashes, "recovered": vec.recovered}
    report["failures"] = [{"suite_level": r.level, "features": r.features, "outcome": r.outcome, "iou": r.iou,
                           "steps": r.steps, "history": r.history} for r in results if not r.success]
    return report


def _rates(rs) -> dict:
    return {
        "episodes": len(rs),
        "success": round(float(np.mean([r.success for r in rs])), 4),
        "clean_success": round(float(np.mean([r.clean for r in rs])), 4),
        # clean success with no wrong policy decision along the way (injected steps excluded)
        "zero_deviation_success": round(float(np.mean([r.clean and r.deviations == 0 for r in rs])), 4),
        "clean_within_expert_plus_2": round(float(np.mean([r.clean and r.steps <= r.expert_steps + 2 for r in rs])), 4),
    }


def summarize(results) -> dict:
    out = {}
    by_level = defaultdict(list)
    for r in results:
        by_level[r.level].append(r)
    for level, rs in sorted(by_level.items()):
        outcomes = defaultdict(int)
        for r in rs:
            outcomes[r.outcome] += 1
        ok = [r for r in rs if r.success]
        rules = defaultdict(list)
        for r in rs:
            if r.rule:
                rules[r.rule].append(r)
        out[level if isinstance(level, str) else f"L{level}"] = {
            **_rates(rs),
            "mean_iou": round(float(np.mean([r.iou for r in rs])), 4),
            "on_policy_agreement": round(float(np.mean([r.agreement for r in rs])), 4),
            "steps_over_expert": round(float(np.mean([r.steps / max(r.expert_steps, 1) for r in ok])), 3) if ok else None,
            "outcomes": dict(outcomes),
            **({"by_rule": {k: _rates(v) for k, v in sorted(rules.items())}} if rules else {}),
        }
    out["overall_success"] = round(float(np.mean([r.success for r in results])), 4) if results else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help=".pt checkpoint, exported directory, or Hugging Face repo id")
    ap.add_argument("--data", help="held-out datagen directory for per-step accuracy")
    ap.add_argument("--episodes", type=int, default=100, help="online episodes per level (0 to skip)")
    ap.add_argument("--suites", nargs="+", default=["iid", "comp", "len"], choices=list(SUITES),
                    help="iid: levels 1-3; comp: held-out compositions; len: held-out length (level 4)")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="probability of replacing the policy's action by a random valid one")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--index-eval", choices=["expected", "identity"],
                    help="override the test-time mapping of randomized positions/ordinals")
    ap.add_argument("--out", help="write the report JSON here")
    args = ap.parse_args()
    device = select_device(args.device)
    if args.ckpt.endswith(".pt"):
        model = load_checkpoint(args.ckpt, device)
    else:  # exported directory or Hugging Face repo id
        from .model.net import from_pretrained

        model = from_pretrained(args.ckpt, device)
    if args.index_eval:
        model.cfg.index_eval = args.index_eval
    report = {}
    if args.data:
        report["per_step"] = offline_metrics(model, load_dataset(args.data, **model.cfg.feature_opts()),
                                             device)
        print(json.dumps({"per_step": report["per_step"]}, indent=2))
    if args.episodes:
        random.seed(0)
        report["episodes"] = online_metrics(model, device, tuple(args.suites), args.episodes, args.workers,
                                            sample=args.sample, perturb=args.perturb)
        print(json.dumps({"episodes": report["episodes"]}, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()

"""Phase 1: synthetic state -> action data from scripted FreeCAD workflows.

Each episode samples a goal (curriculum level 1-3), builds the target solid
with the clean expert, then replays the task from a randomized start
condition while the expert labels every state. With per-episode probability
the executed action is replaced by a random valid action (DART-style noise),
so the data also covers mistakes and their recovery (undo, re-select,
switch back to the right workbench). Only the expert's label is stored; the
noisy action is never a training target.

Run from the project venv; it fans out shards to FreeCAD's interpreter:

    python -m freecad_s1.datagen --out data/train --episodes 2000 3000 4000 --workers 8

Output: gzip JSONL, one record per labeled step:
    {"ep", "level", "step", "state", "actions", "acceptable", "noise", "progress"}
Goals matching a held-out composition rule (goals.heldout_composition) are
never generated, so they remain a clean generalization test.
The goal is written once per episode (record with "goal") and referenced by
"ep" afterwards to keep files small.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

NOISE_LEVELS = [0.0, 0.0, 0.1, 0.2, 0.3]


def run_shard(out: Path, shard: int, episodes: list[int], seed: int, session=None, target_session=None,
              budget_fn=None) -> dict:
    """Generate one shard. `session` defaults to a HeadlessSession; the UI
    data generator (freecad_s1/ui/datagen.py) passes a UiSession plus a
    headless `target_session` for building target solids and its own step
    budget."""
    from .goals import sample_start
    from .runtime.episode import GoalBuildError, new_episode, sample_feasible_goal, score, step_budget
    from .runtime.session import HeadlessSession

    rng = random.Random(seed * 1000 + shard)
    session = session or HeadlessSession()
    budget_fn = budget_fn or step_budget
    path = out / f"shard{shard:03d}.jsonl.gz"
    n_records = n_eps = n_success = 0
    t0 = time.time()
    with gzip.open(path, "wt") as fh:
        for level, count in enumerate(episodes, start=1):
            for k in range(count):
                try:
                    if target_session is None:
                        goal, start, target = new_episode(session, level, rng)
                    else:
                        goal, target = sample_feasible_goal(target_session, level, rng)
                        start = sample_start(rng, level)
                        session.reset(goal, start)
                except GoalBuildError:
                    continue
                ep = f"{seed}-{shard}-{level}-{k}"
                noise = rng.choice(NOISE_LEVELS)
                budget = budget_fn(goal, start) * (3 if noise else 1)
                fh.write(json.dumps({"ep": ep, "goal": goal.to_json()}) + "\n")
                for t in range(budget):
                    state = session.state()
                    acceptable = session.expert()
                    if not acceptable:
                        break  # unrecoverable (beyond FreeCAD's undo history)
                    actions = session.valid_actions(state)
                    rec = {"ep": ep, "level": level, "step": t, "state": state.to_json(),
                           "actions": actions, "acceptable": acceptable, "noise": noise,
                           "progress": session.progress()}
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    n_records += 1
                    if rng.random() < noise:
                        choices = [a for a in actions if a != "Done"]
                        action = rng.choice(choices)
                    else:
                        action = rng.choice(acceptable)
                    if session.step(action)["done"]:
                        break
                n_eps += 1
                n_success += int(session.done and score(session, target)["match"])
    session.shutdown()
    summary = {"shard": shard, "episodes": n_eps, "records": n_records,
               "expert_success": n_success / max(n_eps, 1), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary))
    return summary


def launch(args) -> None:
    from .runtime.fcenv import freecad_env, freecad_python

    py, _ = freecad_python()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_worker = [[c // args.workers + (1 if w < c % args.workers else 0) for c in args.episodes]
                  for w in range(args.workers)]
    procs = []
    for w in range(args.workers):
        cmd = [py, "-m", "freecad_s1.datagen", "--shard", str(w), "--out", str(out), "--seed", str(args.seed),
               "--episodes", *map(str, per_worker[w])]
        procs.append(subprocess.Popen(cmd, env=freecad_env(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
    for p in procs:
        stdout, _ = p.communicate()
        lines = [ln for ln in stdout.splitlines() if ln.startswith("{")]
        print(lines[-1] if lines else f"shard failed (exit {p.returncode})")
        if p.returncode:
            sys.exit(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs=3, default=[1000, 1500, 2000], metavar=("L1", "L2", "L3"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.shard is None:
        launch(args)
    else:
        run_shard(Path(args.out), args.shard, args.episodes, args.seed)


if __name__ == "__main__":
    main()

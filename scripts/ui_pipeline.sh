#!/usr/bin/env bash
# UI-level model: SFT + DAgger on data/ui_train, then evaluation through the
# real (hidden) FreeCAD UI with and without injected random actions.
# Same recipe as scripts/train_final.sh plus --ui. FreeCAD workers are kept
# few (WORKERS, default 3, ~1.3 GB each, capped by the watchdog) to stay well
# within memory; that trades wall-clock time for safety.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=${OUT:-runs/ui_M}
WORKERS=${WORKERS:-3}
EPISODES=${EPISODES:-100}
export S1_MEM_LIMIT_GB=${S1_MEM_LIMIT_GB:-2.5}  # per-FreeCAD watchdog cap for the DAgger / eval envs
mkdir -p "$OUT"
$PY -m freecad_s1.train_sft --ui --data data/ui_train --out "$OUT" --epochs 4 \
  --pos-mode rand --ordinal --invariant-numerics --modular --pointer done --index-eval identity --type-dropout 0.15 \
  --dagger-rounds 2 --dagger-episodes 400 --dagger-epochs 2 --dagger-workers "$WORKERS"
$PY -m freecad_s1.evaluate --ckpt "$OUT/last.pt" --episodes "$EPISODES" --workers "$WORKERS" \
  --suites iid comp comp2 len len2 len3 --out "$OUT/eval.json"
$PY -m freecad_s1.evaluate --ckpt "$OUT/last.pt" --episodes "$EPISODES" --workers "$WORKERS" --perturb 0.2 \
  --suites iid comp comp2 len len2 len3 --out "$OUT/eval_perturb.json"
$PY scripts/summarize.py "$OUT/eval.json" "$OUT/eval_perturb.json"

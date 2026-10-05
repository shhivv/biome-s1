#!/usr/bin/env bash
# Mesa-S1, end to end: the three training stages it was built with, then the
# evaluation through the hidden FreeCAD GUI. Needs data/ui_train from
#   python -m freecad_s1.ui.datagen --out data/ui_train --episodes 2000 4000 6000 --workers 3 --mem-gb 2.5
# At most WORKERS (3) FreeCADs run at once, each capped by the watchdog.
#
#   stage 1  SFT (4 epochs) + 2 DAgger rounds, expert actions mixed in (beta 0.5, 0.25)
#   stage 2  + recent dialog edits name their field; 1 epoch + 3 policy-only DAgger rounds
#   stage 3  + an open dialog's feature does not count as built, + numeric fields say whether they
#            hold the parameter stage's value; 2 epochs + 3 policy-only DAgger rounds, half of the
#            rollout batches with 20% random actions (recovery practice)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=${OUT:-runs/mesa}
WORKERS=${WORKERS:-3}
export S1_MEM_LIMIT_GB=${S1_MEM_LIMIT_GB:-2.5}
ARCH="--ui --pos-mode rand --ordinal --invariant-numerics --modular --pointer done --index-eval identity --type-dropout 0.15"
DAGGER="--dagger-episodes 400 --dagger-epochs 2 --dagger-workers $WORKERS"

$PY -m freecad_s1.train_sft $ARCH --data data/ui_train --out "$OUT/stage1" --epochs 4 \
  --dagger-rounds 2 $DAGGER
$PY -m freecad_s1.train_sft $ARCH --ui-recent-fields --init "$OUT/stage1/last.pt" --data data/ui_train \
  --out "$OUT/stage2" --epochs 1 --warmup 200 --lr 5e-4 --dagger-rounds 3 $DAGGER --beta-start 0
$PY -m freecad_s1.train_sft $ARCH --ui-recent-fields --ui-pending-feature --ui-param-match \
  --init "$OUT/stage2/last.pt" --data data/ui_train --out "$OUT/stage3" --epochs 2 --warmup 200 --lr 5e-4 \
  --dagger-rounds 3 $DAGGER --beta-start 0 --dagger-perturb 0.2 --dagger-perturb-frac 0.5

SUITES="iid comp comp2 len len2 len3"
$PY -m freecad_s1.evaluate --ckpt "$OUT/stage3/last.pt" --episodes 100 --workers "$WORKERS" --suites $SUITES \
  --out "$OUT/eval.json"
$PY -m freecad_s1.evaluate --ckpt "$OUT/stage3/last.pt" --episodes 100 --workers "$WORKERS" --perturb 0.2 \
  --suites $SUITES --out "$OUT/eval_perturb.json"
$PY scripts/summarize.py "$OUT/eval.json" "$OUT/eval_perturb.json"
# export: python -c "from freecad_s1.model.net import *; save_pretrained('release/mesa-s1', load_checkpoint('$OUT/stage3/last.pt'))"

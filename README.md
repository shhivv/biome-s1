# Taiga-S1

![Taiga-S1: parts the model built in FreeCAD](release/hf/assets/cover.png)

**A 1.2M-parameter model that builds CAD parts in FreeCAD.** Weights: [huggingface.co/shhivv/taiga-s1](https://huggingface.co/shhivv/taiga-s1).

*Taiga-S1 is an experiment in whether small, fast decision models can be useful for computer-use agents: a planner decides what to do, and a tiny model handles the step-by-step execution. FreeCAD is the testbed. The next step is the same approach for applications without a scripting API, using the operating system's accessibility tree as the interface.*

Taiga-S1 is the fast "System 1" layer for a CAD agent. You give it a goal, an ordered list of features like *"plate 40×30×10 → Ø6 hole at (10, 0) → polar pattern ×6 → fillet the top edges"*. It builds the part command by command: select a plane, sketch, draw, constrain, pad, pattern, fillet. At every step it reads FreeCAD's live state and scores the commands currently available, in a single forward pass (~1 ms on CPU). No LLM, no vision model, no screenshots.

- **Handles longer goals than it trained on.** Trained on goals of up to 5 features, it completed all 100 held-out 11-feature goals (~55 commands) and 95% of 17-feature goals.
- **Recovers from mistakes.** With 20% of its actions replaced by random ones, it notices the damage, undoes it and still finishes 86–100% of parts.
- **Handles unseen combinations.** Feature pairings that never appear in training: 90–100%.
- **Drives the real FreeCAD app** over a local socket and builds parts live.

The Python package is named `freecad_s1`.

## Results

![Parts built correctly vs. goal length](release/hf/assets/length.png)

| Goal | Built correctly | With 20% random actions injected |
|---|---|---|
| Parts like the training set (1–5 features) | 100% | 93–100% |
| 6–7 features | 100% | 86% |
| 8–9 features | 100% | 87% |
| 11 features (~55 commands) | 100% | 90% |
| 13 / 15 / 17 features | 100 / 100 / 95% | – |
| Feature combinations never seen in training | 90–100% | 94–97% |

"Built correctly" means the model emitted `Done`, the final solid matches the target (volumetric IoU ≥ 0.99), and no stray objects remain. Rows are 100 fresh synthetic goals (same feature vocabulary as training, longer or recombined) in FreeCAD 1.1, except the 13–17-feature row, which is 60 per length. Per-step accuracy against the teacher's choices is 99.8%. The evidence is in `results/`.

## How it works

```
FreeCAD (App/Gui API) ──snapshot──► State ─┐
goal (ordered feature intents + target) ───┼─► StateEncoder (typed-token transformer) ─► context
valid commands ──► ActionEncoder ─► options ─► decoder (active goal item + state) ─► score per command ─► softmax
```

- **State as tokens.** A snapshot turns the document into a typed token sequence: session globals, one token per feature-tree object in build order (sketch geometry and constraints, feature parameters), the selection, the last 8 commands, the target part and the goal items.
- **Commands as options.** Candidates are the commands valid right now, the headless equivalent of FreeCAD's toolbars plus `isCommandActive`. Each one gets a score in one pass, so the action space can vary step to step.
- **A modular "done?" policy.** Each goal item asks "am I built yet?" by matching its ordinal against the feature tree, and the model acts on the first one that isn't. The decision depends only on the state and that one active item, which is what lets it handle goals longer than the ones it trained on.
- **Command type only.** The model picks the command. Numeric values (sizes, depths, radii, counts) come from the goal item via `runtime/params.py`.

### What made it generalize

![What made it generalize: 11-intent goals](release/hf/assets/ablation.png)

The ablations were motivated by the literature on length and compositional generalization. Each variant was trained the same way (SFT, 3 epochs, same data):

1. **Randomized position IDs during training** ([Ruoss et al. 2023](https://arxiv.org/abs/2305.16843)). Position numbers the model had never seen were what broke it on longer parts. Keeping indices consecutive at test time matters too.
2. **Coupled ordinals.** Goal item *k* and the *k*-th feature in the tree share an index ([position coupling](https://arxiv.org/abs/2405.20671), [index hints](https://arxiv.org/abs/2402.09371)).
3. **The modular done-head policy.** It made the longest goals reliable across training seeds. A softmax pointer over goal items ([stage-wise modular policies](https://arxiv.org/html/2607.29687)) did not work as well, because it has to learn "next = built + 1", which breaks past the trained length.
4. **Factorized feature types.** Category embeddings (additive / subtractive / dressup / pattern) plus type dropout, for feature pairings it hasn't seen.

Also tried without gains: a progress-estimation head alone ([Ma et al. 2019](https://arxiv.org/abs/1901.03035)) and length-invariant numeric features.

## Training

- **Data.** 24k synthetic modeling sessions scripted in headless FreeCAD, about 590k decisions. A state-based scripted teacher labels the set of acceptable next commands at every step. Random mistakes are mixed into 60% of sessions so the model also learns to recover (undo off-plan changes, re-select, switch back to the right workbench).
- **Loss.** Multi-label NLL, since steps like sketch constraints can go in any order.
- **DAgger.** Supervised training is followed by 2 rounds where the model drives live FreeCAD and the teacher labels what it actually visits.
- **Calibration.** A softmax temperature is fitted on held-out on-policy states and stored in `config.json`, so `Policy.score()` returns calibrated probabilities.

## Layout

| Path | What |
|---|---|
| `freecad_s1/schema.py`, `actions.py` | State/goal records and the command catalogue; `enumerate_actions(state)` returns the valid set |
| `freecad_s1/goals.py`, `expert.py` | Goal sampler and the state-based teacher |
| `freecad_s1/runtime/` | FreeCAD runtime: headless and GUI sessions, snapshot, executors, the worker/client RPC and the GUI socket server |
| `freecad_s1/datagen.py` | Synthetic data generation |
| `freecad_s1/model/` | Featurization and the PyTorch model, plus `from_pretrained`/`save_pretrained` |
| `freecad_s1/ui/` | Mesa-S1's runtime: acts on FreeCAD's live widget tree, see [Mesa-S1](#mesa-s1-the-same-model-operating-freecads-interface) |
| `release/mesa-s1/` | Mesa-S1: weights, config, card, eval results |
| `freecad_s1/train_sft.py`, `evaluate.py`, `rollout.py` | Training (SFT + DAgger), evaluation, closed-loop rollouts |
| `scripts/` | Pipelines (`train_final.sh`, `final_eval.sh`, `chart_evals.sh`, `ablate.sh`), GUI demo, calibration |
| `viz/` | Remotion project that renders the cover and charts from `results/` (`scripts/export_chart_data.py`, then `viz/render.sh`) |
| `release/hf/` | The published model: weights, config, card, charts |

## Mesa-S1: the same model, operating FreeCAD's interface

Taiga-S1 picks FreeCAD commands, and the GUI runtime fills in their task dialogs behind the scenes. **Mesa-S1** (`release/mesa-s1/`) does everything Taiga-S1 does, but through the interface itself: it clicks toolbar commands, types into dialog fields, picks dropdown entries, ticks check boxes and clicks OK, or Cancel when it opened the wrong dialog. Same architecture, same size (1.2M parameters, ~1 ms per decision).

| Goal | Mesa-S1 | With 20% random actions | Taiga-S1 |
|---|---|---|---|
| Parts like the training set (levels 1–3) | 100% | 98–100% | 100% |
| Feature combinations never seen in training | 100% | 98–100% | 90–100% |
| 6–7 / 8–9 / 11 features | 100 / 99 / 99% | 97 / 95 / 92% | 100 / 100 / 100% |

99.75% of 800 held-out parts built correctly through the GUI; the two failures were FreeCAD crashes, which the runtime otherwise recovers by restarting FreeCAD and replaying the episode (55 of 69 recovered). Full card: [`release/mesa-s1/README.md`](release/mesa-s1/README.md).

How it works (`freecad_s1/ui/`):
- **Elements, not commands.** The options are read from FreeCAD's live Qt widget tree: `cmd:PartDesign_Pad` (the toolbar's QAction), `set:lengthEdit`, `opt:changeMode=Through all`, `toggle:checkBoxReversed`, `click:OK` / `click:Cancel`, `wb:…` and `Done`, each with its live value. Interactions the widget tree cannot see (picking faces in the 3D view, sketch geometry and constraints) stay semantic `canvas:` actions.
- **Same teacher.** The command-level expert decides what to build; `ui/teacher.py` turns that into clicks: fill the fields that differ from their targets (in any order), then OK, or Cancel a dialog opened by mistake. Numbers typed into fields come from the parameter stage, as in Taiga-S1, and each numeric field tells the model whether it already holds that value; dropdown entries, check boxes, OK/Cancel and Undo are the model's choice.
- **Hidden GUI, with guard rails.** Every FreeCAD runs hidden (`open -g -j` on macOS, `xvfb-run` elsewhere) under a watchdog that kills it above a memory cap or time limit, and when the launcher exits (`ui/launch.py`). FreeCAD crashes are recovered by restart and replay (`ui/env.py`).

```bash
python scripts/smoke_ui.py --n 2                       # teacher episodes through the real UI
python -m freecad_s1.ui.datagen --out data/ui_train --episodes 2000 4000 6000 --workers 3 --mem-gb 2.5
scripts/train_mesa.sh                                   # Mesa-S1 recipe: 3 training stages (SFT + DAgger) + evaluation
FREECAD_S1_REPO=$PWD FREECAD_S1_UI=1 /Applications/FreeCAD.app/Contents/MacOS/FreeCAD scripts/freecad_gui_server.FCMacro &
python scripts/gui_demo.py --model release/mesa-s1     # watch it build a part
```

A UI step takes ~60 ms in FreeCAD (vs ~3 ms headless), so UI data is slower to generate: 12k episodes took ~8 h on 3 workers.

## Quick start

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q                       # unit tests + FreeCAD integration tests

# Watch it drive the FreeCAD GUI
FREECAD_S1_REPO=$PWD /Applications/FreeCAD.app/Contents/MacOS/FreeCAD scripts/freecad_gui_server.FCMacro &
.venv/bin/python scripts/gui_demo.py --model shhivv/taiga-s1 --level 3 --split iid --seed 7

# Reproduce the model: data -> SFT + DAgger -> eval
.venv/bin/python -m freecad_s1.datagen --out data/gen_train --episodes 4000 8000 12000 --workers 8 --seed 2
.venv/bin/python -m freecad_s1.datagen --out data/gen_test  --episodes 300 600 900    --workers 8 --seed 3
EXTRA="--type-dropout 0.15" OUT=runs/final_M ./scripts/train_final.sh     # ~25 min on an M-series Mac
.venv/bin/python -c "from freecad_s1.model.net import load_checkpoint, save_pretrained; \
  save_pretrained('release/hf', load_checkpoint('runs/final_M/last.pt'))"
./scripts/final_eval.sh                             # every suite + calibration -> results/
```

FreeCAD is found at `/Applications/FreeCAD.app` or on common Linux paths; otherwise set `FREECAD_PYTHON` and `FREECAD_LIB`. The torch process never imports FreeCAD. It runs FreeCAD's own interpreter as a worker process at ~3 ms per action headless.

Usage from Python, and what to pass in (`state`, `goal`, `actions`), is on the [model card](https://huggingface.co/shhivv/taiga-s1).

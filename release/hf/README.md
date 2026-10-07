---
license: mit
library_name: pytorch
pipeline_tag: other
tags:
  - cad
  - freecad
  - agent
  - imitation-learning
  - length-generalization
---

# Taiga-S1

![Taiga-S1: parts the model built in FreeCAD](assets/cover.png)

**A 1.2M-parameter model that builds CAD parts in FreeCAD.**

*Taiga-S1 is an experiment in whether small, fast decision models can be useful for computer-use agents: a planner decides what to do, and a tiny model handles the step-by-step execution. FreeCAD is the testbed. The next step is the same approach for applications without a scripting API, using the operating system's accessibility tree as the interface.*

Taiga-S1 is part of the **Biome-S1** model family, with [Mesa-S1](https://huggingface.co/shhivv/mesa-s1), which works FreeCAD's buttons and dialogs itself. Code: [github.com/shhivv/biome-s1](https://github.com/shhivv/biome-s1).

Taiga-S1 is the fast "System 1" layer for a CAD agent. You give it a goal, an ordered list of features like *"plate 40×30×10 → Ø6 hole at (10, 0) → polar pattern ×6 → fillet the top edges"*. It builds the part command by command: select a plane, sketch, draw, constrain, pad, pattern, fillet. At every step it reads FreeCAD's live state (feature tree, selection, sketch constraints, workbench) and picks the next command from the ones currently available.

- **Tiny and fast.** 1.2M parameters, trained from scratch, ~1 ms per decision on a CPU. No LLM, no vision model, no screenshots.
- **Handles longer goals than it trained on.** Trained on goals of up to 5 features, it completed all 100 held-out 11-feature goals (~55 commands) and 95% of 17-feature goals.
- **Recovers from mistakes.** With 20% of its actions replaced by random ones, it notices the damage, undoes it and finishes 86–100% of parts.
- **Runs in the real FreeCAD app.** It drives the FreeCAD GUI over a local socket and builds parts live.

## Results

![Parts built correctly vs. goal length](assets/length.png)

| Goal | Built correctly | With 20% random actions injected |
|---|---|---|
| Parts like the training set (1–5 features) | 100% | 93–100% |
| 6–7 features | 100% | 86% |
| 8–9 features | 100% | 87% |
| 11 features (~55 commands) | 100% | 90% |
| 13 / 15 / 17 features | 100 / 100 / 95% | – |
| Feature combinations never seen in training | 90–100% | 94–97% |

"Built correctly" means the model finished and the final solid matches the target exactly (volumetric IoU ≥ 0.99, no stray objects). Each row is 100 fresh synthetic goals (same feature vocabulary as training, longer or recombined) in FreeCAD 1.1 (60 per length for 13–17 features). Per-step accuracy against the teacher's choices is 99.8%.

## What made it generalize

![What made it generalize](assets/ablation.png)

1. **Randomized position IDs during training** ([Ruoss et al. 2023](https://arxiv.org/abs/2305.16843)). Position numbers the model had never seen were what broke it on longer parts.
2. **Coupled ordinals.** Goal item *k* and the *k*-th feature in the tree share an index ([position coupling](https://arxiv.org/abs/2405.20671)).
3. **A modular "done?" policy.** Each goal item asks "am I built yet?", and the model acts on the first one that isn't. This is what made the longest goals reliable across training seeds.
4. **Factorized feature types.** Category embeddings plus type dropout help with feature pairings it hasn't seen.

## Usage

```python
from freecad_s1.model.net import from_pretrained
from freecad_s1.rollout import Policy

model = from_pretrained("shhivv/taiga-s1")
policy = Policy(model, device="cpu")

probs = policy.score(state, goal, actions)   # {command: probability}, best first
next_command = next(iter(probs))
```

What to pass in:
- `state`: a snapshot of the FreeCAD session, from the included runtime (`freecad_s1.runtime`).
- `goal`: the ordered feature list, plus a rough size of the finished part (bounding box, volume). Estimates are fine.
- `actions`: the commands currently available in FreeCAD, as returned by the runtime's `valid_actions()`.

The returned probabilities are calibrated. The fitted temperature is stored in `config.json`.

## How it was trained

- **Data.** 24k synthetic modeling sessions scripted in headless FreeCAD, about 590k decisions. A scripted teacher labels the right next command at every step, and random mistakes are mixed in so the model also learns to recover.
- **Training.** Supervised training, then two rounds of DAgger: the model drives FreeCAD itself and the teacher corrects what it gets wrong.
- **Architecture.** A 3-layer transformer encodes the session state and the goal; candidate commands attend to the state and the active goal item, and each gets one score.

**Scope.** It covers FreeCAD PartDesign workflows: sketches (rectangle, circle, hexagon), pad, pocket, hole, revolve, linear and polar patterns, mirror, fillet, chamfer and shell. Taiga-S1 chooses the command; numeric values come from the goal.

## Citation

```bibtex
@misc{taiga_s1_2026,
  title  = {Taiga-S1: a small next-action model for CAD},
  author = {Shanmugam, Shiv},
  year   = {2026}
}
```

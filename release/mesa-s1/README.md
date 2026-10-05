---
license: mit
library_name: pytorch
pipeline_tag: other
tags:
  - cad
  - freecad
  - agent
  - computer-use
  - gui-agent
  - imitation-learning
  - length-generalization
---

# Mesa-S1

**A 1.2M-parameter model that builds CAD parts by operating FreeCAD's interface: toolbar buttons, dialog fields, dropdowns, OK and Cancel.**

*Mesa-S1 is an experiment in whether small, fast decision models can operate real application interfaces for computer-use agents: a planner decides what to build, and a tiny model does the clicking. It does everything [Taiga-S1](https://huggingface.co/shhivv/taiga-s1) does, but where Taiga-S1 picks FreeCAD commands and lets code fill in each dialog, Mesa-S1 works the dialogs itself. FreeCAD is the testbed; the next step is applications without a scripting API, read through the operating system's accessibility tree.*

You give it a goal, an ordered list of features like *"plate 40×30×12 → Ø4 hole at (8, 3) → linear pattern ×3 over 20 mm → fillet the top edges"*. It builds the part step by step. At every step it reads FreeCAD's live state (the feature tree and selection, plus the interface: which toolbar commands are enabled, which task dialog is open, and the current value of every field in it) and picks the next element to act on:

| Element | What happens in FreeCAD |
|---|---|
| `cmd:PartDesign_Pad` | the toolbar button's action is triggered; FreeCAD opens its Pad dialog |
| `set:lengthEdit` | a value is typed into the dialog's length field |
| `opt:changeMode=Through all` | a dropdown entry is chosen |
| `toggle:checkBoxReversed` | a check box is clicked |
| `click:OK` / `click:Cancel` | the dialog is committed, or backed out of when it was opened by mistake |

- **Tiny and fast.** 1.2M parameters, trained from scratch, ~1 ms per decision on a CPU. No LLM, no vision model, no screenshots.
- **Everything Taiga-S1 does, through the real interface.** Same goals, same feature vocabulary, same test suites, at essentially the same accuracy: 99.75% of 800 held-out parts built correctly.
- **Handles longer goals than it trained on.** Trained on goals of up to 5 features, it builds 99% of 11-feature goals (~80+ interface steps).
- **Recovers from mistakes.** With 20% of its actions replaced by random clicks, it notices the damage (cancels the wrong dialog, undoes the wrong feature) and still builds 92–100% of parts.

## Results

| Goal | Built correctly | With 20% random actions injected | Taiga-S1 (commands, for reference) |
|---|---|---|---|
| Parts like the training set (levels 1 / 2 / 3, up to 5 features) | 100 / 100 / 100% | 100 / 99 / 98% | 100% (98–100% perturbed) |
| Feature combinations never seen in training | 100% | 100% | 90% (96%) |
| Another unseen pairing | 100% | 98% | 100% (95%) |
| 6–7 features | 100% | 97% | 100% (88%) |
| 8–9 features | 99% | 95% | 100% (97%) |
| 11 features | 99% | 92% | 100% (95%) |

"Built correctly" means the model issued `Done` and the final solid matches the target exactly (volumetric IoU ≥ 0.99). Each row is 100 fresh synthetic goals in FreeCAD 1.1, driven through the GUI. Per-step accuracy against the teacher's choices is 99.86% on held-out states.

Across the 800 clean episodes the model made no wrong decision that cost a part; both failures were FreeCAD crashes. **FreeCAD crashes.** Driven through its GUI, FreeCAD 1.1 segfaults in roughly 3–5% of long episodes, inside its own geometry and dialog code. The runtime recovers like FreeCAD's autosave: it restarts FreeCAD, replays the episode so far (episodes are deterministic) and retries the action. In the clean evaluation FreeCAD crashed 69 times, 55 were recovered, and the episodes it could not recover count as failures above. With random actions injected: 79 crashes, 67 recovered.

## Usage

```python
from freecad_s1.model.net import from_pretrained
from freecad_s1.rollout import Policy

model = from_pretrained("HF_REPO_ID")
policy = Policy(model, device="cpu")

probs = policy.score(state, goal, elements)   # {element: probability}, best first
next_element = next(iter(probs))
```

What to pass in:
- `state`: a snapshot of the FreeCAD session from the included UI runtime (`freecad_s1.ui.session.UiSession`, which runs inside the FreeCAD GUI), including the open dialog and its fields.
- `goal`: the ordered feature list, plus a rough size of the finished part (bounding box, volume). Estimates are fine.
- `elements`: the interface elements currently available, as returned by `UiSession.valid_actions()`.

The returned probabilities are calibrated: a softmax temperature (T = 2.66, stored in `config.json`) was fitted on 16k on-policy states with 20% injected random actions, halving the calibration error (ECE 0.092 → 0.046). On those states, many of them mid-recovery, the model's top choice matches the teacher's 88.5% of the time; several recoveries are valid (e.g. Cancel vs. undoing a field change) but the teacher accepts only its own.

To watch it build a part in the FreeCAD GUI (code: [github.com/shhivv/taiga](https://github.com/shhivv/taiga)):

```bash
FREECAD_S1_REPO=$PWD FREECAD_S1_UI=1 /Applications/FreeCAD.app/Contents/MacOS/FreeCAD scripts/freecad_gui_server.FCMacro &
python scripts/gui_demo.py --model release/mesa-s1 --level 3 --split iid --seed 7
```

## How it was trained

- **Data.** 12k synthetic modeling sessions built through FreeCAD's interface in hidden FreeCAD instances, about 391k decisions. A scripted teacher labels the right next element at every step (any order where order doesn't matter, e.g. which dialog field first). Random mistakes are mixed into a quarter of the sessions so the model also learns to recover: cancel a dialog opened by mistake, undo a wrong feature.
- **Training.** Supervised training, then DAgger: the model drives FreeCAD on its own, the teacher labels every state it reaches, including after injected random clicks, and it retrains on its own mistakes.
- **Architecture.** Taiga-S1's architecture: a 3-layer transformer encodes the session state and the goal; each candidate element attends to the state and the active goal item and gets one score. An element is encoded by the command it stands for, the word pieces of its id, its role (command, field, dropdown entry, check box, button) and its live value (the number in a field, whether an entry is selected or a box checked).

**Scope and limits.**
- It covers FreeCAD PartDesign workflows: sketches (rectangle, circle, hexagon), pad, pocket, hole, revolve, linear and polar patterns, mirror, fillet, chamfer and shell.
- Mesa-S1 chooses which element to act on. The numbers typed into fields come from the goal (a parameter stage), as in Taiga-S1, and each numeric field tells the model whether it already holds that value.
- Picking faces and edges in the 3D view and drawing sketch geometry stay semantic actions (`canvas:…`), since the widget tree cannot see the 3D view.
- The interface is read from FreeCAD's Qt widget tree from inside the application, and actions are applied to those widgets. Applications without a Python runtime would need the operating system's accessibility tree instead.

## Citation

```bibtex
@misc{mesa_s1_2026,
  title  = {Mesa-S1: a small model that operates a CAD interface},
  author = {Shanmugam, Shiv},
  year   = {2026}
}
```

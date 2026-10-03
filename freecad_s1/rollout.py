"""Closed-loop episodes: the model drives live FreeCAD workers.

Used by evaluation (episode success), DAgger (collect expert labels on the
states the *policy* visits) and as the reference loop for PPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import Dataset
from .goals import heldout_composition
from .model.featurize import collate, make_example
from .model.net import S1Model
from .runtime.client import Episode, VecEnv
from .schema import State


class Policy:
    def __init__(self, model: S1Model, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def distributions(self, states, goals, action_lists) -> list[torch.Tensor]:
        self.model.eval()
        opts = self.model.cfg.feature_opts()
        batch = collate([make_example(s, g, a, **opts) for s, g, a in zip(states, goals, action_lists)])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits, _ = self.model(batch)
        probs = torch.softmax(logits, -1).cpu()
        return [probs[i, : len(a)] for i, a in enumerate(action_lists)]

    @torch.no_grad()
    def score(self, state, goal, actions: list[str], temperature: float | None = None) -> dict[str, float]:
        """Probability of each candidate in `actions` (any list of command
        strings, e.g. what the interface currently offers), highest first.
        Uses the model's calibrated temperature unless one is given."""
        temperature = temperature or self.model.cfg.temperature
        self.model.eval()
        batch = collate([make_example(state, goal, actions, **self.model.cfg.feature_opts())])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits, _ = self.model(batch)
        probs = torch.softmax(logits[0, : len(actions)].float().cpu() / temperature, -1)
        return dict(sorted(zip(actions, probs.tolist()), key=lambda t: -t[1]))

    def act(self, states, goals, action_lists, sample: bool = False) -> list[str]:
        out = []
        for p, acts in zip(self.distributions(states, goals, action_lists), action_lists):
            j = int(torch.multinomial(p, 1)) if sample else int(p.argmax())
            out.append(acts[j])
        return out


@dataclass
class EpisodeResult:
    level: int
    success: bool  # Done issued and final solid matches the target (IoU >= 0.99)
    clean: bool  # ... and the document holds no leftover off-plan objects
    iou: float
    steps: int
    budget: int
    expert_steps: int
    agreement: float  # fraction of steps where the policy picked an expert-acceptable action
    outcome: str  # success | wrong_geometry | budget | unrecoverable
    features: list[str]
    rule: str | None = None  # held-out composition rule the goal matches, if any
    deviations: int = 0  # policy steps whose action was not expert-acceptable (injected steps excluded)


def run_episodes(policy: Policy, vec: VecEnv, specs: list[dict], sample: bool = False,
                 collect: Dataset | None = None, beta: float = 0.0, rng=None,
                 perturb: float = 0.0) -> list[EpisodeResult]:
    """Run one episode per worker. `specs[i]` is a reset request (e.g.
    {"level": 2, "seed": 7}). With `collect`, every visited state is added
    with the expert's label (DAgger). `beta` mixes in the expert's action
    with that probability (DAgger's beta schedule). `perturb` replaces the
    policy's action by a random valid one with that probability (robustness
    test: the policy must notice and repair the damage); those steps don't
    count toward agreement and the step budget is doubled."""
    eps: list[Episode] = vec.reset(specs)
    expert_len = [ep.budget // 2 - 3 for ep in eps]  # step_budget = 2 * plan + 6
    if perturb > 0:
        for ep in eps:
            ep.budget *= 2
    policy_steps = [0] * len(eps)
    active = list(range(len(eps)))
    while active:
        cur = [eps[i] for i in active]
        chosen = policy.act([e.state for e in cur], [e.goal for e in cur], [e.actions for e in cur], sample=sample)
        for k, i in enumerate(active):
            e = eps[i]
            if collect is not None and e.expert:
                collect.add(make_example(e.state, e.goal, e.actions, e.expert, e.progress,
                                         **policy.model.cfg.feature_opts()), f"dagger-{specs[i].get('seed')}",
                            e.level, -1.0, e.expert)
            if beta > 0 and rng is not None and e.expert and rng.random() < beta:
                chosen[k] = e.expert[0]
            if perturb > 0 and rng is not None and rng.random() < perturb:
                chosen[k] = rng.choice([a for a in e.actions if a != "Done"])
            else:
                e.agree += int(chosen[k] in e.expert)
                policy_steps[i] += 1
        responses = vec.step(active, chosen)
        still = []
        for i, a, r in zip(active, chosen, responses):
            e = eps[i]
            e.steps += 1
            e.history.append(a)
            if r["state"] is not None:
                e.state, e.actions = State.from_json(r["state"]), r["actions"]
            e.progress = r.get("progress", -1)
            prev_expert, e.expert = e.expert, r["expert"]
            if r["info"].get("crashed"):  # FreeCAD died (UI envs); counted as a failure, reported separately
                e.outcome = "crashed"
            elif r["info"]["done"]:
                e.done = True
                e.outcome = "done_clean" if prev_expert == ["Done"] else "done"
            elif not e.expert:
                e.outcome = "unrecoverable"
            elif e.steps >= e.budget:
                e.outcome = "budget"
            else:
                still.append(i)
        active = still
    scores = vec.score(list(range(len(eps))))
    results = []
    for e, sc, n, ps in zip(eps, scores, expert_len, policy_steps):
        success = e.done and sc["match"]
        outcome = "success" if success else ("wrong_geometry" if e.done else e.outcome)
        results.append(EpisodeResult(e.level, success, success and e.outcome == "done_clean", sc["iou"], e.steps,
                                     e.budget, n, e.agree / max(ps, 1), outcome,
                                     [f.kind for f in e.goal.features], heldout_composition(e.goal), ps - e.agree))
    return results

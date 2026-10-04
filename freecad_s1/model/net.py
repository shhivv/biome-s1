"""Taiga-S1: option-attention next-action scorer (~1M params, from scratch).

    state tokens ──► StateEncoder (transformer) ──► context C (B, L, d)
    candidate actions ──► ActionEncoder ──► options O (B, N, d)
    O ──► ActionDecoder: [self-attn over the action set, cross-attn O→C, FFN] x k
      ──► per-action score ──► masked softmax over the valid set

Option attention: candidate actions query the state through attention and
get one logit each, so any number of options is scored in one forward pass.
The context is a typed token sequence built from structured FreeCAD state
plus the goal; options also attend to each other, so a score can depend on
what else is on offer (e.g. whether Undo is available); a value head on the
[CLS] token serves the PPO phase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ..actions import ACTION_IDS, CATEGORIES, SCOPES, WORD_VOCAB
from ..actions import CATALOGUE
from ..schema import NODE_TYPES
from .featurize import (A_OFFSETS, A_VOCAB, ACT_VEC_DIM, B_VOCAB, MAX_ORD, MAX_POS, N_SEGMENTS, NUM_DIM, SEG_GOAL,
                        SEG_NODE, SEG_RECENT, SEG_SEL, UI_ROLES, UI_VEC_DIM, UI_WORD_VOCAB)

# Functional category of each FreeCAD object type, in the action-category vocab.
NODE_CATEGORY = {
    "PartDesign::Body": "body", "Sketcher::SketchObject": "sketch",
    "PartDesign::Pad": "additive", "PartDesign::Revolution": "additive",
    "PartDesign::Pocket": "subtractive", "PartDesign::Groove": "subtractive", "PartDesign::Hole": "subtractive",
    "PartDesign::Fillet": "dressup", "PartDesign::Chamfer": "dressup", "PartDesign::Draft": "dressup",
    "PartDesign::Thickness": "dressup", "PartDesign::Mirrored": "pattern", "PartDesign::LinearPattern": "pattern",
    "PartDesign::PolarPattern": "pattern", "Part::Box": "part_primitive", "Part::Cylinder": "part_primitive",
}


def category_tables(b_vocab: int = B_VOCAB) -> tuple[torch.Tensor, torch.Tensor]:
    """(A_VOCAB,) and (b_vocab,) maps from token ids to CATEGORIES indices:
    node types and recent-action ids via `a`, selected-object types via `b`."""
    cat = {c: i for i, c in enumerate(CATEGORIES)}
    a_cat = torch.zeros(A_VOCAB, dtype=torch.long)
    b_cat = torch.zeros(b_vocab, dtype=torch.long)
    for i, t in enumerate(NODE_TYPES):
        c = cat.get(NODE_CATEGORY.get(t, ""), 0)
        a_cat[A_OFFSETS[SEG_NODE] + i] = c
        b_cat[i] = c
    for i, a in enumerate(ACTION_IDS):
        spec = CATALOGUE.get(a)
        a_cat[A_OFFSETS[SEG_RECENT] + i] = cat.get(spec.category, 0) if spec else 0
    return a_cat, b_cat


@dataclass
class S1Config:
    width: int = 128
    heads: int = 4
    enc_layers: int = 3
    dec_layers: int = 2
    ff: int = 384
    dropout: float = 0.1
    id_dropout: float = 0.1  # replace action ids by <unk> in training so word pieces carry meaning
    # Generalization options (all off = the v2 architecture):
    pos_mode: str = "abs"  # "abs" | "rand": randomized order-preserving positions (Ruoss et al. 2023)
    pos_table: int = 48  # positions per sequence simulated during training ("rand")
    ordinal: bool = False  # coupled goal/tree ordinals (position coupling, Cho et al. 2024)
    ord_table: int = 12  # ordinals simulated during training (randomized like positions)
    progress_head: bool = False  # progress pointer over goal intents + END (Ma et al. 2019)
    invariant_numerics: bool = False  # length-invariant numeric features (featurize.encode_state)
    # Stage-wise modular policy: state tokens never see goal tokens; each goal
    # token sees the state and itself; a pointer (supervised by the expert's
    # progress index, END sentinel included) picks the active intent, and the
    # action decoder sees only the state + that one intent. The decision is
    # then independent of how many intents the goal has.
    modular: bool = False
    # How the modular policy picks the active intent: "softmax" = pointer over
    # intents (needs a learned successor relation between ordinals); "done" =
    # per-intent "already built?" classifier (an ordinal equality match), then
    # the first not-done intent in goal order is active.
    pointer: str = "softmax"
    # Test-time mapping for randomized positions/ordinals: "expected" spreads
    # indices over the table (expected order statistics of the training
    # sample); "identity" keeps them consecutive (the tightest sample).
    index_eval: str = "expected"
    # Factorized object/command types: add a shared functional-category
    # embedding (additive/subtractive/dressup/pattern/...) to tree nodes,
    # selected objects and recent actions, and drop the specific type id with
    # this probability in training, so an unseen pairing of a specific type
    # (e.g. Hole then Mirrored) can fall back on its seen category.
    type_dropout: float = 0.0
    # Softmax temperature for reported probabilities (Policy.score); fitted
    # post hoc by scripts/calibrate.py. Does not change the argmax action.
    temperature: float = 1.0
    # UI-level model (freecad_s1/ui): options are interface elements with a
    # role and live widget values; recent tokens carry the element role.
    ui: bool = False
    ui_recent_fields: bool = False  # recent dialog edits carry which field they touched

    def feature_opts(self) -> dict:
        """Featurization options this model was trained with."""
        opts = {"invariant": self.invariant_numerics, "sentinel": self.modular}
        if self.ui:
            opts["ui"] = True
        if self.ui_recent_fields:
            opts["ui_recent"] = True
        return opts

    @property
    def b_vocab(self) -> int:
        return B_VOCAB + (len(UI_ROLES) if self.ui else 0)


def randomize_index(idx: torch.Tensor, table: int, span: int, training: bool, keep_zero: bool = False,
                    eval_mode: str = "expected") -> torch.Tensor:
    """Map indices 0..table-1 onto a sorted random subset of 0..span-1 (one
    subset per row) during training, and onto the expected value of those
    order statistics at eval time. Order is preserved; absolute values stop
    being tied to sequence length. With keep_zero, index 0 means "none" and
    indices 1.. are remapped into 1..span-1."""
    off = 1 if keep_zero else 0
    rng = span - off
    x = (idx - off).clamp(min=0, max=table - 1)
    if not training and eval_mode == "identity":
        return idx.clamp(max=span - 1)
    if training:
        sample = torch.rand(idx.shape[0], rng, device=idx.device).argsort(-1)[:, :table].sort(-1).values
        mapped = sample.gather(1, x) + off
    else:
        mapped = ((x + 1).float() * rng / (table + 1)).floor().long().clamp(max=rng - 1) + off
    return torch.where(idx < off, idx, mapped) if keep_zero else mapped


class StateEncoder(nn.Module):
    def __init__(self, cfg: S1Config) -> None:
        super().__init__()
        d = cfg.width
        self.a = nn.Embedding(A_VOCAB, d)
        self.b = nn.Embedding(cfg.b_vocab, d)
        self.seg = nn.Embedding(N_SEGMENTS, d)
        self.pos = nn.Embedding(MAX_POS, d)
        self.ord = nn.Embedding(MAX_ORD, d) if cfg.ordinal else None
        if cfg.type_dropout > 0:
            a_cat, b_cat = category_tables(cfg.b_vocab)
            self.register_buffer("a_cat", a_cat, persistent=False)
            self.register_buffer("b_cat", b_cat, persistent=False)
            self.cat = nn.Embedding(len(CATEGORIES), d)
        self.cfg = cfg
        # Segment-specific numeric projections: the numeric columns mean
        # different things for nodes, selection, goal features, ...
        self.num_w = nn.Parameter(torch.randn(N_SEGMENTS, NUM_DIM, d) * NUM_DIM ** -0.5)
        self.num_b = nn.Parameter(torch.zeros(N_SEGMENTS, d))
        self.in_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.enc_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, batch: dict[str, torch.Tensor], attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        seg = batch["seg"]
        per_seg = torch.einsum("blk,skd->blsd", batch["num"], self.num_w)  # all segment projections
        idx = seg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, per_seg.shape[-1])
        num = per_seg.gather(2, idx).squeeze(2) + self.num_b[seg]
        pos = batch["pos"]
        if self.cfg.pos_mode == "rand":
            pos = randomize_index(pos, self.cfg.pos_table, MAX_POS, self.training, eval_mode=self.cfg.index_eval)
        a_ids, b_ids = batch["a"], batch["b"]
        extra = 0
        if self.cfg.type_dropout > 0:
            is_sel = seg.eq(SEG_SEL)
            extra = self.cat(torch.where(is_sel, self.b_cat[b_ids], self.a_cat[a_ids]))
            if self.training:
                drop = torch.rand(a_ids.shape, device=a_ids.device) < self.cfg.type_dropout
                typed = seg.eq(SEG_NODE) | seg.eq(SEG_RECENT)
                offsets = torch.tensor(A_OFFSETS, device=a_ids.device)[seg.long()]
                a_ids = torch.where(drop & typed, offsets, a_ids)  # segment's <unk> id
                b_ids = torch.where(drop & is_sel, torch.zeros_like(b_ids), b_ids)
        x = self.a(a_ids) + self.b(b_ids) + self.seg(seg) + self.pos(pos) + num + extra
        if self.ord is not None:
            ordinal = randomize_index(batch["ord"], self.cfg.ord_table, MAX_ORD, self.training, keep_zero=True,
                                      eval_mode=self.cfg.index_eval)
            x = x + self.ord(ordinal)
        if attn_mask is not None:  # already folds in padding
            x = self.encoder(self.in_norm(x), mask=attn_mask)
        else:
            x = self.encoder(self.in_norm(x), src_key_padding_mask=~batch["token_mask"])
        return self.out_norm(x)


class ActionEncoder(nn.Module):
    def __init__(self, cfg: S1Config) -> None:
        super().__init__()
        d = cfg.width
        self.id = nn.Embedding(len(ACTION_IDS), d)
        self.cat = nn.Embedding(len(CATEGORIES), d)
        self.scope = nn.Embedding(len(SCOPES), d)
        self.words = nn.Embedding(len(UI_WORD_VOCAB if cfg.ui else WORD_VOCAB), d, padding_idx=0)
        self.vec = nn.Linear(ACT_VEC_DIM, d)
        if cfg.ui:
            self.role = nn.Embedding(len(UI_ROLES), d)
            self.ui_vec = nn.Linear(UI_VEC_DIM, d)
        self.ui = cfg.ui
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.ff), nn.GELU(), nn.Linear(cfg.ff, d))
        self.id_dropout = cfg.id_dropout

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ids = batch["act_id"]
        if self.training and self.id_dropout > 0:
            ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < self.id_dropout, 0)
        w = batch["act_words"]
        wmask = w.ne(0).unsqueeze(-1).float()
        words = (self.words(w) * wmask).sum(2) / wmask.sum(2).clamp_min(1)
        x = self.id(ids) + self.cat(batch["act_cat"]) + self.scope(batch["act_scope"]) + words + self.vec(batch["act_vec"])
        if self.ui:
            x = x + self.role(batch["act_role"]) + self.ui_vec(batch["act_ui"])
        return x + self.mlp(x)


class S1Model(nn.Module):
    def __init__(self, cfg: S1Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or S1Config()
        d = cfg.width
        self.state_encoder = StateEncoder(cfg)
        self.action_encoder = ActionEncoder(cfg)
        layer = nn.TransformerDecoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, cfg.dec_layers)
        self.score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        if cfg.progress_head:
            # Pointer over goal-intent tokens plus an END slot: "which intent am
            # I on / am I done". The pointed-to intent conditions every action.
            self.ptr_q = nn.Linear(d, d)
            self.ptr_k = nn.Linear(d, d)
            self.ptr_end = nn.Linear(d, 1)
            self.end_emb = nn.Parameter(torch.zeros(d))
            self.ptr_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        if cfg.modular:
            self.intent_score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.intent_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def modular_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B*heads, L, L) additive mask: state queries see state keys; goal
        queries see state keys and themselves; padding is excluded."""
        seg, tm = batch["seg"], batch["token_mask"]
        is_goal = seg.eq(SEG_GOAL)
        B, L = seg.shape
        eye = torch.eye(L, dtype=torch.bool, device=seg.device)[None]
        allowed = (~is_goal & tm)[:, None, :].expand(B, L, L) | (eye & (is_goal | ~tm)[:, :, None])
        mask = torch.zeros(B, L, L, device=seg.device).masked_fill(~allowed, float("-inf"))
        return mask.repeat_interleave(self.cfg.heads, 0)

    def modular_forward(self, batch: dict[str, torch.Tensor]):
        context = self.state_encoder(batch, attn_mask=self.modular_mask(batch))
        goal_mask = batch["seg"].eq(SEG_GOAL) & batch["token_mask"]
        ptr_logits = self.intent_score(context).squeeze(-1).float()
        if self.cfg.pointer == "done":
            # ptr_logits = per-intent "done" logits; active = first not-done intent.
            L = context.shape[1]
            order = torch.arange(L, 0, -1, device=context.device)[None]  # earlier slots score higher
            not_done = (ptr_logits < 0) & goal_mask
            last_goal = (goal_mask.long() * torch.arange(L, device=context.device)[None]).argmax(-1)  # END sentinel
            chosen = torch.where(not_done.any(-1), not_done.long().mul(order).argmax(-1), last_goal)
            ptr_logits = ptr_logits.masked_fill(~goal_mask, 0.0)
        else:
            ptr_logits = ptr_logits.masked_fill(~goal_mask, torch.finfo(ptr_logits.dtype).min)
            chosen = ptr_logits.argmax(-1)
        if self.training:  # teacher forcing on the gold active intent where labeled
            gold = batch["progress"]
            chosen = torch.where(gold >= 0, gold, chosen)
        if "force_intent" in batch:  # testing/analysis: override the active intent (token slot)
            chosen = batch["force_intent"]
        onehot = torch.nn.functional.one_hot(chosen, context.shape[1]).bool()
        intent = context[torch.arange(context.shape[0], device=context.device), chosen]
        options = self.action_encoder(batch) + self.intent_proj(intent)[:, None, :]
        memory_mask = (~batch["seg"].eq(SEG_GOAL) & batch["token_mask"]) | onehot
        amask = batch["action_mask"]
        h = self.decoder(options, context, tgt_key_padding_mask=~amask, memory_key_padding_mask=~memory_mask)
        logits = self.score(h).squeeze(-1).float()
        logits = logits.masked_fill(~amask, torch.finfo(logits.dtype).min)
        value = self.value_head(context[:, 0]).squeeze(-1)
        return logits, value, ptr_logits

    def progress_pointer(self, context: torch.Tensor, batch: dict[str, torch.Tensor]):
        goal_mask = batch["seg"].eq(SEG_GOAL) & batch["token_mask"]
        q = self.ptr_q(context[:, 0])
        scores = torch.einsum("bd,bld->bl", q, self.ptr_k(context)) / context.shape[-1] ** 0.5
        scores = scores.masked_fill(~goal_mask, torch.finfo(scores.dtype).min)
        ptr_logits = torch.cat([scores, self.ptr_end(context[:, 0])], dim=-1)  # (B, L+1); column L = END
        p = ptr_logits.softmax(-1)
        pointed = torch.einsum("bl,bld->bd", p[:, :-1], context) + p[:, -1:] * self.end_emb
        return ptr_logits, self.ptr_proj(pointed)

    def aux_loss(self, aux: torch.Tensor | None, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Progress supervision for the pointer / done heads (None if n/a)."""
        progress = batch["progress"]
        labeled = progress >= 0
        if aux is None or not bool(labeled.any()):
            return None
        if self.cfg.modular and self.cfg.pointer == "done":
            is_goal = batch["seg"].eq(SEG_GOAL) & batch["token_mask"]
            rank = is_goal.long().cumsum(-1) - 1  # goal index of each goal slot
            active_rank = rank.gather(1, progress.clamp(min=0)[:, None])  # goal index of the active intent
            target = (rank < active_rank).float()
            weight = (is_goal & labeled[:, None]).float()
            bce = nn.functional.binary_cross_entropy_with_logits(aux, target, reduction="none")
            return (bce * weight).sum() / weight.sum().clamp_min(1)
        return nn.functional.cross_entropy(aux[labeled], progress[labeled])

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False):
        """Returns (logits (B, N) with -inf-like fill on padded/invalid slots,
        value (B,)) and, with return_aux, the progress-pointer logits (B, L+1)
        (None without a progress head)."""
        if self.cfg.modular:
            logits, value, ptr_logits = self.modular_forward(batch)
            return (logits, value, ptr_logits) if return_aux else (logits, value)
        context = self.state_encoder(batch)
        options = self.action_encoder(batch)
        ptr_logits = None
        if self.cfg.progress_head:
            ptr_logits, pointed = self.progress_pointer(context, batch)
            options = options + pointed[:, None, :]
        amask = batch["action_mask"]
        h = self.decoder(options, context, tgt_key_padding_mask=~amask, memory_key_padding_mask=~batch["token_mask"])
        logits = self.score(h).squeeze(-1).float()
        logits = logits.masked_fill(~amask, torch.finfo(logits.dtype).min)
        value = self.value_head(context[:, 0]).squeeze(-1)
        if return_aux:
            return logits, value, ptr_logits
        return logits, value


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: str | Path, model: S1Model, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "metadata": metadata or {}}, path)
    path.with_suffix(".json").write_text(json.dumps({"config": asdict(model.cfg), "metadata": metadata or {}}, indent=2))
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> S1Model:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = S1Model(S1Config(**blob["config"]))
    state = blob["state_dict"]
    key = "state_encoder.pos.weight"
    if key in state and state[key].shape[0] < MAX_POS:  # checkpoints from before the 128-slot table
        state[key] = torch.cat([state[key], torch.zeros(MAX_POS - state[key].shape[0], state[key].shape[1])])
    model.load_state_dict(state)
    return model.to(device).eval()


def save_pretrained(directory: str | Path, model: S1Model, metadata: dict | None = None) -> Path:
    """Export as `model.safetensors` + `config.json` (Hugging Face layout)."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, directory / "model.safetensors", metadata={"format": "pt"})
    (directory / "config.json").write_text(json.dumps(
        {"architecture": "S1Model", "config": asdict(model.cfg), "metadata": metadata or {}}, indent=2))
    return directory


def from_pretrained(name_or_path: str | Path, device: str | torch.device = "cpu") -> S1Model:
    """Load from a local directory or a Hugging Face Hub repo id."""
    from safetensors.torch import load_file

    path = Path(name_or_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(str(name_or_path), allow_patterns=["model.safetensors", "config.json"]))
    cfg = json.loads((path / "config.json").read_text())["config"]
    model = S1Model(S1Config(**cfg))
    model.load_state_dict(load_file(path / "model.safetensors"))
    return model.to(device).eval()

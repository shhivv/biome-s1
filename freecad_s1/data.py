"""Load datagen shards into featurized examples (cached per shard)."""

from __future__ import annotations

import gzip
import json
import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .actions import CATALOGUE
from .ui import spec as UIS
from .model.featurize import Example, make_example
from .screen_state import OkMemory
from .schema import Goal, State

CACHE_VERSION = 5


def label_category(acceptable: list[str]) -> str:
    a = acceptable[0]
    role = UIS.role(a)
    if role in ("set", "opt", "toggle"):
        return "ui_field"
    if role == "click":
        return "ui_ok" if a == UIS.OK else "ui_cancel"
    a = UIS.underlying(a) or a
    if a == "Std_Undo":
        return "undo"
    if a.startswith("Select:"):
        return "select"
    return CATALOGUE[a].category if a in CATALOGUE else "other"


@dataclass
class Dataset:
    examples: list[Example] = field(default_factory=list)
    ep: list[str] = field(default_factory=list)
    level: list[int] = field(default_factory=list)
    noise: list[float] = field(default_factory=list)
    category: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def add(self, ex: Example, ep: str, level: int, noise: float, acceptable: list[str]) -> None:
        self.examples.append(ex)
        self.ep.append(ep)
        self.level.append(level)
        self.noise.append(noise)
        self.category.append(label_category(acceptable))

    def extend(self, other: "Dataset") -> None:
        for name in ("examples", "ep", "level", "noise", "category"):
            getattr(self, name).extend(getattr(other, name))

    def subset(self, idx) -> "Dataset":
        out = Dataset()
        for name in ("examples", "ep", "level", "noise", "category"):
            src = getattr(self, name)
            setattr(out, name, [src[i] for i in idx])
        return out

    def split(self, every: int = 20) -> tuple["Dataset", "Dataset"]:
        """Deterministic split by episode id (no episode straddles both sides)."""
        import zlib

        val = np.array([zlib.crc32(e.encode()) % every == 0 for e in self.ep])
        return self.subset(np.flatnonzero(~val)), self.subset(np.flatnonzero(val))


def read_records(path: str):
    """Records of one gzip JSONL shard. A shard whose writer was killed
    (the UI data generator's watchdog uses SIGKILL) ends mid-stream: keep
    everything up to the last complete line instead of failing the load."""
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"warning: {path}: truncated record, stopping there")
                    return
    except (EOFError, gzip.BadGzipFile, zlib.error) as exc:
        print(f"warning: {path}: truncated shard ({type(exc).__name__}), using the records before the cut")


def _load_shard(args: tuple[str, dict]) -> Dataset:
    path, opts = args
    tag = "".join(f".{k}" for k, v in sorted(opts.items()) if v)
    cache = Path(path + f".v{CACHE_VERSION}{tag}.pkl")
    if cache.exists() and cache.stat().st_mtime >= Path(path).stat().st_mtime:
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    ds = Dataset()
    goals: dict[str, Goal] = {}
    memory: dict[str, OkMemory] = {}  # per episode (records of several episodes interleave)
    for rec in read_records(path):
        if "goal" in rec:
            goals[rec["ep"]] = Goal.from_json(rec["goal"])
            continue
        state = State.from_json(rec["state"])
        if opts.get("ui_param") or opts.get("ui_ok"):
            UIS.annotate_targets(state.ui, goals[rec["ep"]], rec.get("progress"))
        if opts.get("ui_ok"):
            mem = memory.setdefault(rec["ep"], OkMemory())
            mem.after(state)  # the action of this episode's previous record led here
            mem.before(state)
        if not set(rec["acceptable"]) & set(rec["actions"]):
            continue  # no correct option offered (rare GUI corner case): no learning signal
        ex = make_example(state, goals[rec["ep"]], rec["actions"], rec["acceptable"], rec.get("progress"), **opts)
        ds.add(ex, rec["ep"], rec["level"], rec["noise"], rec["acceptable"])
    with open(cache, "wb") as fh:
        pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ds


def load_dataset(directory: str | Path, workers: int = 8, **opts) -> Dataset:
    """`opts` are featurization options (S1Config.feature_opts())."""
    paths = sorted(str(p) for p in Path(directory).glob("*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no *.jsonl.gz shards in {directory}")
    out = Dataset()
    with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
        for ds in pool.map(_load_shard, [(p, opts) for p in paths]):
            out.extend(ds)
    return out

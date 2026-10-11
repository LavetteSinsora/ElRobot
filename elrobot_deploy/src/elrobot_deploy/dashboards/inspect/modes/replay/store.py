"""What one policy replay looks like on disk, and how to list them.

P9's grammar, deliberately: flat `np.fromfile`-able streams plus a
`manifest.json` that carries each stream's own `descr`, so
`elrobot_deploy.rollout_record_utils.reader.load` reads a replay with no code from this package. A
replay is a rollout record's peer, and every tool that understands one
understands the other.

    <record>/policy_replays/<id>/
        chunks.bin     the replayed chunks, in the LIVE record's own chunk dtype
        anchors.bin    which rows of the record made each observation
        manifest.json  written LAST — its presence is what "finished" means
    <record>/policy_replays/<id>.partial/   while it runs (or if it died)
    <record>/policy_replays/_jobs/          the dashboard's job logs

`chunks.bin` matching the live dtype byte for byte is what lets the dashboard
run one overlay builder over a replay and over the recording's own chunks. The
fields that describe WHEN (`t_trigger`, `t_anchor`, `t_obs_received`, ...) are
copied from the source chunk, because they describe the live replan this row
stands in for; the replay's own timing is in the manifest.

One difference from live, and it is the useful one: the live recorder stores
only the `P` rows the robot executed (sync), a replay stores every row the
policy returned. `n_actions` still says what would have run, so the rest is
visible as the tail the policy predicted and never got to use.

**Every replay is its own directory**, even of the same checkpoint with the
same settings: the spread between repeats is the sampler's own noise, the
floor every other difference has to be read against.

Written by the replay producer (`run.py`), listed and read by the replay mode.
For a user: `docs/deploying_a_policy/rollout_inspection.md`, "policy checkpoint (`replay`)".
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "elrobot_deploy.policy_replay.v1"

#: Where a record's replays live, relative to the record.
DIR = "policy_replays"

#: Written while a replay is in flight.
PARTIAL_SUFFIX = ".partial"

#: The recording's own chunks, listed as one more entry.
LIVE_ID = "live"

#: Why a replay failed, in the directory it leaves behind (`Writer.fail`).
ERROR = "error.txt"


def replays_dir(record_dir: Path) -> Path:
    return Path(record_dir) / DIR


def new_id() -> str:
    """`20260926-101512-3fa2`: when it started, plus four hex so two replays
    started in one second cannot share a directory."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def anchor_dtype(cameras: list[str]) -> np.dtype:
    """Which rows of the source record went into each observation."""
    return np.dtype([
        ("chunk_id", "<u8"),
        ("state_row", "<i8"),
        ("frame_rows", "<i8", (max(1, len(cameras)),)),
        ("primary_exact", "u1"),
        ("ambiguous", "u1"),
        ("state_gap_s", "<f8"),
        ("t_infer", "<f8"),                 # wall seconds of this row's infer()
    ])


def _descr(dt: np.dtype) -> list:
    """A dtype as the manifest records it: exactly `rollout_record_utils.reader._dtype`'s input."""
    out = []
    for name in dt.names:
        f = dt.fields[name][0]
        if f.subdtype is None:
            out.append([name, f.str])
        else:
            base, shape = f.subdtype
            out.append([name, base.str, list(shape)])
    return out


class Writer:
    """Accumulates one replay in memory, then writes it in one pass, manifest
    last. In memory because a replay is small (a few MB of poses) and because a
    streaming writer that died would leave a directory that looks complete."""

    def __init__(self, out_dir: Path, *, chunk_dtype: np.dtype, cameras: list[str]):
        self.out_dir = Path(out_dir)
        self.partial = self.out_dir.with_name(self.out_dir.name + PARTIAL_SUFFIX)
        self.chunk_dtype = chunk_dtype
        self.cameras = list(cameras)
        self.anchor_dtype = anchor_dtype(self.cameras)
        self._chunks: list[np.ndarray] = []
        self._anchors: list[np.ndarray] = []

    def begin(self) -> None:
        """Make the `.partial` directory, so the dashboard lists it as running."""
        self.partial.mkdir(parents=True, exist_ok=True)

    def add(self, row: np.ndarray, spec, t_infer: float) -> None:
        self._chunks.append(np.asarray(row, dtype=self.chunk_dtype).reshape(1))
        a = np.zeros(1, dtype=self.anchor_dtype)
        a["chunk_id"] = int(spec.chunk_id)
        a["state_row"] = int(spec.state_row)
        a["frame_rows"][0, :len(self.cameras)] = [int(spec.frames.get(c, -1))
                                                  for c in self.cameras]
        a["primary_exact"] = int(bool(spec.primary_exact))
        a["ambiguous"] = int(bool(spec.ambiguous))
        a["state_gap_s"] = float(spec.state_gap_s)
        a["t_infer"] = float(t_infer)
        self._anchors.append(a)

    @property
    def n(self) -> int:
        return len(self._chunks)

    def rows(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=self.chunk_dtype)
        return np.concatenate(self._chunks)

    def fail(self, exc: BaseException) -> None:
        """Leave a failed replay as a directory WITHOUT a manifest, carrying
        the reason, so the list says "broken: why" instead of "running"
        forever."""
        try:
            self.partial.mkdir(parents=True, exist_ok=True)
            (self.partial / ERROR).write_text(f"{type(exc).__name__}: {exc}\n")
            os.replace(self.partial, self.out_dir)
        except OSError:
            pass

    def close(self, manifest: dict) -> Path:
        """Write the streams, then the manifest, then make the name real."""
        self.partial.mkdir(parents=True, exist_ok=True)
        streams = {}
        for name, rows, dt in (("chunks", self._chunks, self.chunk_dtype),
                               ("anchors", self._anchors, self.anchor_dtype)):
            arr = np.concatenate(rows) if rows else np.zeros(0, dtype=dt)
            arr.tofile(self.partial / f"{name}.bin")
            streams[name] = {"file": f"{name}.bin", "descr": _descr(dt),
                             "records": len(arr)}
        full = dict(manifest, schema=SCHEMA, streams=streams)
        (self.partial / "manifest.json").write_text(
            json.dumps(full, indent=2, default=_jsonable))
        os.replace(self.partial, self.out_dir)
        return self.out_dir


def _jsonable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-serialisable")


def read_manifest(replay_dir: Path) -> dict | None:
    """The manifest, or None when the replay is unfinished or not one."""
    path = Path(replay_dir) / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if data.get("schema") == SCHEMA else None


def load_chunks(replay_dir: Path) -> np.ndarray:
    """The `chunks` stream of one replay — the live record's own dtype."""
    from elrobot_deploy.rollout_record_utils import reader as record_read

    return record_read.load(Path(replay_dir))["chunks"]


@dataclass
class ReplayInfo:
    """One row of the replay list."""

    id: str
    kind: str                               # "live" | "replay"
    label: str
    state: str = "ready"                    # ready | running | broken
    dir: str = ""
    created: str = ""
    entry: str = ""
    policy_class: str = ""
    checkpoint: str = ""
    prompt: str = ""
    settings: dict = field(default_factory=dict)
    seed: int | None = None
    n_chunks: int = 0
    action_mode: str = ""
    summary: dict = field(default_factory=dict)
    note: str = ""

    def as_json(self) -> dict:
        return dict(self.__dict__)


def checkpoint_label(checkpoint: str) -> str:
    """`/runs/my_run/29999` -> `my_run/29999`; "" -> "no checkpoint"."""
    if not checkpoint:
        return "no checkpoint"
    parts = [p for p in Path(checkpoint).parts if p not in ("/", "")]
    return "/".join(parts[-2:])


def label_of(man: dict, recorded: dict | None = None) -> str:
    """What a replay's switch says: the checkpoint (if any) and the sample steps, plus
    the prompt and the policy when they differ from the rollout's — so two
    replays of one checkpoint at different settings are told apart at a glance.
    The rollout's own values come from the replay's manifest (`recorded_*`),
    or from `recorded` for a replay written without them."""
    recorded = recorded or {}
    bits = [checkpoint_label(str(man.get("checkpoint") or ""))]
    settings = man.get("settings") or {}
    if "sample_steps" in settings:
        bits.append(f"sample steps {settings['sample_steps']}")
    prompt = man.get("prompt")
    rec_prompt = man.get("recorded_prompt", recorded.get("prompt"))
    if prompt is not None and prompt != rec_prompt:
        short = prompt if len(prompt) <= 28 else prompt[:27] + "…"
        bits.append(f"“{short}”")
    rec_entry = man.get("recorded_entry", recorded.get("entry"))
    if man.get("entry") and rec_entry and man["entry"] != rec_entry:
        bits.append(str(man.get("class") or man["entry"]).rsplit(".", 1)[-1])
    return " · ".join(bits)


def list_replays(record_dir: Path, recorded: dict | None = None) -> list[ReplayInfo]:
    """Every replay directory of a record, oldest first.

    A `.partial` directory is a replay in flight (or one that died): listed as
    `running` rather than hidden, so pressing Replay makes something appear at
    once and a crash leaves evidence instead of silence.
    """
    base = replays_dir(record_dir)
    out: list[ReplayInfo] = []
    if not base.is_dir():
        return out
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        name = child.name
        if name.endswith(PARTIAL_SUFFIX):
            real = name[:-len(PARTIAL_SUFFIX)]
            out.append(ReplayInfo(real, "replay", f"{real} (running)",
                                  state="running", dir=str(child),
                                  note="replaying…"))
            continue
        man = read_manifest(child)
        if man is None:
            try:
                why = (child / ERROR).read_text().strip()
            except OSError:
                why = "no manifest: the replay did not finish"
            out.append(ReplayInfo(name, "replay", f"{name} (failed)",
                                  state="broken", dir=str(child), note=why))
            continue
        out.append(ReplayInfo(
            name, "replay", label_of(man, recorded), dir=str(child),
            created=str(man.get("created", "")), entry=str(man.get("entry", "")),
            policy_class=str(man.get("class", "")),
            checkpoint=str(man.get("checkpoint", "")),
            prompt=str(man.get("prompt", "")),
            settings=dict(man.get("settings") or {}), seed=man.get("seed"),
            n_chunks=int(man.get("n_chunks", 0)),
            action_mode=str(man.get("action_mode", "")),
            summary=dict(man.get("summary") or {})))
    _disambiguate(out)
    return out


def _disambiguate(rows: list[ReplayInfo]) -> None:
    """Where a label repeats — repeats of one checkpoint at one setting, which
    is how the sampler's spread becomes visible — append the time it ran."""
    seen: dict[str, list[ReplayInfo]] = {}
    for row in rows:
        seen.setdefault(row.label, []).append(row)
    for group in seen.values():
        if len(group) < 2:
            continue
        times = [row.id[9:15] for row in group]
        for row, t in zip(group, times):
            stamp = f"{t[:2]}:{t[2:4]}:{t[4:6]}" if t.isdigit() and len(t) == 6 else ""
            # Two started within one second share the time: the id's own
            # suffix then tells them apart.
            if not stamp or times.count(t) > 1:
                stamp = (stamp + " " if stamp else "") + row.id[-4:]
            row.label += f" · {stamp}"


__all__ = ["DIR", "ERROR", "LIVE_ID", "PARTIAL_SUFFIX", "SCHEMA", "ReplayInfo", "Writer",
           "checkpoint_label", "label_of", "list_replays", "load_chunks",
           "new_id", "read_manifest", "replays_dir"]

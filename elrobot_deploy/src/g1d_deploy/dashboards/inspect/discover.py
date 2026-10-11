"""What rollouts exist on disk, for the recording dropdown.

    g1d_deploy inspect <root>          every record under <root>
    g1d_deploy inspect <record dir>    just that one

**A record is a directory holding a `manifest.json` that P9 wrote.** Nothing
else is assumed: no naming convention, no experiment tree, no git checkout to
walk. `<root>` is searched recursively, and a directory that is a record is not
searched further (a record's own sidecars — `policy_replays/`, IK bench files —
live inside it and are never records themselves).

"Written by P9" is decided by the manifest's own content: P9's manifests carry a
`streams` table with a `state` stream, because a rollout without state samples
is not a rollout. A replay's result directory has a manifest too, of a
different schema, and is not offered.

**The dropdown groups by the record's path relative to `<root>`**, minus its own
name. So `pick_fruit/rollout_record/20260926-101010` groups under
`pick_fruit/rollout_record`, a flat directory of stamps groups under
nothing, and any layout anybody keeps works without being described here.

**The manifest header is read here, and only the header.** Which policy (and
checkpoint) and which settings a rollout ran with is the first thing anyone asks of it, so
`specifics()` lifts the run-identifying fields out of `manifest.json` and they
travel with the listing. No stream is opened and nothing is decoded: reading a
rollout is still `loader`'s job.

**An unreadable manifest is listed, not hidden.** Dropping it would present
"this rollout cannot be opened" as "this rollout does not exist", the more
confusing of the two. It comes back with `loadable = False` and the parser's
reason.

Run by the inspection server at start and on refresh. For a user: `docs/deploying_a_policy/rollout_inspection.md`,
"What it opens".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

#: Directory names never descended into. Hidden directories (`.git`, a
#: `.partial` in flight) and caches can be huge and hold nothing P9 wrote.
_PRUNE = {"__pycache__", "node_modules"}

MANIFEST = "manifest.json"


def is_record_manifest(manifest: dict) -> bool:
    """Did P9 write this? A `streams` table with a `state` stream says so."""
    streams = manifest.get("streams") if isinstance(manifest, dict) else None
    return isinstance(streams, dict) and "state" in streams


def _checkpoint_id(checkpoint: str) -> tuple[str, str]:
    """A checkpoint path -> `(run, step)`, both for display only.

        /runs/my_run/29999          -> ("my_run", "29999")
        /runs/my_run/best/4500      -> ("my_run/best", "4500")
        /x/checkpoints/my_weights   -> ("my_weights", "")

    The step is the trailing directory when it is a number (the one layout
    g1d_deploy knows, `policy_contract/checkpoints.py`); the run is the directory above it,
    with one more component when that directory is a non-numeric alias nested
    inside a run (`best/`). Nothing here requires the path to exist: it is a
    path on the machine that ran the rollout, parsed as POSIX for that reason.
    """
    if not checkpoint:
        return "", ""
    parts = [p for p in PurePosixPath(checkpoint).parts if p not in ("/", "")]
    if not parts:
        return "", ""
    step = parts[-1] if parts[-1].isdigit() else ""
    head = parts[:-1] if step else parts
    if not head:
        return "", step
    if not step:
        return head[-1], ""
    # `<run>/best/<step>`: a short lowercase alias directory (`best`, `ema`)
    # between run and step. Two components keep `my_run/best/4500` from
    # reading as `my_run/4500`, which is a different network.
    alias = head[-1]
    if len(head) >= 2 and alias.isalpha() and alias.islower() and len(alias) <= 6:
        return "/".join(head[-2:]), step
    return head[-1], step


def _sha(block: dict | None) -> tuple[str, bool]:
    block = block or {}
    return (str(block.get("sha") or "")[:12], bool(block.get("dirty")))


def specifics(manifest: dict) -> dict:
    """The facts that say which RUN a recording is, from its manifest.

    Read by the listing, so the record chip can answer before anything is
    loaded, and by `loader.Rollout.summary()`, so the two cannot disagree.

    The rollout's own settings come from `rollouts[0].active_config` (what P0
    activated at that START), falling back to `deploy_config` for records
    written before settings could change between rollouts. `P` and `H` are
    asked for under their field names — `replan_period_ticks`,
    `action_horizon` — because `P`/`H` are properties, not fields; the short
    names are accepted second for records that dumped them.
    """
    cfg = manifest.get("deploy_config") or {}
    rollouts = manifest.get("rollouts") or [{}]
    roll = rollouts[0] if isinstance(rollouts[0], dict) else {}
    active = roll.get("active_config") or {}
    policy = manifest.get("policy") or {}
    robot = manifest.get("robot") or {}
    infra = manifest.get("infra") or {}
    checkpoint = str((manifest.get("checkpoint") or {}).get("path")
                     or cfg.get("checkpoint") or "")
    run, step = _checkpoint_id(checkpoint)
    code_sha, code_dirty = _sha(manifest.get("code") or infra.get("git"))
    infra_sha, infra_dirty = _sha(infra.get("git"))
    policy_sha, policy_dirty = _sha(policy.get("git"))
    code = manifest.get("code") or infra.get("git") or {}

    def pick(active_key, cfg_key, short=None):
        if active_key in active:
            return active[active_key]
        if cfg_key in cfg:
            return cfg[cfg_key]
        return cfg.get(short) if short else None

    return {
        "created": manifest.get("created"),
        "started": roll.get("started"),
        "prompt": roll.get("prompt", cfg.get("prompt")),
        "rollout_index": roll.get("index"),
        "generation": roll.get("generation"),
        "active_config": dict(active),
        "checkpoint": checkpoint,
        "checkpoint_run": run,
        "checkpoint_step": step,
        "sync": pick("sync", "sync"),
        "P": pick("replan_period", "replan_period_ticks", "P"),
        "H": cfg.get("action_horizon", cfg.get("H")),
        "control_hz": pick("control_hz", "control_hz"),
        "sample_steps": pick("sample_steps", "sample_steps"),
        "servo": active.get("servo") or cfg.get("servo"),
        "blend": cfg.get("blend"),
        # Which IK flew: the solver's name (`runtime/ik` `SOLVERS`). Records
        # made before `ik` named the solver carried the mode there and the
        # solver, if any, in `ik_solver`.
        "ik": cfg.get("ik_solver") or pick("ik", "ik"),
        # "eef" | "joint" (the policy class's `action_mode`); a joint rollout
        # used no IK, whatever `ik` the config carried.
        "action_mode": policy.get("action_mode") or "",
        "warm": manifest.get("warm"),
        "policy_entry": policy.get("entry") or cfg.get("policy_entry") or "",
        "policy_class": policy.get("class") or "",
        "anchor_camera": policy.get("anchor_camera") or cfg.get("primary_camera"),
        "uses_cameras": list(policy.get("uses_cameras")
                             or cfg.get("observed_cameras") or ()),
        "robot": robot.get("name") or cfg.get("robot") or "",
        "robot_files": [f.get("path", "") for f in robot.get("files") or ()
                        if isinstance(f, dict)],
        "end_effector": robot.get("end_effector"),
        "infra_version": infra.get("version"),
        "sha": code_sha,
        "dirty": code_dirty,
        "dirty_files": int(code.get("dirty_file_count")
                           or len(code.get("dirty_files") or ())),
        "infra_sha": infra_sha, "infra_dirty": infra_dirty,
        "policy_sha": policy_sha, "policy_dirty": policy_dirty,
        # A replay's provenance (an executor replay names the rollout it re-ran
        # in `source_name`). Empty for an ordinary rollout.
        "labels": dict(cfg.get("labels") or {}),
    }


@dataclass(frozen=True)
class Recording:
    """One record directory."""

    #: The record's path relative to the root, POSIX — the identifier the
    #: browser sends back. `"."` when the root IS the record.
    key: str
    #: The relative path of the directory holding it: the dropdown group.
    group: str
    name: str
    path: Path
    loadable: bool
    #: Why not, when `loadable` is False. Empty otherwise.
    reason: str = ""
    #: Bytes on disk, for the UI to show what a Load will cost.
    size_bytes: int = 0
    #: `specifics()` of this recording's manifest. Empty when unreadable.
    details: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """`20260820-163255` -> `2026-08-20 16:32:55`, which is how a human
        holds a rollout in mind; any other name is shown as it is."""
        n = self.name
        if len(n) >= 15 and n[8] == "-" and n[:8].isdigit() and n[9:15].isdigit():
            return (f"{n[:4]}-{n[4:6]}-{n[6:8]} {n[9:11]}:{n[11:13]}:{n[13:15]}"
                    + n[15:])
        return n


def _read(path: Path, key: str, group: str) -> Recording | None:
    """One candidate directory -> a `Recording`, or None if P9 did not write it."""
    size = 0
    try:
        size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    except OSError:
        pass
    try:
        manifest = json.loads((path / MANIFEST).read_text())
    except (OSError, ValueError) as exc:
        # `loader` rebuilds every dtype from the manifest, so one that will not
        # parse is a record nothing can open. Said here, with the parser's own
        # words, rather than at Load.
        return Recording(key, group, path.name, path, False,
                         f"manifest.json could not be read "
                         f"({type(exc).__name__}: {exc}). Nothing records what "
                         f"this rollout's streams contain.", size)
    if not is_record_manifest(manifest):
        return None
    return Recording(key, group, path.name, path, True, "", size,
                     specifics(manifest))


def scan(root: Path | str) -> list[Recording]:
    """Every record under `root`, grouped by relative parent path, the newest
    name first within a group. `root` itself may be a record."""
    root = Path(root).expanduser().resolve()
    if (root / MANIFEST).is_file():
        rec = _read(root, ".", "")
        return [rec] if rec is not None else []
    out: list[Recording] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        here = Path(dirpath)
        if MANIFEST in filenames and here != root:
            rel = here.relative_to(root).as_posix()
            parent = here.parent.relative_to(root).as_posix()
            rec = _read(here, rel, "" if parent == "." else parent)
            if rec is not None:
                out.append(rec)
                dirnames[:] = []            # a record never holds records
                continue
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d not in _PRUNE)
    out.sort(key=lambda r: r.name, reverse=True)
    out.sort(key=lambda r: r.group)         # stable: groups A-Z, newest first
    return out


def resolve(key: str, root: Path | str) -> Path:
    """A dropdown selection -> the record directory, or `KeyError`.

    Resolved by walking `scan()` rather than by joining `key` onto `root`, so a
    crafted or stale selection cannot address a directory outside the tree.
    """
    for rec in scan(root):
        if rec.key == key:
            if not rec.loadable:
                raise KeyError(f"{key}: {rec.reason}")
            return rec.path
    raise KeyError(f"no such recording: {key}")


def as_json(records: list[Recording], root: Path | str) -> dict:
    """The listing, shaped for the browser: groups in order, records inside."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.group, []).append({
            "key": r.key, "name": r.name, "label": r.label,
            "loadable": r.loadable, "reason": r.reason,
            "size_mb": round(r.size_bytes / 1e6, 1),
            # Sent with the listing rather than fetched per selection: it is
            # already in hand, a few hundred bytes, and it means selecting a
            # recording shows its checkpoint with no round trip at all.
            "details": r.details,
            "dir": str(r.path),
        })
    root = Path(root).expanduser().resolve()
    return {"root": str(root), "single": (root / MANIFEST).is_file(),
            "groups": [{"group": g, "records": rs} for g, rs in groups.items()]}


__all__ = ["MANIFEST", "Recording", "as_json", "is_record_manifest", "resolve",
           "scan", "specifics"]

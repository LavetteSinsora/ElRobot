"""What a launch was made of, for every rollout manifest.

`build(resolved)` returns a JSON-able dict that P0 hands to P9 and P9 writes
into each `manifest.json` next to the config dump:

    infra       g1d_deploy's version and git revision
    robot       the robot file chain, VERBATIM, plus what it resolved to
    machine     the machine file's resolved values (not its text: it holds
                the board's SSH password)
    deploy      the deploy file verbatim, the tree after command-line
                overrides, and the command line itself
    policy      the entry, the class, its file, and the git revision of the
                repository the policy lives in (not g1d_deploy's)
    checkpoint  the resolved path ("" for a policy that loads nothing)
    resolution  per field, which layer won (infra / trained / deploy / cli),
                the policy's declared `trained_*` rates, and every warning
                printed

Files are copied as text because a profile or a deploy file can change after
the run, and a record has to say what it ran against, not what the name
means today.

Built by P0 at launch (`runtime/launcher/`), written by P9, read back by
inspection (`dashboards/inspect/discover.py`, `record_robot.py`).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

#: Dirty paths listed before the list is cut; the count stays exact.
_MAX_DIRTY_FILES = 50


def git_revision(path: str | Path) -> dict:
    """The git revision of the repo containing `path`, and whether it is dirty.

    Never raises: a policy outside any repo, or a box with no `git`, still gets
    a record — `{"sha": None, ...}` says why.
    """
    p = Path(path)
    where = p if p.is_dir() else p.parent
    try:
        def git(*args):
            return subprocess.run(("git", "-C", str(where), *args),
                                  capture_output=True, text=True, timeout=10)

        head = git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"sha": None, "error": head.stderr.strip()[:200]}
        top = git("rev-parse", "--show-toplevel").stdout.strip()
        changed = [line[3:] for line in git("status", "--porcelain").stdout.splitlines()]
        return {"sha": head.stdout.strip(), "root": top, "dirty": bool(changed),
                "dirty_files": sorted(changed)[:_MAX_DIRTY_FILES],
                "dirty_file_count": len(changed)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"sha": None, "error": f"{type(exc).__name__}: {exc}"[:200]}


def infra_version() -> str:
    """g1d_deploy's version: the installed distribution's, else pyproject's."""
    try:
        from importlib.metadata import version
        return version("g1d_deploy")
    except Exception:                          # noqa: BLE001 - not installed
        pass
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        from g1d_deploy.config_loading.resolved import tomllib
        return str(tomllib.loads(pyproject.read_text())["project"]["version"])
    except Exception:                          # noqa: BLE001
        return "unknown"


def _spec(spec) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(spec).items()}


def robot_block(name: str, robot) -> dict:
    """The manifest's `robot`: the robot file chain the run used, VERBATIM,
    each file named and marked shipped (one of `robot/profiles/*.toml`) or not, plus
    whether the model is the package's own. Inspection rebuilds the robot
    from this text and this package's assets — no path in it has to exist on
    the reading machine (`dashboards/inspect/record_robot.py`)."""
    from g1d_deploy.config_loading.robot_file import ROBOTS_DIR

    files = []
    for f in robot.files:
        p = Path(f.path)
        shipped = p.parent == ROBOTS_DIR
        files.append({"name": p.stem if shipped else p.name, "shipped": shipped,
                      "path": f.path, "text": f.text})
    urdf = str((robot.raw.get("model") or {}).get("urdf") or "")
    return {"name": name, "shipped": all(f["shipped"] for f in files),
            "extends": [f["name"] for f in files],
            "model": {"urdf": urdf, "shipped": model_is_shipped(urdf)},
            "end_effector": robot.end_effector,
            "image_server": robot.image_server,
            "files": files,
            "cameras": {n: _spec(s) for n, s in robot.cameras.items()}}


def model_is_shipped(urdf: str) -> bool:
    """A robot file's `[model] urdf` names the package's own model: a path
    relative to the package that exists in it."""
    from g1d_deploy.config_loading.resolved import MODEL_ROOT
    p = Path(urdf)
    return bool(urdf) and not p.is_absolute() and (MODEL_ROOT / p).is_file()


def build(resolved, argv: list[str] | None = None) -> dict:
    """The provenance block for one launch (`deploy_file.Resolved`)."""
    r = resolved
    info = r.policy
    policy_file = info.file if info is not None else None
    return {
        "infra": {"name": "g1d_deploy", "version": infra_version(),
                  "git": git_revision(Path(__file__).resolve().parent)},
        "robot": robot_block(r.cfg.robot, r.robot),
        "machine": {"path": r.machine.path, "robot_ip": r.hw.camera_host,
                    "iface": r.machine.iface,
                    "sources": {k: v for k, v in r.machine.sources.items()
                                if "password" not in k},
                    "board_ssh_user": r.hw.board_ssh_user,
                    "cpu_source": r.hw.cpu_source},
        "deploy": {"path": r.deploy_path, "text": r.deploy_text,
                   "resolved_tree": r.tree, "argv": list(argv or [])},
        "policy": {"entry": r.cfg.policy_entry,
                   "class": info.class_path if info is not None else None,
                   "file": policy_file,
                   "python": r.cfg.policy_python or sys.executable,
                   "numpy": info.numpy_version if info is not None else None,
                   "git": git_revision(policy_file) if policy_file else None,
                   "action_mode": r.cfg.policy_action_mode or None,
                   "anchor_camera": r.cfg.primary_camera,
                   "uses_cameras": list(r.cfg.observed_cameras)},
        "checkpoint": {"path": r.cfg.checkpoint},
        "resolution": {"sources": dict(r.sources), "trained": dict(r.trained),
                       "warnings": list(r.warnings)},
    }


__all__ = ["build", "git_revision", "infra_version", "model_is_shipped",
           "robot_block"]

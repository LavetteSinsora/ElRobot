"""The Astra workspace: the folder Codex is opened on, and one folder per episode inside it.

    launch = prepare(cfg)                      # at `elrobot_deploy astra`: episodes/, sockets
    Launcher(cfg, astra=launch, ...).run()
    folder = new_episode(cfg, launch, task)    # the harness, at each `start(task)`

    ElRobot/astra/                  `[astra] workspace`; open it in Codex
      AGENTS.md                     what Astra reads first (Codex loads it by itself): the
                                    robot, frames, units, tools, the folder. Kept by hand: its
                                    numbers (limits, gripper, home) follow the profile.
      .codex/config.toml            the `elrobot` MCP server, for this project only (Codex
                                    reads it once the project is trusted). Kept by hand.
      episodes/<stamp>_<task>/      one per `start`; `episodes/current` -> the newest (not
                                    versioned)

`AstraLaunch` is what every process of the runtime is told (pickled at spawn). The sockets live
in a fixed short directory under /tmp (`SOCKET_DIR`), not in the workspace: a socket path is
capped at 104 bytes on macOS, and the MCP server Codex starts must find the harness without
being told where it is.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from elrobot_deploy.astra.frames import TaskFrame

#: Where the harness listens, always: the MCP server's default.
SOCKET_DIR = "/tmp/elrobot-astra"
HARNESS_SOCKET = "harness.sock"
PLAYER_SOCKET = "player.sock"


@dataclass(frozen=True)
class AstraLaunch:
    workspace: str
    record_base: str
    socket_dir: str = SOCKET_DIR

    @property
    def harness_socket(self) -> str:
        return str(Path(self.socket_dir) / HARNESS_SOCKET)

    @property
    def player_socket(self) -> str:
        return str(Path(self.socket_dir) / PLAYER_SOCKET)

    @property
    def episodes(self) -> Path:
        return Path(self.workspace) / "episodes"


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:n].rstrip("_") or "task"


def home_tcp(cfg) -> tuple[np.ndarray, np.ndarray]:
    """The home pose's TCP in Astra's base frame: (xyz m, rpy deg)."""
    from elrobot_deploy.robot.elrobot_model import ElRobotModel, mat_to_pose
    T = ElRobotModel(cfg.urdf).fk(cfg.poses.home[:7])
    return TaskFrame.from_config(cfg).describe(mat_to_pose(T))


def configured_cameras(cfg) -> list[str]:
    """`[astra] cameras` that this launch has, in their order."""
    have = {c.name for c in cfg.cameras}
    return [c for c in cfg.astra.cameras if c in have]


def anchor_camera(cfg, cams) -> str | None:
    if cfg.astra.anchor_camera in cams:
        return cfg.astra.anchor_camera
    return cams[0] if cams else None


def check_workspace(ws: Path, *, log=print) -> None:
    """`AGENTS.md` and `.codex/config.toml` are kept by hand in the repo (`ElRobot/astra/`); a
    launch only says when one is missing, or when the config names a python that is not
    there (another machine, a moved checkout)."""
    import tomllib
    if not (ws / "AGENTS.md").exists():
        log(f"WARNING {ws / 'AGENTS.md'} is missing: Astra will not know the robot or the tools.")
    path = ws / ".codex" / "config.toml"
    if not path.exists():
        log(f"WARNING {path} is missing: Codex will not see the robot's tools.")
        return
    try:
        command = tomllib.loads(path.read_text())["mcp_servers"]["elrobot"]["command"]
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        log(f"WARNING {path}: no [mcp_servers.elrobot] command ({exc}); Codex will not see the "
            f"robot's tools.")
        return
    if not Path(command).exists():
        log(f"WARNING {path}: the MCP server's python {command} does not exist; Codex will not "
            f"start the robot's tools. This launch runs {sys.executable}.")


def prepare(cfg, *, socket_dir: str = SOCKET_DIR) -> AstraLaunch:
    """At each launch: the places (the workspace's episodes/, the socket directory), and a
    check of the workspace's two hand-kept files."""
    ws = Path(cfg.astra.workspace)
    (ws / "episodes").mkdir(parents=True, exist_ok=True)
    check_workspace(ws)
    Path(socket_dir).mkdir(parents=True, exist_ok=True)
    return AstraLaunch(workspace=str(ws), record_base=str(cfg.record.dir), socket_dir=socket_dir)


def new_episode(cfg, launch: AstraLaunch, task: str, cams, *, stamp: str | None = None) -> Path:
    """One episode's folder (`episode.json`, actions/, frames/), and `episodes/current`."""
    stamp = stamp or time.strftime("%Y-%m-%d_%H%M%S")
    folder = launch.episodes / f"{stamp}_{slug(task)}"
    n = 1
    while folder.exists():
        n += 1
        folder = launch.episodes / f"{stamp}_{slug(task)}-{n}"
    (folder / "actions").mkdir(parents=True)
    (folder / "frames").mkdir()
    xyz, rpy = home_tcp(cfg)
    a = cfg.astra
    write_json(folder / "episode.json", {
        "episode": folder.name,
        "task": task,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "robot": {"name": "ElRobot", "profile": cfg.profile, "urdf": str(cfg.urdf),
                  "calibration": str(cfg.calibration), "bus": cfg.bus.driver},
        "cameras": list(cams),
        "anchor_camera": anchor_camera(cfg, list(cams)),
        "frames": {"base": "x forward, y left, z up (m)",
                   "tcp": "x out of the jaws, y the opening direction, z = x cross y",
                   "rotation": "roll, pitch, yaw deg; R = Rz(yaw) Ry(pitch) Rx(roll)",
                   "gripper": "opening between the fingers, mm"},
        "home_tcp": {"xyz": np.round(xyz, 4).tolist(), "rpy": np.round(rpy, 2).tolist(),
                     "gripper_mm": a.gripper_open_mm},
        "limits": {"gripper_mm": [a.gripper_closed_mm, a.gripper_open_mm],
                   "max_speed_m_s": a.max_speed_m_s, "max_rot_deg_s": a.max_rot_deg_s,
                   "max_accel_m_s2": a.max_accel_m_s2,
                   "max_rot_accel_deg_s2": a.max_rot_accel_deg_s2,
                   "max_gripper_mm_s": a.max_gripper_mm_s,
                   "joint_speed_margin": a.joint_speed_margin,
                   "max_rotation_deg": a.max_rotation_deg},
        "clock": None,                   # filled at tick 0
        "record": None,                  # P9's record of this rollout, once found
        "finish": None,
    })
    current = launch.episodes / "current"
    try:
        if current.is_symlink() or current.exists():
            current.unlink()
        current.symlink_to(folder.name)
    except OSError:
        pass
    return folder


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def update_json(path: Path, **fields) -> None:
    obj = json.loads(path.read_text())
    obj.update(fields)
    write_json(path, obj)


__all__ = ["SOCKET_DIR", "AstraLaunch", "anchor_camera", "check_workspace",
           "configured_cameras", "home_tcp", "new_episode", "prepare", "slug", "update_json",
           "write_json"]

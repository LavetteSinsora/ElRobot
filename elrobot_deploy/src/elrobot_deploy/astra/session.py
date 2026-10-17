"""An Astra session's folder: created before the launch, filled by the harness.

    launch = create(cfg, task="put the red cube in the cup")
    Launcher(cfg, astra=launch, ...).run()

`AstraLaunch` is what every process of the session is told (pickled at spawn): where the
folder is, where P9 records, and the two Unix sockets. The sockets live in a short directory
under /tmp, not in the folder: a socket path is limited to 104 bytes on macOS, and the
session folder's path (under the repo) is far longer.
"""

from __future__ import annotations

import json
import re
import shlex
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from elrobot_deploy.astra.frames import TaskFrame
from elrobot_deploy.astra.replies import pose_line

README_TEMPLATE = Path(__file__).with_name("session_readme.md")


@dataclass(frozen=True)
class AstraLaunch:
    session_id: str
    session_dir: str
    record_base: str
    socket_dir: str
    task: str

    @property
    def harness_socket(self) -> str:
        return str(Path(self.socket_dir) / "harness.sock")

    @property
    def player_socket(self) -> str:
        return str(Path(self.socket_dir) / "player.sock")

    @property
    def record_dir(self) -> str:
        return str(Path(self.record_base) / self.session_id)


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:n].rstrip("_") or "task"


def home_tcp(cfg) -> tuple[np.ndarray, np.ndarray]:
    """The home pose's TCP in Astra's base frame: (xyz m, rpy deg)."""
    from elrobot_deploy.robot.elrobot_model import ElRobotModel, mat_to_pose
    T = ElRobotModel(cfg.urdf).fk(cfg.poses.home[:7])
    return TaskFrame.from_config(cfg).describe(mat_to_pose(T))


def create(cfg, task: str, *, stamp: str | None = None) -> AstraLaunch:
    """Make the session folder (README.md, session.json) and the socket directory."""
    stamp = stamp or time.strftime("%Y-%m-%d_%H%M%S")
    sid = f"{stamp}_{slug(task)}"
    base = Path(cfg.astra.sessions_dir)
    folder = base / sid
    (folder / "actions").mkdir(parents=True, exist_ok=False)
    (folder / "frames").mkdir()
    sockets = Path("/tmp") / f"elrobot-astra-{stamp}"
    sockets.mkdir(parents=True, exist_ok=True)
    launch = AstraLaunch(session_id=sid, session_dir=str(folder),
                         record_base=str(cfg.record.dir), socket_dir=str(sockets), task=task)
    xyz, rpy = home_tcp(cfg)
    cams = [c for c in cfg.astra.cameras if c in {k.name for k in cfg.cameras}]
    a = cfg.astra
    write_readme(cfg, launch, cams, home=(xyz, rpy))
    write_json(folder / "session.json", {
        "session_id": sid,
        "task": task,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "robot": {"name": "ElRobot", "profile": cfg.profile, "urdf": str(cfg.urdf),
                  "calibration": str(cfg.calibration), "bus": cfg.bus.driver},
        "cameras": cams,
        "anchor_camera": anchor_camera(cfg, cams),
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
        "clock": None,                   # filled at the session start (tick 0)
        "record": launch.record_dir,
        "sockets": {"harness": launch.harness_socket, "player": launch.player_socket},
        "finish": None,
    })
    current = base / "current"
    try:
        if current.is_symlink() or current.exists():
            current.unlink()
        current.symlink_to(sid)
    except OSError:
        pass
    return launch


def write_readme(cfg, launch: AstraLaunch, cams, *, home=None) -> None:
    """The session's README.md (`session_readme.md` filled in). Written at creation with the
    profile's cameras, and again by the harness with the ones actually recorded."""
    xyz, rpy = home if home is not None else home_tcp(cfg)
    a = cfg.astra
    values = {
        "session_id": launch.session_id, "task": launch.task,
        "camera_list": ", ".join(f"`{c}` ({_camera_words(cfg, c)})" for c in cams) or "none",
        "anchor": anchor_camera(cfg, cams) or "first",
        "home_pose": pose_line(xyz, rpy, a.gripper_open_mm),
        "gripper_open_mm": f"{a.gripper_open_mm:g}", "gripper_closed_mm": f"{a.gripper_closed_mm:g}",
        "max_speed_m_s": f"{a.max_speed_m_s:g}", "max_rot_deg_s": f"{a.max_rot_deg_s:g}",
        "max_gripper_mm_s": f"{a.max_gripper_mm_s:g}",
        "max_rotation_deg": f"{a.max_rotation_deg:g}",
    }
    readme = string.Template(README_TEMPLATE.read_text()).substitute(values)
    (Path(launch.session_dir) / "README.md").write_text(readme)


def anchor_camera(cfg, cams) -> str | None:
    if cfg.astra.anchor_camera in cams:
        return cfg.astra.anchor_camera
    return cams[0] if cams else None


def _camera_words(cfg, name: str) -> str:
    cam = cfg.camera(name)
    if cam.lens == "fisheye":
        return "a fisheye on the wrist, looking along the jaws; a circular view"
    return "a fixed camera that sees the arm and the table"


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def update_json(path: Path, **fields) -> None:
    obj = json.loads(path.read_text())
    obj.update(fields)
    write_json(path, obj)


def opening_prompt(task: str, observation: str) -> str:
    return "\n".join([
        ("You are controlling a real robot arm (ElRobot: 7 joints and a two-finger gripper) "
         "through the MCP tools `send_action`, `inspect_observation` and `finish`. Your "
         "working directory is this session's folder: read README.md first. It explains the "
         "frames, the units, the tools and every file here."),
        "",
        f"Task: {task}",
        "",
        "When you are done (or cannot go on), call `finish` with your verdict.",
        "",
        observation,
        "",
    ])


def codex_script(launch: AstraLaunch) -> str:
    """`start_codex.sh`: Codex on this session, with the MCP server wired in by `-c`
    overrides (no edit of ~/.codex/config.toml) and the opening observation attached."""
    def toml(v) -> str:
        return json.dumps(v, ensure_ascii=False)

    args = ["-m", "elrobot_deploy.astra.mcp_server", "--session", launch.session_dir]
    overrides = [
        f"mcp_servers.elrobot.command={toml(sys.executable)}",
        f"mcp_servers.elrobot.args={toml(args)}",
        "mcp_servers.elrobot.startup_timeout_sec=30",
        # One send_action blocks for the whole motion; Codex's default (60 s) could cut a long
        # one off.
        "mcp_servers.elrobot.tool_timeout_sec=900",
    ]
    flags = " \\\n  ".join(f"-c {shlex.quote(o)}" for o in overrides)
    return "\n".join([
        "#!/bin/sh",
        "# Codex (GPT-6 Astra) on this session: the folder as its working directory, the",
        "# ElRobot MCP tools, the task and the opening observation. Run it in its own terminal",
        "# while `elrobot_deploy astra` runs. ASTRA_MODEL=<model> picks the model.",
        'cd "$(dirname "$0")" || exit 1',
        'exec codex ${ASTRA_MODEL:+-m "$ASTRA_MODEL"} \\',
        f"  {flags} \\",
        '  "$(cat opening_prompt.md)" \\',
        "  --image actions/0001/observation_now.jpg",
        "",
    ])


__all__ = ["AstraLaunch", "anchor_camera", "codex_script", "create", "home_tcp",
           "opening_prompt", "slug", "update_json", "write_json", "write_readme"]

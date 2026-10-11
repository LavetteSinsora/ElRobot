"""The robot a record was made on, rebuilt from the record itself.

Every manifest P9 writes carries the robot file chain it ran with, VERBATIM
(`rollout_record_utils/manifest.py::build`: `robot.files = [{name, shipped, path, text}, ...]`, base
first; `robot.name`, `robot.shipped`, `robot.extends`). That text is what this
module reads — not whatever the named profile says today — so the joints, the
end effector and its TCP are the ones the run used.

**The model is the one thing a record cannot carry**, so the rule is simple: a
record opens when its robot's `[model] urdf` is the package's own (a path
relative to the package that exists here: every shipped profile, and any
custom file that extends them without changing the model). Its geometry then
comes from this package's `assets/`, and nothing in the record is an absolute
path that has to exist on this machine — a record copied to ~/Downloads opens.
A record whose robot names a custom URDF, or that carries no robot file at
all, is refused with `UnsupportedRecord` rather than drawn as a guess.

`hardware(manifest)` builds the full `Hardware` a policy is constructed
with, for the policy replay: the record's robot, and an offline machine (no NIC,
the record's own CPU table), because a replay never talks to a robot.

Used by the loader, the IK bench and the policy replay. The manifest side:
`docs/setup/robot_toml.md`, "Records carry their robot".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class UnsupportedRecord(ValueError):
    """A record whose robot cannot be rebuilt on this machine."""


#: The message for a record whose model is not the package's own.
CUSTOM_MODEL = ("unsupported: this record's robot model (custom URDF) isn't "
                "available here")


@dataclass
class RecordRobot:
    """What the scene and the replay need to know about the recorded robot."""

    #: The profile name the manifest recorded.
    name: str
    #: Always "record": the robot file text came from the manifest.
    source: str
    #: The chain's file names as recorded (a shipped profile's name, else the
    #: custom file's).
    files: list[str]
    #: The merged TOML, `extends` resolved.
    raw: dict
    joints_arm: list[int]
    joints_waist: list[int]
    hand: str
    hand_min: float
    hand_max: float
    #: `(7,)` flange -> virtual TCP, or None when the robot has none.
    tcp: np.ndarray | None
    #: "record", or "none" when the robot file states no TCP.
    tcp_source: str
    #: The URDF in THIS package, resolved.
    urdf: str
    #: Always "package".
    urdf_source: str
    waist_joint_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _package_root() -> Path:
    from g1d_deploy.config_loading import resolved as config
    return config.MODEL_ROOT


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _tables(manifest: dict) -> tuple[list[dict], list[str]]:
    """The recorded chain's parsed tables, base first, and their names.
    Raises `UnsupportedRecord` when there is no chain or a file will not parse."""
    from g1d_deploy.config_loading.resolved import tomllib

    files = (manifest.get("robot") or {}).get("files") or []
    if not files:
        raise UnsupportedRecord("unsupported: the manifest carries no robot file")
    tables, names = [], []
    for f in files:
        try:
            table = tomllib.loads(str(f.get("text", "")))
        except Exception as exc:                            # noqa: BLE001
            raise UnsupportedRecord(
                f"unsupported: the recorded robot file {f.get('name')!r} does "
                f"not parse: {exc}") from exc
        table.pop("extends", None)
        tables.append(table)
        names.append(str(f.get("name") or Path(str(f.get("path", ""))).name))
    return tables, names


def merged_raw(manifest: dict) -> tuple[dict, list[str]]:
    """`(raw, file names)` from the manifest's robot file chain, merged in
    order the way `robot_file.load_robot` merges it, with the end effector
    resolved by the loader's own rule (`robot_file.end_effector_from_tables`)."""
    from g1d_deploy.config_loading.robot_file import end_effector_from_tables

    tables, names = _tables(manifest)
    raw: dict = {}
    for table in tables:
        raw = _merge(raw, table)
    for key in ("joints", "model"):
        if key not in raw:
            raise UnsupportedRecord(
                f"unsupported: the recorded robot file has no [{key}]")
    ee = end_effector_from_tables(tables)
    if not ee:
        raise UnsupportedRecord("unsupported: the recorded robot file names "
                                "no end effector")
    raw["end_effector"] = ee
    return raw, names


def package_urdf(raw: dict) -> str:
    """The recorded `[model] urdf` in THIS package, or `UnsupportedRecord`.

    Only a path relative to the package that exists here is the package's own
    model; an absolute path is a custom URDF on the recording machine."""
    rel = str((raw.get("model") or {}).get("urdf") or "")
    p = Path(rel)
    if not rel or p.is_absolute() or not (_package_root() / p).is_file():
        raise UnsupportedRecord(CUSTOM_MODEL)
    return str(_package_root() / p)


def from_manifest(manifest: dict) -> RecordRobot:
    """The recorded robot, or `UnsupportedRecord`."""
    from g1d_deploy.config_loading.resolved import parse_tcp
    from g1d_deploy.robot.end_effectors import END_EFFECTORS

    raw, files = merged_raw(manifest)
    urdf = package_urdf(raw)
    name = (manifest.get("robot") or {}).get("name") or ""
    kind = raw["end_effector"]["type"]
    ee = END_EFFECTORS.get(kind)
    if ee is None:
        raise UnsupportedRecord(f"unsupported: end effector {kind!r} is not "
                                f"one this version knows")
    notes: list[str] = []
    tcp = parse_tcp(raw["end_effector"].get("tcp"))
    tcp_source = "record"
    if tcp is None:
        tcp_source = "none"
        notes.append(f"tcp: the robot file names no [end_effector] tcp "
                     f"(type {kind}); no TCP marker is drawn")

    j = raw.get("joints") or {}
    model = manifest.get("model") or {}
    return RecordRobot(
        name=name, source="record", files=files, raw=raw,
        joints_arm=list(j["arm"]), joints_waist=list(j["waist"]),
        hand=kind, hand_min=ee.hand_min, hand_max=ee.hand_max,
        tcp=tcp, tcp_source=tcp_source, urdf=urdf, urdf_source="package",
        waist_joint_names=list(model.get("waist_joint_names") or ()),
        notes=notes)


def robot_object(manifest: dict):
    """A `robot_file.Robot` for the recorded chain — what `resolved.load_hardware`
    takes. Cameras that no longer parse under today's rules are dropped (a
    replay reads frames from the record, never from a camera)."""
    from g1d_deploy.config_loading.robot_file import Robot, RobotFile, _camera

    rr = from_manifest(manifest)
    raw = copy.deepcopy(rr.raw)
    # The model path `load_hardware` resolves: the package's own, relative.
    cameras = {}
    for name, spec in sorted((raw.get("cameras") or {}).items()):
        try:
            cameras[name] = _camera(name, spec)
        except Exception:                                   # noqa: BLE001
            continue
    files = tuple(RobotFile(str(f.get("name") or f.get("path", "")),
                            str(f.get("text", "")))
                  for f in (manifest.get("robot") or {}).get("files") or ())
    board = raw.get("image_server") or {}
    return Robot(name=rr.name or "record", files=files, raw=raw,
                 cameras=cameras, end_effector=rr.hand,
                 image_server=str(board.get("type", "none")))


def hardware(manifest: dict):
    """`Hardware` for the recorded robot on an offline machine."""
    from g1d_deploy.config_loading.resolved import load_hardware
    from g1d_deploy.config_loading.machine_file import DEFAULT_SUBNET, Machine

    cpu = manifest.get("cpu") or {}
    pin = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
           for k, v in cpu.items()}
    machine = Machine(
        path=None, text=None,
        iface="", subnet=DEFAULT_SUBNET,
        cpu={"pin": pin, "reserved": list(manifest.get("cpu_reserved") or ())},
        sources={"iface": "inspection (offline)"})
    return load_hardware(robot_object(manifest), machine)


__all__ = ["CUSTOM_MODEL", "RecordRobot", "UnsupportedRecord", "from_manifest",
           "hardware", "merged_raw", "package_urdf", "robot_object"]

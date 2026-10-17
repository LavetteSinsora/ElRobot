"""Robot files: what the robot IS — joints, gains, limits, URDF, the end
effector, and the cameras that exist.

Read at config resolution (`resolved.load_hardware`, `deploy_file.resolve`),
before any process starts. Design: `docs/setup/robot_toml.md`.

Shipped profiles live in `robot/profiles/*.toml` and are loaded **by name**
(`load_robot("g1d_fisheye_dex1")`); anything with a `/` or a `.toml` suffix is
a path. A file says `extends = "<name or path>"` to start from another one;
tables merge key by key, the extending file winning, so a custom rig is a few
lines on top of a shipped profile (`examples/robot_custom.toml`).

**No camera flags.** A robot file lists the cameras that exist, each under
`[cameras.<name>]`; CHECK waits for exactly those. A rig with another camera
set or another hand is a different file, not a flag.

The loader's output is what the rest of the runtime consumes:
`resolved.Hardware` (via `resolved.load_hardware`) plus a `CameraSpec` per camera.
Everything the robot file says that the runtime does not use — a camera's
`intrinsics` / `extrinsics` — is carried through untouched for the policy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from elrobot_deploy.robot import end_effectors
from elrobot_deploy.config_loading.resolved import (CAMERA_SOURCES, CameraSpec, parse_tcp,
                                                tomllib)
from elrobot_deploy.config_loading.errors import ConfigError

#: Where the shipped profiles are. Inside the package, because they are loaded
#: by name at runtime and files outside `src/` are not installed.
ROBOTS_DIR = Path(__file__).resolve().parents[1] / "robot" / "profiles"

#: The profile every entry point assumes when none is named: the rig every
#: rollout on record ran on.
DEFAULT_ROBOT = "g1d_fisheye_dex1"

#: `[cameras.<name>] kind`. It drives the lens model: a `fisheye180` camera
#: has a lens circle, fitted at every launch (`runtime/drivers/fisheye_circle_fit.py`) and the board's
#: exposure servo meters inside it; a `normal` camera is picture edge to edge.
CAMERA_KINDS = ("fisheye180", "normal")

#: `[cameras.<name>] layout`. `side_by_side` is the G1-D head: one frame, two
#: eyes. Declared, not inferred from the aspect ratio.
CAMERA_LAYOUTS = ("side_by_side",)

#: `[image_server] type`: who publishes the `zmq` cameras on the robot board.
#:
#: * `fisheye`     ours (`fish_eye_image_server/`); P0 SSHes in (the machine
#:                 file's `[board]` login), stops the vendor's teleimager and
#:                 starts it from `dir` in conda env `conda_env`
#: * `teleimager`  the vendor's systemd service; P0 never touches the board
#: * `none`        no board cameras (the default when the table is absent)
IMAGE_SERVERS = ("fisheye", "teleimager", "none")

#: Keys `[image_server]` may carry.
_IMAGE_SERVER_KEYS = {"type", "dir", "conda", "conda_env"}


@dataclass(frozen=True)
class ImageServer:
    """`[image_server]`: which server publishes the board's cameras, and —
    for ours — where it is installed on the board."""

    type: str = "none"
    #: The server's directory on the board (`publisher.py` lives there).
    dir: str | None = None
    #: The board's conda installation (`<conda>/etc/profile.d/conda.sh`).
    conda: str | None = None
    #: The conda env with pyuvc, pyzmq and pyyaml.
    conda_env: str | None = None


def _image_server(raw: dict | None) -> ImageServer:
    if raw is None:
        return ImageServer()
    where = "[image_server]"
    unknown = sorted(set(raw) - _IMAGE_SERVER_KEYS)
    if unknown:
        raise ConfigError(f"{where} has unknown keys {unknown}; known: "
                          f"{sorted(_IMAGE_SERVER_KEYS)}")
    kind = raw.get("type")
    if kind not in IMAGE_SERVERS:
        raise ConfigError(f"{where} type = {kind!r}, must be one of "
                          f"{list(IMAGE_SERVERS)}")
    spec = ImageServer(type=kind, dir=raw.get("dir"), conda=raw.get("conda"),
                       conda_env=raw.get("conda_env"))
    if kind == "fisheye":
        missing = [k for k in ("dir", "conda", "conda_env")
                   if not isinstance(getattr(spec, k), str) or not getattr(spec, k)]
        if missing:
            raise ConfigError(
                f"{where} type = \"fisheye\" needs {missing}: where the server "
                f"is installed on the board, e.g.\n\n    [image_server]\n    "
                f"type = \"fisheye\"\n    dir = \"/home/unitree/"
                f"fish_eye_image_server\"\n    conda = \"/home/unitree/"
                f"miniconda3\"\n    conda_env = \"tv\"")
    return spec

#: Keys a camera table may carry. Anything else is a typo, refused by name —
#: a misspelt `circle_defualt` would otherwise run on no fallback at all.
_CAMERA_KEYS = {"source", "host", "port", "device", "serial", "shape", "hz",
                "kind", "layout", "fallback_lens_circle",
                "intrinsics", "extrinsics"}


#: The G1-D's factory address and login for its onboard computer.
DEFAULT_BOARD_IP = "192.168.123.164"
DEFAULT_BOARD_SSH_USER = "unitree"
DEFAULT_BOARD_SSH_PASSWORD = "123"
_BOARD_KEYS = {"ip", "ssh_user", "ssh_password"}


@dataclass(frozen=True)
class Board:
    """`[board]`: the robot's onboard computer — where its cameras are read
    from, and the login P0 uses to start the fisheye image server there
    (`runtime/drivers/board/image_server_launch.py`). A fact about one robot,
    so it is the robot file's; the defaults are the G1-D's factory values."""

    ip: str = DEFAULT_BOARD_IP
    ssh_user: str = DEFAULT_BOARD_SSH_USER
    ssh_password: str = field(default=DEFAULT_BOARD_SSH_PASSWORD, repr=False)


def _board(raw: dict | None) -> Board:
    raw = dict(raw or {})
    unknown = sorted(set(raw) - _BOARD_KEYS)
    if unknown:
        raise ConfigError(f"[board] has unknown keys {unknown}; known: "
                          f"{sorted(_BOARD_KEYS)}")
    return Board(ip=str(raw.get("ip", DEFAULT_BOARD_IP)),
                 ssh_user=str(raw.get("ssh_user", DEFAULT_BOARD_SSH_USER)),
                 ssh_password=str(raw.get("ssh_password", DEFAULT_BOARD_SSH_PASSWORD)))


@dataclass(frozen=True)
class RobotFile:
    """One file of the `extends` chain, as read — the manifest copies these."""

    path: str
    text: str


@dataclass(frozen=True)
class Robot:
    """A robot file, `extends` resolved.

    `raw` is the merged TOML, which `resolved.load_hardware` reads the robot's
    half of `Hardware` from. `files` is the chain base-first, verbatim, so a
    record can say exactly what it ran against even after a profile changes.
    """

    name: str
    files: tuple[RobotFile, ...]
    raw: dict
    cameras: dict[str, CameraSpec]
    end_effector: str
    #: `[image_server] type` (`IMAGE_SERVERS`).
    image_server: str
    #: The whole `[image_server]` table.
    image_server_spec: ImageServer = ImageServer()
    #: `[board]`: the robot's onboard computer.
    board: Board = Board()

    @property
    def path(self) -> str:
        return self.files[-1].path

    def describe(self) -> str:
        cams = ", ".join(f"{n} ({s.source}{'' if s.port is None else f':{s.port}'}"
                         f" {s.height}x{s.width} {s.kind})"
                         for n, s in sorted(self.cameras.items()))
        return (f"{self.name}  [{' < '.join(Path(f.path).stem for f in self.files)}]"
                f"  hand={self.end_effector}  cameras: {cams or 'none'}")


#: Keys `[end_effector]` may carry. Per-type specifics (topics, travel) are
#: built in (`robot/end_effectors.py`), not written here.
_END_EFFECTOR_KEYS = {"type", "tcp"}


def end_effector_of(ee: dict) -> tuple[str, object]:
    """A resolved `[end_effector]` -> `(type, tcp as written or None)`."""
    return str(ee["type"]), ee.get("tcp")


_SHAPE = (f"    [end_effector]\n    type = \"dex1\"          # "
          f"{' | '.join(end_effectors.TYPES)}\n"
          f"    tcp = [x, y, z, qw, qx, qy, qz]   # optional, flange frame")


def _check_end_effector(path: Path, table: dict) -> None:
    """One file's `[end_effector]`: only `type` and `tcp`, a known type, and
    a TCP that parses."""
    where = f"robot file {path} [end_effector]"
    unknown = sorted(set(table) - _END_EFFECTOR_KEYS)
    if unknown:
        raise ConfigError(f"{where} has unknown keys {unknown}; the table is\n\n"
                          f"{_SHAPE}\n\n(each type's topics and travel are built "
                          f"in: elrobot_deploy/robot/end_effectors.py)")
    if "type" in table and table["type"] not in end_effectors.TYPES:
        raise ConfigError(f"{where} type = {table['type']!r}, must be one of "
                          f"{list(end_effectors.TYPES)}")
    if "tcp" in table:
        parse_tcp(table["tcp"], f"{where} tcp")


def end_effector_from_tables(tables) -> dict:
    """The `[end_effector]` of an `extends` chain (tables base first), with the
    one rule plain merging gets wrong: **a TCP belongs to its type.** A file
    that changes `type` without stating its own `tcp` drops the one it
    inherited — a BrainCo profile extending a Dex-1 one must not pivot, draw or
    solve at the Dex-1's jaw midpoint. Returns `{"type", "tcp"?}`, or {} when
    no table names a type."""
    ee_type, tcp = None, None
    for table in tables:
        ee = table.get("end_effector")
        if not isinstance(ee, dict):
            continue
        new_type = ee.get("type", ee_type)
        if new_type != ee_type and "tcp" not in ee:
            tcp = None
        ee_type = new_type
        if "tcp" in ee:
            tcp = ee["tcp"]
    if ee_type is None:
        return {}
    return {"type": str(ee_type), **({"tcp": tcp} if tcp is not None else {})}


def _end_effector(chain) -> dict:
    """`end_effector_from_tables` over a live chain, every file checked."""
    for path, _text, table in chain:
        if "end_effector" in table:
            _check_end_effector(path, table["end_effector"])
    return end_effector_from_tables([table for _p, _t, table in chain])


def resolve_path(name_or_path: str | Path, *, relative_to: Path | None = None) -> Path:
    """A shipped profile name, or a path, -> the file.

    A bare name (no `/`, no `.toml`) is a shipped profile. Anything else is a
    path, relative to `relative_to` (the file that names it) when given — so a
    deploy file or a custom robot file can say `extends = "my_rig.toml"` and
    mean the one beside it, wherever it is launched from.
    """
    s = str(name_or_path)
    if "/" not in s and not s.endswith(".toml"):
        p = ROBOTS_DIR / f"{s}.toml"
        if not p.is_file():
            known = sorted(q.stem for q in ROBOTS_DIR.glob("*.toml"))
            raise ConfigError(f"no shipped robot profile {s!r}; shipped: {known}. "
                              f"A custom robot file is named by its path.")
        return p
    p = Path(s).expanduser()
    if not p.is_absolute() and relative_to is not None:
        p = relative_to / p
    p = p.resolve()
    if not p.is_file():
        raise ConfigError(f"robot file {s!r} does not exist (looked at {p})")
    return p


def _merge(base: dict, over: dict) -> dict:
    """Deep merge, `over` winning. Tables merge key by key; anything else —
    including an array — is replaced whole, because half a joint list is not a
    joint list."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _chain(path: Path, seen: tuple[Path, ...] = ()) -> list[tuple[Path, str, dict]]:
    """`[(path, text, table), ...]` base-first, following `extends`."""
    if path in seen:
        raise ConfigError(f"robot files extend each other in a loop: "
                          f"{' -> '.join(str(p) for p in (*seen, path))}")
    text = path.read_text()
    try:
        table = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"robot file {path}: {exc}") from exc
    parent = table.pop("extends", None)
    if parent is None:
        return [(path, text, table)]
    return _chain(resolve_path(parent, relative_to=path.parent),
                  (*seen, path)) + [(path, text, table)]


def _camera(name: str, raw: dict) -> CameraSpec:
    """One `[cameras.<name>]` table -> a `CameraSpec`, every key checked."""
    where = f"[cameras.{name}]"
    unknown = set(raw) - _CAMERA_KEYS
    if unknown:
        raise ConfigError(f"{where} has unknown keys {sorted(unknown)}; "
                          f"known: {sorted(_CAMERA_KEYS)}")
    source = raw.get("source")
    if source not in CAMERA_SOURCES:
        raise ConfigError(f"{where} source = {source!r}, must be one of "
                          f"{list(CAMERA_SOURCES)}")
    shape = raw.get("shape")
    if not (isinstance(shape, list) and len(shape) == 2
            and all(isinstance(v, int) and v > 0 for v in shape)):
        raise ConfigError(f"{where} shape = {shape!r}, must be [height, width] "
                          f"in pixels — the native resolution P2 decodes at")
    kind = raw.get("kind", "normal")
    if kind not in CAMERA_KINDS:
        raise ConfigError(f"{where} kind = {kind!r}, must be one of {list(CAMERA_KINDS)}")
    layout = raw.get("layout")
    if layout is not None and layout not in CAMERA_LAYOUTS:
        raise ConfigError(f"{where} layout = {layout!r}, must be one of "
                          f"{list(CAMERA_LAYOUTS)} or absent")

    port = device = None
    if source == "zmq":
        if not isinstance(raw.get("port"), int):
            raise ConfigError(f"{where} source = \"zmq\" needs an integer `port` "
                              f"on the board's image server")
        port = int(raw["port"])
    elif source == "v4l2":
        # No "auto" here: nothing in the runtime can yet tell two fisheye units
        # apart by USB id (they share a serial), so the only stable address is
        # the socket — a /dev/v4l/by-path node. `resolved.CameraSpec` says why.
        device = raw.get("device")
        if not device or device == "auto":
            raise ConfigError(f"{where} source = \"v4l2\" needs `device` = a "
                              f"/dev/v4l/by-path/... node; `ls -l /dev/v4l/by-path/`")
    elif source == "realsense":
        serial = raw.get("serial", "auto")
        device = None if serial in (None, "", "auto") else str(serial)

    circle = raw.get("fallback_lens_circle")
    if circle is not None:
        try:
            circle = (float(circle["center_x"]), float(circle["center_y"]),
                      float(circle["radius"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{where} fallback_lens_circle must be {{ center_x = .., "
                              f"center_y = .., radius = .. }} in full-sensor "
                              f"pixels") from exc
    if kind == "fisheye180" and layout == "side_by_side":
        raise ConfigError(f"{where} a fisheye180 camera cannot be side_by_side")

    return CameraSpec(
        height=int(shape[0]), width=int(shape[1]), port=port, source=source,
        device=device, hz=float(raw.get("hz", 30.0)),
        binocular=layout == "side_by_side",
        host=raw.get("host"), kind=kind, layout=layout,
        fallback_lens_circle=circle,
        intrinsics=dict(raw["intrinsics"]) if "intrinsics" in raw else None,
        extrinsics=dict(raw["extrinsics"]) if "extrinsics" in raw else None)


def load_robot(name_or_path: str | Path = DEFAULT_ROBOT, *,
               relative_to: Path | None = None) -> Robot:
    """Read a robot file and everything it extends. Every refusal a robot file
    can earn is raised here, as `ConfigError`, before anything is spawned."""
    path = resolve_path(name_or_path, relative_to=relative_to)
    chain = _chain(path)
    raw: dict = {}
    for _p, _t, table in chain:
        raw = _merge(raw, table)

    for key in ("rates", "dds", "joints", "gains", "limits", "model"):
        if key not in raw:
            raise ConfigError(
                f"robot file {path} has no [{key}] after resolving `extends`. A rig "
                f"profile should start with extends = \"g1d_base\".")

    server = _image_server(raw.get("image_server"))
    board = _board(raw.get("board"))

    cameras = {name: _camera(name, spec)
               for name, spec in sorted(raw.get("cameras", {}).items())}

    ee = _end_effector(chain)
    if not ee:
        raise ConfigError(
            f"robot file {path} names no end effector. A rig profile says which "
            f"one is bolted on:\n\n{_SHAPE}")
    raw["end_effector"] = ee

    return Robot(
        name=path.stem if path.parent == ROBOTS_DIR else str(path),
        files=tuple(RobotFile(str(p), t) for p, t, _ in chain),
        raw=raw, cameras=cameras, end_effector=ee["type"],
        image_server=server.type, image_server_spec=server, board=board)


def shipped() -> list[str]:
    """The shipped profile names."""
    return sorted(p.stem for p in ROBOTS_DIR.glob("*.toml"))


__all__ = ["CAMERA_KINDS", "CAMERA_LAYOUTS", "DEFAULT_ROBOT", "IMAGE_SERVERS",
           "ImageServer", "ROBOTS_DIR", "Robot", "RobotFile", "load_robot", "resolve_path",
           "shipped"]

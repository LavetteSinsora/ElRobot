"""The robot profile, resolved into the one settings object every process is handed.

    cfg = load()                                   # robot/profiles/elrobot.toml
    cfg = load("my_arm.toml", sets=["bus.driver=sim", "teleop.lead_s=0.05"])

A profile is a TOML file (`robot/profiles/elrobot.toml` is the shipped one and documents every
key). `sets` override single keys for one run, dotted, with TOML values (`--set` on the command
line). The result is a frozen `Config`: what the robot is (URDF, calibration, limits, poses),
how its bus is driven (`bus`, `executor`) and how it is driven by a person (`ik`, `teleop`).
It is pickled into every child process at spawn, so it holds plain values and numpy arrays only.

Joint limits are not configured here: they are the URDF's (`<limit lower/upper>`, the
calibrated safe ranges written by `description/elrobot_ours/make_description.py`), and the
velocity limit is the URDF's `<limit velocity>` times `limits.velocity_scale` (the gripper's times
`limits.gripper_velocity_scale`).
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.robot.elrobot_joints import N_JOINTS, JointLimits, urdf_limits
from elrobot_deploy.runtime.execution.bus_servo import ExecutorSettings
from elrobot_deploy.runtime.execution.guard import GuardSettings

#: The repository root (ElRobot/): relative paths in a profile are relative to it.
REPO = Path(__file__).resolve().parents[4]
PROFILES = Path(__file__).resolve().parents[1] / "robot" / "profiles"
DEFAULT_PROFILE = "elrobot"

_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
         "+z": (0, 0, 1), "-z": (0, 0, -1), "x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


@dataclass(frozen=True)
class BusConfig:
    driver: str = "st3215"           # st3215 | sim
    port: str = ""                   # "" = autodetect the adapter
    hz: float = 250.0
    slow_every: int = 5


@dataclass(frozen=True)
class DriverGains:
    """What the bus process writes to the servos at connect (and restores at release)."""

    p_coef: dict = field(default_factory=dict)          # servo id -> P
    d_coef: dict = field(default_factory=dict)          # servo id -> D
    torque_limit: dict = field(default_factory=dict)    # servo id -> 0.1 % of stall
    goal_speed_factor: float = 1.5
    acceleration: int = 0


@dataclass(frozen=True)
class Poses:
    home: np.ndarray                 # (8,) rad
    exit: np.ndarray                 # (8,) rad
    min_s: float = 2.0
    velocity_margin: float = 0.5
    home_tol: float = math.radians(5.0)
    release_at_exit: bool = True


@dataclass(frozen=True)
class IKConfig:
    solver: str = "tcp_qp"
    options: dict = field(default_factory=dict)         # keyword arguments for the factory


@dataclass(frozen=True)
class TeleopConfig:
    control_hz: float = 60.0
    lead_s: float = 0.040
    port: int = 5005
    stale_s: float = 0.25
    scale: float = 1.0
    gripper_rate: float = math.radians(360.0)           # rad/s
    max_step_m: float = 0.05
    max_step: float = math.radians(30.0)                # rad
    phone_forward: tuple = (-1.0, 0.0, 0.0)             # device axis, ARKit camera frame
    base_forward: tuple = (0.0, 1.0, 0.0)
    base_up: tuple = (0.0, 0.0, 1.0)
    #: An optional box (base frame, m) the TCP is clamped into; None = none (the default).
    workspace_min: tuple | None = None
    workspace_max: tuple | None = None


@dataclass(frozen=True)
class DashboardConfig:
    host: str = "127.0.0.1"           # 0.0.0.0 to open it from another device on the LAN
    port: int = 8080


@dataclass(frozen=True)
class CameraConfig:
    """One `[cameras.<name>]` table (the profile documents every key)."""

    name: str
    source: str = "usb"              # usb | avfoundation | v4l2 | realsense | fake
    device: str = "auto"             # which camera (see the profile)
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    lens: str = "fisheye"            # fisheye: fit the circle, publish its square | normal
    fallback_lens_circle: tuple | None = None    # (cx, cy, r) px at this resolution
    exposure: str = "camera_first"   # runtime/cameras/exposure.py, "THE MODES"
    takeover_clip_pct: float = 15.0  # camera_first: % of the disc clipped that is "failing"
    pinned_exposure: int = 200       # `pinned`: 100 us units
    pinned_gain: int = 64
    white_balance: int = 0           # kelvin to pin while we own exposure; 0 = leave it auto
    slots: int = 8                   # frames the arena keeps
    required: bool = False           # refuse the session if this camera cannot start
    use: str = "always"              # always | astra (only `elrobot_deploy astra` starts it)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    def line(self) -> str:
        return (f"{self.name}: {self.source} {self.device!r} {self.width}x{self.height} "
                f"@ {self.fps:g} fps, {self.lens}, exposure {self.exposure}"
                f"{', required' if self.required else ''}")


CAMERA_SOURCES = ("usb", "avfoundation", "v4l2", "realsense", "fake")
CAMERA_USES = ("always", "astra")


@dataclass(frozen=True)
class RecordConfig:
    """`[record]`: P9, the recorder (`runtime/processes/p9_record.py`)."""

    dir: Path = REPO / "logs" / "records"
    jpeg_quality: int = 90


@dataclass(frozen=True)
class AstraConfig:
    """`[astra]`: the harness that lets an agent drive the arm (`astra/`, `docs/astra.md`).
    Speeds are the trajectory law's caps in the TCP's own units; the joint speeds the IK needs
    are held to `joint_speed_margin` x the run's velocity limit on top of them."""

    sessions_dir: Path = REPO / "logs" / "astra"
    cameras: tuple = ("exo", "wrist")       # left to right in every image; absent ones skipped
    anchor_camera: str = "wrist"            # whose frames are the ticks (else the first running)
    gripper_open_mm: float = 55.0           # the jaws' inner gap at the open end (3 deg)
    gripper_closed_mm: float = 0.0          # ... and at the closed end of the calibrated range
    max_speed_m_s: float = 0.10
    max_rot_deg_s: float = 45.0
    max_accel_m_s2: float = 0.5
    max_rot_accel_deg_s2: float = 180.0
    max_gripper_mm_s: float = 60.0
    joint_speed_margin: float = 0.8
    min_segment_s: float = 0.1
    settle_s: float = 0.3
    max_rotation_deg: float = 170.0
    jpeg_quality: int = 90


@dataclass(frozen=True)
class Config:
    profile: str                      # the file it came from
    urdf: Path
    calibration: Path
    limits: JointLimits               # the URDF's, velocity unscaled
    velocity_scale: float
    bus: BusConfig
    executor: ExecutorSettings
    gains: DriverGains
    poses: Poses
    ik: IKConfig
    teleop: TeleopConfig
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    guard: GuardSettings = field(default_factory=GuardSettings)
    #: The `[cameras.<name>]` tables, in name order; () = no camera process.
    cameras: tuple = ()
    record: RecordConfig = field(default_factory=RecordConfig)
    astra: AstraConfig = field(default_factory=AstraConfig)
    #: Every key that a `--set` changed, -> its value as given. For the run's printout.
    overrides: dict = field(default_factory=dict)

    @property
    def qdot_max(self) -> np.ndarray:
        """(8,) rad/s: the velocity limit everything after the IK is held to."""
        scale = np.full(N_JOINTS, self.velocity_scale)
        scale[N_JOINTS - 1] = self.executor.gripper_velocity_scale
        return self.limits.velocity * scale

    @property
    def dt(self) -> float:
        """The target grid's period: one target per phone frame."""
        return 1.0 / self.teleop.control_hz

    def driver_kw(self) -> dict:
        """The keyword arguments `st3215_driver.make_driver(self.bus.driver, ...)` takes."""
        if self.bus.driver == "sim":
            return {"q0": self.poses.exit.copy(), "qdot_max": float(np.max(self.qdot_max))}
        g = self.gains
        return {"port": self.bus.port or None, "calibration": str(self.calibration),
                "qdot_max": self.qdot_max.copy(), "goal_speed_factor": g.goal_speed_factor,
                "acceleration": g.acceleration, "torque_limit": dict(g.torque_limit) or None,
                "p_coef": dict(g.p_coef), "d_coef": dict(g.d_coef)}

    def with_driver(self, driver: str) -> Config:
        return replace(self, bus=replace(self.bus, driver=driver))

    def with_cameras(self, *, fake: bool = False, none: bool = False) -> Config:
        """Every camera replaced by the synthetic one (`fake`), or no cameras (`none`)."""
        if none:
            return replace(self, cameras=())
        if fake:
            return replace(self, cameras=tuple(replace(c, source="fake") for c in self.cameras))
        return self

    def cameras_for(self, *, astra: bool) -> Config:
        """The cameras a launch starts: an ordinary session leaves out the `use = "astra"`
        ones (the exocentric iPhone: a Continuity Camera would take the teleop phone's
        camera, and nothing but the harness looks at it)."""
        if astra:
            return self
        return replace(self, cameras=tuple(c for c in self.cameras if c.use != "astra"))

    def camera(self, name: str) -> CameraConfig:
        for c in self.cameras:
            if c.name == name:
                return c
        raise ConfigError(f"no camera {name!r} (the profile has "
                          f"{[c.name for c in self.cameras] or 'none'})")

    def describe(self) -> str:
        e, t = self.executor, self.teleop
        return "\n".join([
            f"profile     {self.profile}",
            f"urdf        {self.urdf}",
            f"bus         {self.bus.driver} {self.bus.port or '(autodetect)'} "
            f"{self.bus.hz:g} Hz",
            f"executor    filter {e.filter_hz:g} Hz{' lead' if e.filter_lead else ''}, "
            f"squeeze {math.degrees(e.squeeze_margin):.1f} deg, friction "
            f"{np.round(np.degrees(np.broadcast_to(e.friction, (N_JOINTS,))), 2).tolist()} deg",
            f"gains       P {self.gains.p_coef or '(servo)'}  D {self.gains.d_coef or '(servo)'}",
            f"limits      velocity x {self.velocity_scale:g} = "
            f"{np.round(self.qdot_max, 2).tolist()} rad/s",
            f"poses       home {np.round(np.degrees(self.poses.home), 1).tolist()} deg",
            f"            exit {np.round(np.degrees(self.poses.exit), 1).tolist()} deg"
            f"{' (released there)' if self.poses.release_at_exit else ''}",
            f"ik          {self.ik.solver} {self.ik.options}",
            f"teleop      {t.control_hz:g} Hz, lead {t.lead_s * 1e3:.0f} ms, UDP {t.port}",
        ] + [f"camera      {c.line()}" for c in self.cameras] + [f"--set      {k} = {v!r}" for k, v in self.overrides.items()])


# --- loading --------------------------------------------------------------------------------


def profile_path(name: str | Path | None = None) -> Path:
    """A shipped profile's name (`elrobot`) or a path to one."""
    name = DEFAULT_PROFILE if name is None else name
    p = Path(name)
    if p.suffix == ".toml" or p.is_file():
        if not p.is_file():
            raise ConfigError(f"no profile at {p}")
        return p.resolve()
    shipped = PROFILES / f"{name}.toml"
    if not shipped.exists():
        names = sorted(f.stem for f in PROFILES.glob("*.toml"))
        raise ConfigError(f"unknown profile {name!r}: shipped profiles are {names}, or give a path")
    return shipped


def _parse_value(text: str):
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_sets(tree: dict, sets) -> dict:
    """`section.key=value` strings onto a parsed profile (a new dict). Unknown keys are refused:
    a typo must not silently do nothing. A camera's keys are one level deeper:
    `cameras.wrist.exposure=...` (any `CameraConfig` key, listed in its table or not)."""
    import copy
    out = copy.deepcopy(tree)
    for item in sets or ():
        key, eq, text = str(item).partition("=")
        if not eq or "." not in key:
            raise ConfigError(f"--set {item!r}: expected section.key=value")
        *path, name = key.strip().split(".")
        node, where = out, ""
        for part in path:
            if not isinstance(node.get(part), dict):
                have = sorted(k for k, v in node.items() if isinstance(v, dict))
                raise ConfigError(f"--set {item!r}: no table [{where + part}] (have {have})")
            node, where = node[part], f"{where}{part}."
        open_keys = path == ["ik"] or (len(path) == 2 and path[0] == "cameras"
                                       and name in CameraConfig.__dataclass_fields__)
        if name not in node and not open_keys:
            raise ConfigError(f"--set {item!r}: [{where[:-1]}] has no key {name!r} "
                              f"(have {sorted(node)})")
        node[name] = _parse_value(text.strip())
    return out


def _cameras(tables: dict) -> tuple:
    """The `[cameras.<name>]` tables, validated, in name order."""
    from elrobot_deploy.runtime.cameras.exposure import MODES
    out = []
    for name in sorted(tables):
        t = tables[name]
        if not isinstance(t, dict):
            raise ConfigError(f"[cameras] {name}: expected a table [cameras.{name}]")
        fields = CameraConfig.__dataclass_fields__
        unknown = set(t) - set(fields) - {"name"}
        if unknown:
            raise ConfigError(f"[cameras.{name}]: unknown keys {sorted(unknown)} "
                              f"(have {sorted(set(fields) - {'name'})})")
        d = CameraConfig(name=name)
        circle = t.get("fallback_lens_circle")
        cam = CameraConfig(
            name=name, source=str(t.get("source", d.source)),
            device=str(t.get("device", d.device)), width=int(t.get("width", d.width)),
            height=int(t.get("height", d.height)), fps=float(t.get("fps", d.fps)),
            lens=str(t.get("lens", d.lens)),
            fallback_lens_circle=tuple(float(v) for v in circle) if circle else None,
            exposure=str(t.get("exposure", d.exposure)),
            takeover_clip_pct=float(t.get("takeover_clip_pct", d.takeover_clip_pct)),
            pinned_exposure=int(t.get("pinned_exposure", d.pinned_exposure)),
            pinned_gain=int(t.get("pinned_gain", d.pinned_gain)),
            white_balance=int(t.get("white_balance", d.white_balance)),
            slots=int(t.get("slots", d.slots)), required=bool(t.get("required", d.required)),
            use=str(t.get("use", d.use)))
        if cam.source not in CAMERA_SOURCES:
            raise ConfigError(f"[cameras.{name}] source {cam.source!r}: one of {CAMERA_SOURCES}")
        if cam.use not in CAMERA_USES:
            raise ConfigError(f"[cameras.{name}] use {cam.use!r}: one of {CAMERA_USES}")
        if cam.lens not in ("fisheye", "normal"):
            raise ConfigError(f"[cameras.{name}] lens {cam.lens!r}: fisheye or normal")
        if cam.exposure not in MODES:
            raise ConfigError(f"[cameras.{name}] exposure {cam.exposure!r}: one of {MODES}")
        if cam.fallback_lens_circle is not None and len(cam.fallback_lens_circle) != 3:
            raise ConfigError(f"[cameras.{name}] fallback_lens_circle: [cx, cy, r] in pixels")
        if not (cam.width > 0 and cam.height > 0 and cam.fps > 0 and cam.slots >= 3):
            raise ConfigError(f"[cameras.{name}]: width, height, fps > 0 and slots >= 3")
        out.append(cam)
    return tuple(out)


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else (REPO / p)


def _per_joint(values, name: str, convert=float) -> np.ndarray | None:
    if values is None or len(values) == 0:
        return None
    if len(values) != N_JOINTS:
        raise ConfigError(f"{name}: {len(values)} values, expected {N_JOINTS} (servo order)")
    return np.array([convert(v) for v in values], dtype=np.float64)


def _ids(values, name: str) -> dict:
    a = _per_joint(values, name, int)
    return {} if a is None else {i + 1: int(v) for i, v in enumerate(a)}


def _axis(value, name: str) -> tuple:
    if isinstance(value, str):
        if value.lower() not in _AXES:
            raise ConfigError(f"{name}: {value!r} is not an axis (+x, -x, +y, ...)")
        v = np.array(_AXES[value.lower()], dtype=float)
    else:
        v = np.asarray(value, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not n > 0:
        raise ConfigError(f"{name}: zero vector")
    return tuple((v / n).tolist())


def load(profile: str | Path | None = None, *, sets=()) -> Config:
    """Read a profile, apply `sets`, resolve paths and limits, validate."""
    path = profile_path(profile)
    try:
        tree = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    tree = apply_sets(tree, sets)
    overrides = {k.strip(): _parse_value(v.strip())
                 for k, _, v in (str(s).partition("=") for s in sets or ())}

    def sec(name):
        return tree.get(name, {})

    robot = sec("robot")
    urdf = _resolve(robot.get("urdf", "description/elrobot_ours/elrobot.urdf"))
    if not urdf.exists():
        raise ConfigError(f"[robot] urdf: {urdf} does not exist")
    calibration = _resolve(robot.get("calibration", "calibration/calibration.json"))
    limits = urdf_limits(urdf)

    b = sec("bus")
    bus = BusConfig(driver=str(b.get("driver", "st3215")), port=str(b.get("port", "")),
                    hz=float(b.get("hz", 250.0)), slow_every=int(b.get("slow_every", 5)))
    if bus.driver not in ("st3215", "sim"):
        raise ConfigError(f"[bus] driver: {bus.driver!r} is not st3215 or sim")
    if bus.driver == "st3215" and not calibration.exists():
        raise ConfigError(f"[robot] calibration: {calibration} does not exist")

    velocity_scale = float(sec("limits").get("velocity_scale", 0.5))
    if not 0 < velocity_scale <= 1:
        raise ConfigError(f"[limits] velocity_scale {velocity_scale}: must be in (0, 1]")
    gripper_velocity_scale = float(sec("limits").get("gripper_velocity_scale", 1.5))
    if not 0 < gripper_velocity_scale <= 2:
        raise ConfigError(f"[limits] gripper_velocity_scale {gripper_velocity_scale}: must be "
                          f"in (0, 2]")

    e = sec("executor")
    friction = _per_joint(e.get("friction"), "[executor] friction")
    friction = 0.0 if friction is None else np.radians(friction)
    if np.any(np.asarray(friction) < 0):
        raise ConfigError("[executor] friction: must be >= 0")
    executor = ExecutorSettings(
        hz=bus.hz, filter_hz=float(e.get("filter_hz", 15.0)),
        filter_lead=bool(e.get("filter_lead", True)), velocity_scale=velocity_scale,
        gripper_velocity_scale=gripper_velocity_scale,
        squeeze_margin=math.radians(float(e.get("squeeze_margin", 10.0))),
        jaw_free_lead=math.radians(float(e.get("jaw_free_lead", 90.0))),
        jaw_stall_speed=math.radians(float(e.get("jaw_stall_speed", 15.0))),
        jaw_contact_s=float(e.get("jaw_contact_s", 0.08)),
        slow_every=bus.slow_every, friction=friction,
        friction_v=math.radians(float(e.get("friction_v", 2.0))))
    gains = DriverGains(p_coef=_ids(e.get("p_coef"), "[executor] p_coef"),
                        d_coef=_ids(e.get("d_coef"), "[executor] d_coef"),
                        torque_limit=_ids(e.get("torque_limit"), "[executor] torque_limit"),
                        goal_speed_factor=float(e.get("goal_speed_factor", 1.5)),
                        acceleration=int(e.get("acceleration", 0)))

    p = sec("poses")
    home = _per_joint(p.get("home"), "[poses] home")
    exit_ = _per_joint(p.get("exit"), "[poses] exit")
    if home is None or exit_ is None:
        raise ConfigError("[poses] home and exit are required (8 angles each, degrees)")
    poses = Poses(home=np.radians(home), exit=np.radians(exit_),
                  min_s=float(p.get("min_s", 2.0)),
                  velocity_margin=float(p.get("velocity_margin", 0.5)),
                  home_tol=math.radians(float(p.get("home_tol", 5.0))),
                  release_at_exit=bool(p.get("release_at_exit", True)))
    for name, q in (("home", poses.home), ("exit", poses.exit)):
        out = np.flatnonzero((q < limits.lower - 1e-9) | (q > limits.upper + 1e-9))
        if out.size:
            raise ConfigError(
                f"[poses] {name}: joints {[int(i) + 1 for i in out]} are outside the URDF's "
                f"limits ({np.round(np.degrees(limits.lower[out]), 1).tolist()} .. "
                f"{np.round(np.degrees(limits.upper[out]), 1).tolist()} deg)")
    if not 0 < poses.velocity_margin <= 2 / math.pi:
        raise ConfigError(f"[poses] velocity_margin {poses.velocity_margin}: must be in "
                          f"(0, 2/pi]: a cosine ramp peaks at pi/2 times its mean speed")

    k = dict(sec("ik"))
    ik = IKConfig(solver=str(k.pop("solver", "tcp_qp")), options=k)

    t = sec("teleop")
    d = TeleopConfig()
    teleop = TeleopConfig(
        control_hz=float(t.get("control_hz", d.control_hz)),
        lead_s=float(t.get("lead_s", d.lead_s)), port=int(t.get("port", d.port)),
        stale_s=float(t.get("stale_s", d.stale_s)), scale=float(t.get("scale", d.scale)),
        gripper_rate=math.radians(float(t.get("gripper_rate", 360.0))),
        max_step_m=float(t.get("max_step_m", d.max_step_m)),
        max_step=math.radians(float(t.get("max_step_deg", 30.0))),
        phone_forward=_axis(t.get("phone_forward", "-x"), "[teleop] phone_forward"),
        base_forward=_axis(t.get("base_forward", d.base_forward), "[teleop] base_forward"),
        base_up=_axis(t.get("base_up", d.base_up), "[teleop] base_up"),
        workspace_min=(tuple(float(v) for v in t["workspace_min"])
                       if t.get("workspace_min") else None),
        workspace_max=(tuple(float(v) for v in t["workspace_max"])
                       if t.get("workspace_max") else None))
    if abs(float(np.dot(teleop.base_forward, teleop.base_up))) > 1e-6:
        raise ConfigError("[teleop] base_forward and base_up must be perpendicular")
    if (teleop.workspace_min is None) != (teleop.workspace_max is None):
        raise ConfigError("[teleop] workspace_min and workspace_max: give both, or neither")
    if teleop.workspace_min is not None and not all(
            lo < hi for lo, hi in zip(teleop.workspace_min, teleop.workspace_max)):
        raise ConfigError("[teleop] workspace_min must be below workspace_max on every axis")
    if not 0 < teleop.lead_s < 0.5:
        raise ConfigError(f"[teleop] lead_s {teleop.lead_s}: must be in (0, 0.5) s")

    g, gd = sec("safety"), GuardSettings()
    guard = GuardSettings(**{k: type(getattr(gd, k))(g[k]) for k in gd.__dataclass_fields__
                             if k in g})
    unknown = set(g) - set(gd.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"[safety]: unknown keys {sorted(unknown)}")

    r, rd = sec("record"), RecordConfig()
    record = RecordConfig(dir=_resolve(str(r.get("dir", "logs/records"))),
                          jpeg_quality=int(r.get("jpeg_quality", rd.jpeg_quality)))
    if not 1 <= record.jpeg_quality <= 100:
        raise ConfigError(f"[record] jpeg_quality {record.jpeg_quality}: must be in 1..100")
    astra = _astra(sec("astra"))

    db = sec("dashboard")
    dashboard = DashboardConfig(host=str(db.get("host", "127.0.0.1")),
                                port=int(db.get("port", 8080)))

    return Config(profile=str(path), urdf=urdf, calibration=calibration, limits=limits,
                  velocity_scale=velocity_scale, bus=bus, executor=executor, gains=gains,
                  poses=poses, ik=ik, teleop=teleop, dashboard=dashboard, guard=guard,
                  cameras=_cameras(sec("cameras")), record=record, astra=astra,
                  overrides=overrides)


def _astra(t: dict) -> AstraConfig:
    d = AstraConfig()
    unknown = set(t) - set(d.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"[astra]: unknown keys {sorted(unknown)}")
    kw = {}
    for name in d.__dataclass_fields__:
        if name not in t:
            continue
        if name == "sessions_dir":
            kw[name] = _resolve(str(t[name]))
        elif name == "anchor_camera":
            kw[name] = str(t[name])
        elif name == "cameras":
            kw[name] = tuple(str(v) for v in t[name])
        else:
            kw[name] = type(getattr(d, name))(t[name])
    a = AstraConfig(**kw)
    if not a.gripper_open_mm > a.gripper_closed_mm >= 0:
        raise ConfigError("[astra] gripper_open_mm must be above gripper_closed_mm (>= 0)")
    positive = ("max_speed_m_s", "max_rot_deg_s", "max_accel_m_s2", "max_rot_accel_deg_s2",
                "max_gripper_mm_s", "min_segment_s")
    for name in positive:
        if not getattr(a, name) > 0:
            raise ConfigError(f"[astra] {name}: must be > 0")
    if not 0 < a.joint_speed_margin <= 1:
        raise ConfigError("[astra] joint_speed_margin: must be in (0, 1]")
    if not 0 < a.max_rotation_deg < 180:
        raise ConfigError("[astra] max_rotation_deg: must be in (0, 180)")
    if not a.settle_s >= 0:
        raise ConfigError("[astra] settle_s: must be >= 0")
    return a


__all__ = [
    "REPO",
    "AstraConfig",
    "BusConfig",
    "CameraConfig",
    "Config",
    "DashboardConfig",
    "DriverGains",
    "IKConfig",
    "Poses",
    "RecordConfig",
    "TeleopConfig",
    "apply_sets",
    "load",
    "profile_path",
]

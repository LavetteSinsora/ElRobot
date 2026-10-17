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
    workspace_min: tuple = (-0.30, 0.08, 0.02)
    workspace_max: tuple = (0.30, 0.40, 0.35)


@dataclass(frozen=True)
class DashboardConfig:
    host: str = "127.0.0.1"           # 0.0.0.0 to open it from another device on the LAN
    port: int = 8080


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
        ] + [f"--set       {k} = {v!r}" for k, v in self.overrides.items()])


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
    a typo must not silently do nothing."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in tree.items()}
    for item in sets or ():
        key, eq, text = str(item).partition("=")
        if not eq or "." not in key:
            raise ConfigError(f"--set {item!r}: expected section.key=value")
        section, _, name = key.strip().partition(".")
        if section not in out or not isinstance(out[section], dict):
            raise ConfigError(f"--set {item!r}: no section [{section}] (have {sorted(out)})")
        if name not in out[section] and not (section == "ik"):
            raise ConfigError(f"--set {item!r}: [{section}] has no key {name!r} "
                              f"(have {sorted(out[section])})")
        out[section][name] = _parse_value(text.strip())
    return out


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
        squeeze_margin=math.radians(float(e.get("squeeze_margin", 5.0))),
        jaw_free_lead=math.radians(float(e.get("jaw_free_lead", 45.0))),
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
        workspace_min=tuple(float(v) for v in t.get("workspace_min", d.workspace_min)),
        workspace_max=tuple(float(v) for v in t.get("workspace_max", d.workspace_max)))
    if abs(float(np.dot(teleop.base_forward, teleop.base_up))) > 1e-6:
        raise ConfigError("[teleop] base_forward and base_up must be perpendicular")
    if not all(lo < hi for lo, hi in zip(teleop.workspace_min, teleop.workspace_max)):
        raise ConfigError("[teleop] workspace_min must be below workspace_max on every axis")
    if not 0 < teleop.lead_s < 0.5:
        raise ConfigError(f"[teleop] lead_s {teleop.lead_s}: must be in (0, 0.5) s")

    g, gd = sec("safety"), GuardSettings()
    guard = GuardSettings(**{k: type(getattr(gd, k))(g[k]) for k in gd.__dataclass_fields__
                             if k in g})
    unknown = set(g) - set(gd.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"[safety]: unknown keys {sorted(unknown)}")

    db = sec("dashboard")
    dashboard = DashboardConfig(host=str(db.get("host", "127.0.0.1")),
                                port=int(db.get("port", 8080)))

    return Config(profile=str(path), urdf=urdf, calibration=calibration, limits=limits,
                  velocity_scale=velocity_scale, bus=bus, executor=executor, gains=gains,
                  poses=poses, ik=ik, teleop=teleop, dashboard=dashboard, guard=guard,
                  overrides=overrides)


__all__ = [
    "REPO",
    "BusConfig",
    "Config",
    "DashboardConfig",
    "DriverGains",
    "IKConfig",
    "Poses",
    "TeleopConfig",
    "apply_sets",
    "load",
    "profile_path",
]

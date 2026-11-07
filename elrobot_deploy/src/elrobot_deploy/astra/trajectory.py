"""The trajectory law: Astra's waypoints in, a timed path on the control grid out.

    waypoints, speed, errors = parse_action(args, limits)       # send_action's input, checked
    plan = build(ref, waypoints, limits, dt, speed=speed, stretch=joint_check)

**Relative, one frame.** Every waypoint is an offset from the arm's REFERENCE pose (the last
commanded pose), in that pose's TCP frame: `p = p_ref + R_ref dp`, `R = R_ref R_d`, `R_d` from
the waypoint's roll / pitch / yaw (`frames.mat_from_rpy`). Offsets are all from the same
reference, never chained. A waypoint's gripper width is absolute (mm).

**The path** runs through the reference (knot 0) and every waypoint, at rest at both ends and
without stopping at a waypoint: a monotone cubic (PCHIP, Fritsch-Carlson slopes) per
coordinate, on position (base frame), on the rotation vector of each waypoint's rotation from
the reference (`R(t) = R_ref exp(r(t))`), and on the gripper width. Between two knots each
coordinate stays between their values: no overshoot, no dip below the lower waypoint.

**The timing is ours** (Astra never gives a duration): each segment gets the shortest
duration on the grid that keeps the sampled peaks inside `Limits` (linear and angular speed
and acceleration, gripper speed, `min_segment_s`), and then, through `stretch`, inside the
joint speeds the IK needs (P12 solves the sampled path and says which segments ask too much
of which joint). Knots land on whole grid ticks, so the command at a waypoint's deadline is
exactly the composed waypoint. `speed = "slow"` doubles every segment.

Numpy only: no MuJoCo, no arena. P12 (`runtime/processes/p12_astra.py`) runs it; the harness
uses `parse_action` to refuse a malformed action before anything moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from elrobot_deploy.astra.frames import (
    mat_from_rotvec,
    mat_from_rpy,
    rotation_angle,
    rotvec_from_mat,
)

SPEEDS = ("normal", "slow")
#: `slow` stretches every segment by this factor.
SLOW_FACTOR = 2.0
#: Check the peaks this many times finer than the control grid (a peak between two samples).
CHECK_SUBSTEPS = 4
#: Stretch a segment a little past the factor that just fits, so the loop converges.
STRETCH_SLACK = 1.02
MAX_ITERATIONS = 60

WAYPOINT_FIELDS = ("xyz", "rpy", "gripper_mm")


@dataclass(frozen=True)
class Limits:
    """The caps the timing keeps (SI: m, rad, s; gripper mm)."""

    v: float = 0.10                     # m/s
    w: float = math.radians(45.0)       # rad/s
    a: float = 0.5                      # m/s^2
    alpha: float = math.radians(180.0)  # rad/s^2
    g: float = 60.0                     # mm/s
    min_segment_s: float = 0.1
    max_rotation: float = math.radians(170.0)
    open_mm: float = 55.0
    closed_mm: float = 0.0

    @classmethod
    def from_config(cls, cfg) -> Limits:
        a = cfg.astra
        return cls(v=a.max_speed_m_s, w=math.radians(a.max_rot_deg_s), a=a.max_accel_m_s2,
                   alpha=math.radians(a.max_rot_accel_deg_s2), g=a.max_gripper_mm_s,
                   min_segment_s=a.min_segment_s, max_rotation=math.radians(a.max_rotation_deg),
                   open_mm=a.gripper_open_mm, closed_mm=a.gripper_closed_mm)


@dataclass(frozen=True)
class Waypoint:
    """One of Astra's waypoints, as sent: an offset in the reference TCP frame."""

    xyz: np.ndarray                     # (3,) m
    rpy: np.ndarray                     # (3,) deg
    gripper_mm: float


@dataclass(frozen=True)
class Reference:
    """What the offsets are from: a pose in the runtime's base frame, and a gripper width."""

    p: np.ndarray                       # (3,) m, base_link
    R: np.ndarray                       # (3, 3) TCP axes in base_link
    gripper_mm: float


@dataclass
class Plan:
    """A timed path, sampled on the control grid. Sample `i` is due at `i * dt` after the
    first; sample 0 is the reference, at rest, and the last is the last waypoint, at rest."""

    dt: float
    p: np.ndarray                       # (N, 3) base_link, m
    r: np.ndarray                       # (N, 3) rotation vector from the reference, rad
    R: np.ndarray                       # (N, 3, 3) base_link
    g: np.ndarray                       # (N,) mm
    knots: list                         # sample index of the reference (0) and each waypoint
    seg_s: np.ndarray                   # (K,) each segment's duration
    ref: Reference
    targets: list = field(default_factory=list)     # the composed waypoints: (p, R, g)

    @property
    def n(self) -> int:
        return len(self.g)

    @property
    def duration(self) -> float:
        return float(self.seg_s.sum())

    def segment_of(self) -> np.ndarray:
        """(N,) which segment each sample belongs to (the knot it heads for, minus one)."""
        seg = np.searchsorted(np.asarray(self.knots), np.arange(self.n), side="left") - 1
        return np.clip(seg, 0, len(self.seg_s) - 1)


# --- the input ------------------------------------------------------------------------------


def _vec3(value, where: str, errors: list) -> np.ndarray | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
        errors.append(f"{where}: expected a list of 3 numbers, got {value!r}")
        return None
    out = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        errors.append(f"{where}: expected finite numbers, got {value!r}")
        return None
    return out


def parse_action(args, limits: Limits) -> tuple[list[Waypoint], str, list[str]]:
    """`send_action`'s arguments -> (waypoints, speed, errors). Every problem is listed, each
    naming the waypoint (from 1) and the field; with any error nothing may move."""
    errors: list[str] = []
    if not isinstance(args, dict):
        return [], "normal", [(f"expected an object with `waypoints` (and `speed`), got "
                               f"{type(args).__name__}")]
    unknown = sorted(set(args) - {"waypoints", "speed"})
    if unknown:
        errors.append(f"unknown argument(s) {unknown}: only `waypoints` and `speed`")
    speed = args.get("speed", "normal")
    if speed not in SPEEDS:
        errors.append(f"speed: {speed!r} is not one of {list(SPEEDS)}")
        speed = "normal"
    raw = args.get("waypoints")
    if not isinstance(raw, list) or not raw:
        errors.append("waypoints: expected a non-empty list of {xyz, rpy, gripper_mm}")
        return [], speed, errors
    out = []
    lo, hi = sorted((limits.closed_mm, limits.open_mm))
    for i, w in enumerate(raw, start=1):
        where = f"waypoint {i}"
        if not isinstance(w, dict):
            errors.append(f"{where}: expected an object {{xyz, rpy, gripper_mm}}, got {w!r}")
            continue
        extra = sorted(set(w) - set(WAYPOINT_FIELDS))
        if extra:
            errors.append(f"{where}: unknown field(s) {extra}: only xyz, rpy, gripper_mm")
        missing = [k for k in WAYPOINT_FIELDS if k not in w]
        if missing:
            errors.append(f"{where}: missing {', '.join(missing)} (all three are required)")
            continue
        xyz = _vec3(w["xyz"], f"{where}.xyz", errors)
        rpy = _vec3(w["rpy"], f"{where}.rpy", errors)
        g = w["gripper_mm"]
        if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g):
            errors.append(f"{where}.gripper_mm: expected a number, got {g!r}")
            g = None
        elif not lo <= g <= hi:
            errors.append(f"{where}.gripper_mm: {g:g} is outside {lo:g} ... {hi:g} mm")
            g = None
        if rpy is not None:
            angle = rotation_angle(mat_from_rpy(rpy))
            if angle >= limits.max_rotation - 1e-9:
                errors.append(f"{where}.rpy: rotates {math.degrees(angle):.0f} deg from the "
                              f"reference; the limit is < {math.degrees(limits.max_rotation):.0f}"
                              f" deg (turn further in two actions)")
                rpy = None
        if xyz is not None and rpy is not None and g is not None:
            out.append(Waypoint(xyz, rpy, float(g)))
    return out, speed, errors


def compose(ref: Reference, waypoints) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Each waypoint as an absolute pose (base_link) and width: `p_ref + R_ref dp`,
    `R_ref R_d`."""
    return [(ref.p + ref.R @ w.xyz, ref.R @ mat_from_rpy(w.rpy), float(w.gripper_mm))
            for w in waypoints]


# --- PCHIP ----------------------------------------------------------------------------------


def pchip_slopes(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson slopes (scipy's `PchipInterpolator` rule) at every knot, per column,
    with ZERO slope at both ends (the path starts and ends at rest)."""
    h = np.diff(t)[:, None]
    d = np.diff(y, axis=0) / h
    m = np.zeros_like(y)
    if len(t) > 2:
        h0, h1 = h[:-1], h[1:]
        d0, d1 = d[:-1], d[1:]
        w1, w2 = 2 * h1 + h0, h1 + 2 * h0
        same = (d0 * d1) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            hm = (w1 + w2) / (w1 / d0 + w2 / d1)
        m[1:-1] = np.where(same, hm, 0.0)
    return m


def hermite(t_knots, y, m, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cubic Hermite through (t_knots, y) with slopes `m`, at times `t`: value, first and
    second derivative, each (len(t), columns)."""
    t = np.asarray(t, dtype=np.float64)
    j = np.clip(np.searchsorted(t_knots, t, side="right") - 1, 0, len(t_knots) - 2)
    h = (t_knots[j + 1] - t_knots[j])[:, None]
    s = ((t - t_knots[j])[:, None] / h).clip(0.0, 1.0)
    y0, y1, m0, m1 = y[j], y[j + 1], m[j] * h, m[j + 1] * h
    s2, s3 = s * s, s * s * s
    val = ((2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * m0
           + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * m1)
    d1 = ((6 * s2 - 6 * s) * y0 + (3 * s2 - 4 * s + 1) * m0
          + (-6 * s2 + 6 * s) * y1 + (3 * s2 - 2 * s) * m1) / h
    d2 = ((12 * s - 6) * y0 + (6 * s - 4) * m0 + (-12 * s + 6) * y1 + (6 * s - 2) * m1) / (h * h)
    return val, d1, d2


# --- the timing -----------------------------------------------------------------------------


def _initial(Y: np.ndarray, lim: Limits) -> np.ndarray:
    """A first guess per segment: a lone cubic with rest ends peaks at 1.5x its mean speed and
    6 d / T^2 in acceleration."""
    dY = np.abs(np.diff(Y, axis=0))
    dp = np.linalg.norm(np.diff(Y[:, :3], axis=0), axis=1)
    dr = np.linalg.norm(np.diff(Y[:, 3:6], axis=0), axis=1)
    dg = dY[:, 6]
    return np.maximum.reduce([1.5 * dp / lim.v, 1.5 * dr / lim.w, 1.5 * dg / lim.g,
                              np.sqrt(6 * dp / lim.a), np.sqrt(6 * dr / lim.alpha),
                              np.full(len(dp), lim.min_segment_s)])


def _on_grid(T: np.ndarray, dt: float, lim: Limits) -> np.ndarray:
    T = np.maximum(T, lim.min_segment_s)
    return np.maximum(np.ceil(T / dt - 1e-9), 1) * dt


def _over(Y: np.ndarray, T: np.ndarray, dt: float, lim: Limits) -> np.ndarray:
    """Per segment, by how much its sampled peaks exceed the limits (time factor; <= 1 fits)."""
    t_knots = np.r_[0.0, np.cumsum(T)]
    m = pchip_slopes(t_knots, Y)
    out = np.ones(len(T))
    for j in range(len(T)):
        n = max(round(T[j] / dt) * CHECK_SUBSTEPS, 2)
        t = np.linspace(t_knots[j], t_knots[j + 1], n + 1)
        _, d1, d2 = hermite(t_knots, Y, m, t)
        ratios = [np.linalg.norm(d1[:, :3], axis=1).max() / lim.v,
                  np.linalg.norm(d1[:, 3:6], axis=1).max() / lim.w,
                  np.abs(d1[:, 6]).max() / lim.g,
                  math.sqrt(np.linalg.norm(d2[:, :3], axis=1).max() / lim.a),
                  math.sqrt(np.linalg.norm(d2[:, 3:6], axis=1).max() / lim.alpha)]
        out[j] = max(ratios)
    return out


def _sample(Y: np.ndarray, T: np.ndarray, dt: float, ref: Reference, targets) -> Plan:
    steps = np.rint(T / dt).astype(int)
    knots = [0] + np.cumsum(steps).tolist()
    t_knots = np.asarray(knots, dtype=np.float64) * dt
    t = np.arange(knots[-1] + 1) * dt
    val, _, _ = hermite(t_knots, Y, pchip_slopes(t_knots, Y), t)
    # The knots exactly (the cubic passes through them; this removes the rounding).
    val[knots] = Y
    r = val[:, 3:6]
    R = np.einsum("ij,njk->nik", ref.R, np.stack([mat_from_rotvec(v) for v in r]))
    return Plan(dt=dt, p=val[:, :3].copy(), r=r.copy(), R=R, g=val[:, 6].copy(), knots=knots,
                seg_s=steps * dt, ref=ref, targets=list(targets))


def build(ref: Reference, waypoints, limits: Limits, dt: float, *, speed: str = "normal",
          stretch=None) -> Plan:
    """The timed path through `waypoints` from `ref`.

    `stretch(plan) -> (K,)`, optional: per segment, the factor its duration must grow by for
    something this module cannot see (the joint speeds; <= 1 = fine). Called on the timed plan
    until it asks for nothing more (a few rounds at most).
    """
    if not waypoints:
        raise ValueError("no waypoints")
    targets = compose(ref, waypoints)
    Y = np.zeros((len(targets) + 1, 7))
    Y[:, :3] = np.vstack([ref.p] + [p for p, _, _ in targets])
    Y[1:, 3:6] = np.vstack([rotvec_from_mat(mat_from_rpy(w.rpy)) for w in waypoints])
    Y[:, 6] = [ref.gripper_mm] + [g for _, _, g in targets]

    T = _on_grid(_initial(Y, limits), dt, limits)
    rounds = 0
    while True:
        for _ in range(MAX_ITERATIONS):
            over = _over(Y, T, dt, limits)
            if np.all(over <= 1 + 1e-6):
                break
            T = _on_grid(np.where(over > 1 + 1e-6, T * over * STRETCH_SLACK, T), dt, limits)
        plan = _sample(Y, T * (SLOW_FACTOR if speed == "slow" else 1.0), dt, ref, targets)
        if stretch is None:
            return plan
        need = np.asarray(stretch(plan), dtype=np.float64).reshape(len(T))
        if not np.all(np.isfinite(need)) or np.any(need <= 0):
            raise ValueError("joint-speed check returned invalid timing factors; action refused")
        if np.all(need <= 1 + 1e-6):
            return plan
        if rounds >= 5:
            raise ValueError("cannot satisfy joint speed limits after 5 retiming rounds; "
                             "action refused (try smaller waypoints or a different posture)")
        T = _on_grid(np.where(need > 1 + 1e-6, T * need * STRETCH_SLACK, T), dt, limits)
        rounds += 1


__all__ = [
    "SPEEDS",
    "Limits",
    "Plan",
    "Reference",
    "Waypoint",
    "build",
    "compose",
    "hermite",
    "parse_action",
    "pchip_slopes",
]

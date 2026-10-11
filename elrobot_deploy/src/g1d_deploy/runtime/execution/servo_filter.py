"""How P7 shapes its 500 Hz command: a smoother, a feedforward, and the line they read.

Runs inside P7. Design, and how to write a smoother or a feedforward:
`docs/infra_logic/500hz_execution.md`.

P7 walks the 30 Hz target line (`Segment`) and, with no shaping, publishes that
line and its slope (`LINEAR`). Shaping adds two independent pieces, each
behind a small interface so another one can be dropped in:

  Smoother     `step(reference, h) -> (q, v, a)`
               turns the target line into a smooth command. Shipped: `CritDamped`.
  Feedforward  `tau(v, a) -> (17,) N m`
               the torque the commanded motion needs, added to gravity comp.
               Shipped: `ModelFeedforward` (inertia + friction).

`build(servo, hw)` turns the config's `ServoShaping` and the robot file's
`[servo_model]` into a `Shaping` for P7; a deploy file's own
`[control.servo] smoother` / `feedforward` replaces either piece. P7 does not
change.

**The look-ahead.** A critically damped filter trails a ramp by `2/w`. Reading
the target line that far AHEAD cancels it, and `line_ahead` is that read. It
never reads past the last target P6 has written: when the line runs out (a new
chunk's first targets are ~66 ms out when they appear, less than the 80 ms lead
at 4 Hz), it holds the line's end, so the lead shrinks for those few periods
instead of skipping targets or guessing.

Everything here is pure numpy and allocation-light: it runs inside P7's 2 ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.execution.servo_segment import EPS_SPAN


# --- the interfaces ---------------------------------------------------------------


class Smoother(Protocol):
    def step(self, reference: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance `h` seconds toward `reference`; return command `(q, v, a)`."""
        ...


class Feedforward(Protocol):
    def tau(self, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """(17,) commanded velocity and acceleration -> (17,) N m."""
        ...


@dataclass(frozen=True)
class Shaping:
    """Everything P7 needs to shape its command. `LINEAR` is the unshaped servo."""

    smoother: Smoother | None = None
    lead_s: float = 0.0            # how far ahead the smoother reads the line
    feedforward: Feedforward | None = None
    velocity_ff: float = 1.0       # scale on the published dq


LINEAR = Shaping()


# --- the smoother -----------------------------------------------------------------


class CritDamped:
    """`y'' = w^2 (r - y) - 2 w y'` per joint, `w = 2 pi fc`.

    Critically damped: the fastest response that never overshoots, so the
    command cannot swing past a target. Its lag behind a ramp is `2/w`, which is
    what `lead_s` reads ahead by.

    Advanced by the exact solution for a reference held over `h` — with
    `e = y - r`, `c = v + w e`:  e(h) = (e + c h) exp(-w h),
    v(h) = (v - w c h) exp(-w h) — so any period, however jittery, is stable.
    """

    def __init__(self, fc: float, q0: np.ndarray):
        self.w = 2.0 * np.pi * float(fc)
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.v = np.zeros_like(self.q)

    @property
    def lead_s(self) -> float:
        return 2.0 / self.w

    def step(self, reference, h):
        w = self.w
        e = self.q - reference
        c = self.v + w * e
        decay = np.exp(-w * h)
        e = (e + c * h) * decay
        self.v = (self.v - w * c * h) * decay
        self.q = reference + e
        a = -w * w * e - 2.0 * w * self.v
        return self.q, self.v, a


# --- the feedforward --------------------------------------------------------------


class ModelFeedforward:
    """`scale * (J a + b v + Fc tanh(v / eps))` on the 14 arm joints, bounded.

    The model is the robot's (the robot file's `[servo_model]`, identified
    on the reference robot). The waist gets none: it was not identified, and the bow rests on a
    mechanical stop where this model means nothing.
    """

    def __init__(self, model, scale: float):
        self.J = scale * np.asarray(model.inertia, dtype=np.float64)
        self.b = scale * np.asarray(model.viscous, dtype=np.float64)
        self.Fc = scale * np.maximum(np.asarray(model.coulomb, dtype=np.float64), 0.0)
        self.eps = float(model.eps_v)
        self.tau_max = float(model.tau_max)

    def tau(self, v, a):
        arm = schema.ARM_DOF
        t = self.J * a[:arm] + self.b * v[:arm] + self.Fc * np.tanh(v[:arm] / self.eps)
        out = np.zeros(schema.BODY_DOF)
        out[:arm] = np.clip(t, -self.tau_max, self.tau_max)
        return out


# --- the reference ----------------------------------------------------------------


def line_ahead(seg, ring, t_now: float, lead: float) -> np.ndarray:
    """The target line P7 is walking, read `lead` seconds ahead of `t_now`.

    The line starts where `seg` stands — the segment in flight, or the held
    command during a stall — and continues through every written target after
    it, in deadline order. If it ends before `t_now + lead`, its end is held:
    the smoother then simply has less look-ahead for as long as P6 is behind.
    """
    t_ref = t_now + lead
    if not seg.stalled and t_ref <= seg.t_end:
        return seg.at(seg.u_raw(t_ref))
    if seg.stalled:
        t0, q0 = seg.t_start, seg.x_start
    else:
        t0, q0 = seg.t_end, seg.x_end
    for target in ring.line_after(t0, t_ref, seg.world):
        if target.deadline >= t_ref:
            u = (t_ref - t0) / max(target.deadline - t0, EPS_SPAN)
            return q0 + u * (target.payload - q0)
        t0, q0 = target.deadline, target.payload
    return q0


# --- the factory ------------------------------------------------------------------


def load(spec: str):
    """`"package.module:factory"` or `"path/file.py:factory"` -> the factory. `ValueError` if it does not
    import or is not callable. Imported in P7's process, i.e. g1d_deploy's own
    Python, never the policy's."""
    from g1d_deploy.policy_contract.loading import load_attr
    try:
        factory = load_attr(spec)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"{spec!r} cannot be imported: {exc}") from exc
    if not callable(factory):
        raise ValueError(f"{spec!r} is not callable")
    return factory


def build(servo, hw, q0: np.ndarray, *, smoother: str = "",
          feedforward: str = "") -> Shaping:
    """`ServoShaping` (the task's choice) + `hw.servo_model` (the robot's) -> P7's `Shaping`.

    `smoother` / `feedforward` name the deploy file's own modules
    (`[control.servo] smoother = "pkg.mod:factory"`), which REPLACE the
    shipped `CritDamped` / `ModelFeedforward` — under the same switches, so
    `filter_hz = 0` and `torque_ff = 0` still mean "off" and the dashboard's
    straight-line servo is still one setting away:

        smoother factory(servo, hw, q0) -> object with
            step(reference (17,), h s) -> (q, v, a), each (17,)
            lead_s (optional): seconds to read the target line ahead, used
                when `servo.filter_lead`
        feedforward factory(servo, hw) -> object with
            tau(v (17,), a (17,)) -> (17,) N m, added to gravity comp

    `servo` is the rollout's `ServoShaping`, so a factory may take its
    bandwidth from `filter_hz` and its scale from `torque_ff`. Both run inside
    P7's 2 ms period at 500 Hz, including the homing move and the exit ramp.
    """
    if servo.filter_hz > 0:
        sm = (load(smoother)(servo, hw, q0) if smoother
              else CritDamped(servo.filter_hz, q0))
    else:
        sm = None
    lead = float(getattr(sm, "lead_s", 0.0)) if sm is not None and servo.filter_lead else 0.0
    if servo.torque_ff > 0:
        ff = (load(feedforward)(servo, hw) if feedforward
              else ModelFeedforward(hw.servo_model, servo.torque_ff))
    else:
        ff = None
    return Shaping(smoother=sm, lead_s=lead, feedforward=ff,
                   velocity_ff=float(servo.velocity_ff))


__all__ = ["CritDamped", "Feedforward", "LINEAR", "ModelFeedforward", "Shaping",
           "Smoother", "build", "line_ahead", "load"]

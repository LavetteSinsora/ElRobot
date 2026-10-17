"""How the bus process shapes its 250 Hz command: a smoother, and the target line it reads.

The bus process walks the 60 Hz target line (`Segment`) and, with no shaping, publishes that
line. Shaping is one piece behind a small interface, so another can be dropped in:

  Smoother     `step(reference, h) -> (q, v, a)`
               turns the target line into a smooth command. Shipped: `CritDamped`.

**The look-ahead.** A critically damped filter trails a ramp by `2/w`. Reading the target line
that far AHEAD cancels it, and `line_ahead` is that read. It never reads past the last target
written: when the line runs out it holds the line's end, so the lead shrinks for those few
periods instead of skipping targets or guessing.

Everything here is pure numpy and allocation-light: it runs inside the bus process's 4 ms.
(g1d's version also had a torque feedforward, which a position-only servo has no input for:
`legacy_g1d/src/elrobot_deploy/runtime/execution/servo_filter.py`.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from elrobot_deploy.runtime.execution.servo_segment import EPS_SPAN

# --- the interface ----------------------------------------------------------------


class Smoother(Protocol):
    def step(self, reference: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance `h` seconds toward `reference`; return command `(q, v, a)`."""
        ...


@dataclass(frozen=True)
class Shaping:
    """Everything the executor needs to shape its command. `LINEAR` is the unshaped line."""

    smoother: Smoother | None = None
    lead_s: float = 0.0            # how far ahead the smoother reads the line


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


# --- the reference ----------------------------------------------------------------


def line_ahead(seg, ring, t_now: float, lead: float) -> np.ndarray:
    """The target line the executor is walking, read `lead` seconds ahead of `t_now`.

    The line starts where `seg` stands — the segment in flight, or the held
    command during a stall — and continues through every written target after
    it, in deadline order. If it ends before `t_now + lead`, its end is held:
    the smoother then simply has less look-ahead for as long as the writer is behind.
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


__all__ = ["LINEAR", "CritDamped", "Shaping", "Smoother", "line_ahead"]

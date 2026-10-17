"""One iteration of the ElRobot executor: the target line in, one goal position per joint out.

This is g1d's `BodyServo` (`runtime/processes/p7_body.py`) for a position-only servo. What it
keeps, unchanged and shared:

  * **the walk** (`servo_segment.Segment`): pick the next deadline, interpolate toward it, hold
    the last command frozen when the ring runs dry, abandon a segment the instant
    `control.intervene` changes;
  * **the shaping** (`servo_filter`): a critically damped filter reading the line `lead_s` ahead
    (`line_ahead`), so it smooths without lagging;
  * **the last-defence slew clamp**: no tick moves a joint more than `qdot_max / hz`.

What changes, because the ST3215 takes a goal position and nothing else:

  * **no velocity and no torque channel.** The servo has no input for them; the line's slope and
    the gravity term P7 computed have nowhere to go;
  * **the calibrated position limits** clamp every command (`JointLimits`, from our URDF). P7
    relied on the IK for that; here the gripper is commanded from a button, and a joint
    commanded into its stop is a stall at full current;
  * **the gripper squeeze bound**, the one clamp that reads the measurement: the jaw command may
    lead the MEASURED jaw toward closed by at most `squeeze_margin`. On a position servo the grip
    force is roughly `P x (commanded - measured)`, so this, not the command, sets how hard an
    object is held (PORTING_PLAN.md C2). It is applied here, every tick, against the freshest
    read, because this is the last thing before the wire.

`step()` is pure: no clock, no bus. The process (`runtime/processes/p7_bus.py`) supplies both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from elrobot_deploy.robot.elrobot_joints import GRIPPER, JointLimits
from elrobot_deploy.runtime.execution.servo_filter import (
    CritDamped,
    Shaping,
    line_ahead,
)
from elrobot_deploy.runtime.execution.servo_segment import Phase, Segment


@dataclass(frozen=True)
class ExecutorSettings:
    """The executor's knobs. Defaults are the ones agreed for iPhone teleop (PORTING_PLAN.md §1)."""

    hz: float = 250.0                 # goal writes per second (one sync_write each)
    filter_hz: float = 15.0           # critically damped cutoff; 0 = publish the line unshaped
    filter_lead: bool = True          # read the line 2/w ahead, so the filter adds no lag
    velocity_scale: float = 0.5       # x the URDF's 4.71 rad/s (about the ST3215's no-load speed)
    squeeze_margin: float = math.radians(5.0)   # gripper rad; a first guess, to calibrate on force
    slow_every: int = 5               # speed / load / temperature every Nth tick (50 Hz at 250)

    def shaping(self, q0: np.ndarray) -> Shaping:
        if self.filter_hz <= 0:
            return Shaping()
        sm = CritDamped(self.filter_hz, q0)
        return Shaping(smoother=sm, lead_s=sm.lead_s if self.filter_lead else 0.0)


class BusServo:
    """The walk, the filter and the clamps for the 8 joints. One instance per bus process."""

    def __init__(self, *, ring, control, limits: JointLimits, settings: ExecutorSettings,
                 q0: np.ndarray, t0: float, cursor: bool = True):
        self.ring = ring
        self.control = control
        self.limits = limits
        self.settings = settings
        self.shaping = settings.shaping(np.asarray(q0, dtype=np.float64))

        self.qdot_max = limits.velocity * float(settings.velocity_scale)
        self.slew = self.qdot_max / float(settings.hz)
        self.h_max = 4.0 / float(settings.hz)

        # As in P7: the LINE being walked and what was PUBLISHED, which the slew clamp bounds.
        # Both start at the measured posture, so the first command is "stay where you are".
        self.q_cmd_last = np.asarray(q0, dtype=np.float64).copy()
        self.q_line_last = self.q_cmd_last.copy()
        self.t_last = float(t0)
        self.seg = Segment(ring, control, self.q_line_last, t0, cursor=cursor)
        #: Which branch the last published command came from: "hold" or "inside".
        self.phase = "hold"

    stalled = property(lambda self: self.seg.stalled)

    def step(self, t_now: float, q_meas: np.ndarray | None = None) -> np.ndarray | None:
        """The goal positions to write at `t_now`, or None if this iteration only latched a new
        segment, in which case call again at once (P7's ADVANCED branch: no frame is dropped).

        `q_meas` is the newest measured posture; only the squeeze bound reads it. Without one
        the bound is skipped (the walk and the other clamps still apply).
        """
        phase = self.seg.phase(t_now, self.q_line_last)
        if phase is Phase.ADVANCED:
            return None
        self.phase = "hold" if phase is Phase.HOLD else "inside"

        q_line = self._walk(t_now, phase)
        if self.shaping.smoother is None:
            q = q_line
        else:
            reference = line_ahead(self.seg, self.ring, t_now, self.shaping.lead_s)
            h = min(max(t_now - self.t_last, 0.0), self.h_max)
            q, _, _ = self.shaping.smoother.step(reference, h)
        self.t_last = t_now

        q = np.clip(q, self.q_cmd_last - self.slew, self.q_cmd_last + self.slew)
        if q_meas is not None:
            # + closes the gripper, so the bound is an upper one.
            q[GRIPPER] = min(q[GRIPPER], q_meas[GRIPPER] + self.settings.squeeze_margin)
        q = self.limits.clip(q)
        self.q_cmd_last = q
        return q

    def _walk(self, t_now: float, phase: Phase) -> np.ndarray:
        """The target line at `t_now` (P7's `_walk`, without the slope it no longer needs)."""
        if phase is Phase.HOLD:
            return self.q_line_last
        q = np.clip(self.seg.at(self.seg.u_raw(t_now)), self.q_line_last - self.slew,
                    self.q_line_last + self.slew)
        self.q_line_last = q
        return q


__all__ = ["BusServo", "ExecutorSettings"]

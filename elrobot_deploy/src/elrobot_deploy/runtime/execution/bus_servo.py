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
  * **the gripper squeeze bound**, the one clamp that reads the measurement (`JawBound`): once
    the jaw is ON something, its command may lead the MEASURED jaw toward closed by at most
    `squeeze_margin`. On a position servo the grip force is roughly `P x (commanded - measured)`,
    so this, not the command, sets how hard an object is held (PORTING_PLAN.md C2). In free air
    the lead may be up to `jaw_free_lead`, so the servo closes at full effort: a 5 deg lead in
    free air made the jaw crawl (session 2 on the arm). Contact is the jaw stalling (closing
    slower than `jaw_stall_speed` for `jaw_contact_s`) while told to close. Applied here, every
    tick, against the freshest read, because this is the last thing before the wire;
  * **friction compensation** (off by default, `friction`): the arm joints' goal leads the line
    by `friction * tanh(v / friction_v)` in the direction of motion. A P-only servo makes
    torque from position error, so it stalls until the error beats static friction and then
    lurches (measured on joint 6: speed swinging +-40 %, stalls up to 60 ms, a 0.5-0.8 deg dead
    zone at each reversal). Leading by about that dead zone supplies the error up front; tanh
    makes it fade through zero at a reversal instead of flipping.

`step()` is pure: no clock, no bus. The process (`runtime/processes/p7_bus.py`) supplies both.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from elrobot_deploy.robot.elrobot_joints import GRIPPER, JointLimits
from elrobot_deploy.runtime.execution.servo_filter import (
    CritDamped,
    Shaping,
    line_ahead,
)
from elrobot_deploy.runtime.execution.servo_segment import Phase, Segment


class JawBound:
    """How far the jaw command may lead the measured jaw toward closed (+ closes), and whether the
    jaw is on something. `limit(t, meas, wanted)` -> the largest command allowed now."""

    WINDOW_S = 0.04

    def __init__(self, margin: float, free_lead: float, stall_speed: float, contact_s: float):
        self.margin, self.free_lead = float(margin), float(free_lead)
        self.stall_speed, self.contact_s = float(stall_speed), float(contact_s)
        self.contact = False
        self._stall_since: float | None = None
        self._hist: deque = deque()

    def limit(self, t: float, meas: float, wanted: float) -> float:
        self._hist.append((t, meas))
        while len(self._hist) > 2 and t - self._hist[1][0] >= self.WINDOW_S:
            self._hist.popleft()
        t0, m0 = self._hist[0]
        speed = (meas - m0) / (t - t0) if t > t0 else 0.0        # + = closing
        pushing = wanted - meas > self.margin
        if self.contact:
            # Off the object: the operator opens, or the jaw moves on (it slipped, or gave).
            if not pushing or speed > 2.0 * self.stall_speed:
                self.contact = False
                self._stall_since = None
        elif pushing and speed < self.stall_speed:
            if self._stall_since is None:
                self._stall_since = t
            if t - self._stall_since >= self.contact_s:
                self.contact = True
        else:
            self._stall_since = None
        return meas + (self.margin if self.contact else self.free_lead)


@dataclass(frozen=True)
class ExecutorSettings:
    """The executor's knobs. Defaults are the ones agreed for iPhone teleop (PORTING_PLAN.md §1)."""

    hz: float = 250.0                 # goal writes per second (one sync_write each)
    filter_hz: float = 15.0           # critically damped cutoff; 0 = publish the line unshaped
    filter_lead: bool = True          # read the line 2/w ahead, so the filter adds no lag
    velocity_scale: float = 0.5       # x the URDF's 4.71 rad/s (about the ST3215's no-load speed)
    gripper_velocity_scale: float = 1.0   # the gripper's own: a light jaw may run at full speed
    squeeze_margin: float = math.radians(10.0)  # gripper rad; a first guess, to calibrate on force
    jaw_free_lead: float = math.radians(90.0)   # the lead allowed in free air (no contact)
    jaw_stall_speed: float = math.radians(15.0)  # rad/s: closing slower than this ...
    jaw_contact_s: float = 0.08                 # ... this long while told to close = contact
    slow_every: int = 5               # speed / load / temperature every Nth tick (50 Hz at 250)
    #: rad the goal leads in the direction of motion: one value for every arm joint, or (8,)
    #: per joint (the gripper's is ignored: the squeeze bound owns the jaw). 0 = off.
    friction: float | np.ndarray = 0.0
    friction_v: float = math.radians(2.0)   # rad/s over which that lead fades in (tanh scale)

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

        scale = np.full(len(limits.velocity), float(settings.velocity_scale))
        scale[GRIPPER] = float(settings.gripper_velocity_scale)
        self.qdot_max = limits.velocity * scale
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
        self._zero = np.zeros_like(self.q_cmd_last)
        # Friction compensation, per joint, never on the gripper.
        self._friction = np.broadcast_to(np.asarray(settings.friction, np.float64),
                                         self.q_cmd_last.shape).copy()
        self._friction[GRIPPER] = 0.0
        self._friction_on = bool(np.any(self._friction > 0.0))
        self.jaw = JawBound(settings.squeeze_margin, settings.jaw_free_lead,
                            settings.jaw_stall_speed, settings.jaw_contact_s)

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
            u = self.seg.u_raw(t_now)
            v = (self.seg.delta / self.seg.span if phase is Phase.INSIDE and 0.0 <= u <= 1.0
                 else self._zero)
        else:
            reference = line_ahead(self.seg, self.ring, t_now, self.shaping.lead_s)
            h = min(max(t_now - self.t_last, 0.0), self.h_max)
            q, v, _ = self.shaping.smoother.step(reference, h)
        self.t_last = t_now
        if self._friction_on:
            q = q + self._friction * np.tanh(v / self.settings.friction_v)

        q = np.clip(q, self.q_cmd_last - self.slew, self.q_cmd_last + self.slew)
        if q_meas is not None:
            # + closes the gripper, so the bound is an upper one.
            q[GRIPPER] = min(q[GRIPPER], self.jaw.limit(t_now, float(q_meas[GRIPPER]),
                                                        float(q[GRIPPER])))
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


__all__ = ["BusServo", "ExecutorSettings", "JawBound"]

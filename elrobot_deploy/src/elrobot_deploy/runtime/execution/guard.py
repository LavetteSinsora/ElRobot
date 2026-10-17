"""The bus process's guards: is every joint still obeying its goal, and can the arm be seen?

Run every tick on the command just written and the position just read (`p7_bus.py`). Three
faults, each with its own response in the bus process:

  runaway   a joint is far off its command AND moving further away from it. That is the
            servo not obeying the goal at all (the ST3215's multi-turn offset trap drives a joint
            toward a stop whatever it is told), so that joint's torque goes off at once.
            Distance alone is not the test: a joint lagging a fast command is far off too, but
            it is moving TOWARD the command. So the rule is: more than `runaway_deg` off, and over
            the last `runaway_window_s` the error grew by `runaway_grow_deg` with the joint itself
            moving that way. Anything more than `gross_deg` off counts whatever it is doing
            (not the gripper: an object may hold the jaw anywhere).
  blocked   a joint more than `lag_deg` behind for `lag_s` without a break AND not making
            progress (it moved less than `stuck_deg` in that time): held by an obstacle. A joint
            that trails a long fast sweep is still moving, so it is not blocked. Not the gripper,
            which is held off its command by whatever it grips (the squeeze bound limits how
            far).
  read      `read_fails` position reads in a row failed: the arm cannot be seen.

Joints whose torque was turned off (`ignore`) are not checked again. The gripper is not checked
at all unless `guard_gripper`: it fights its command by design (squeezing an object, held off
it, springing back on release), and a runaway jaw only closes on its own stop at its hardware
torque limit (servo 8's TORQUE_LIMIT).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from elrobot_deploy.robot.elrobot_joints import GRIPPER


@dataclass(frozen=True)
class GuardSettings:
    """The thresholds (the profile's `[safety]`, degrees and seconds)."""

    runaway_deg: float = 25.0
    runaway_grow_deg: float = 10.0
    runaway_window_s: float = 0.3
    gross_deg: float = 60.0
    #: The blocked rule at all. False (`run --runaway-only`): only a runaway (and lost reads,
    #: which hold rather than release anything) can stop a session.
    blocked: bool = True
    lag_deg: float = 15.0
    lag_s: float = 1.5
    stuck_deg: float = 3.0
    read_fails: int = 5
    #: Check the gripper too (runaway rule). Off: its comparisons misfire (session 2 on the arm).
    guard_gripper: bool = False
    #: On a runaway or blocked fault, keep following the ring so P0 can bring the arm to rest
    #: before torque goes off (`schema.BUS_RECOVERING`). False: hold where it is.
    recover: bool = True
    #: Recover through the home pose first (home, then rest) instead of straight to rest.
    recover_via_home: bool = False


class Fault(NamedTuple):
    kind: str               # runaway | blocked | read
    joints: list[int]       # indices, 0-based
    message: str


class Guard:
    def __init__(self, settings: GuardSettings = GuardSettings()):
        self.s = settings
        n = GRIPPER + 1
        self.lag_since = np.full(n, np.nan)
        self.lag_from = np.zeros(n)                 # where the joint was when it fell behind
        self.read_fails = 0
        self.ignore: set[int] = set()
        #: True once a fault started a recovery: the blocked rule is off from then on (the
        #: recovery ramps must be allowed to finish), the runaway and read rules stay on.
        self.recovering = False
        self._hist: deque = deque()             # (t, meas, err) over the runaway window
        self._lag_mask = np.ones(n, dtype=bool)
        self._lag_mask[GRIPPER] = False

    def check(self, t: float, cmd: np.ndarray, meas: np.ndarray | None) -> Fault | None:
        s = self.s
        if meas is None:
            self.read_fails += 1
            if self.read_fails >= s.read_fails:
                return Fault("read", [], f"{self.read_fails} position reads in a row failed: "
                                         f"the arm cannot be seen")
            return None
        self.read_fails = 0
        err = np.asarray(meas, dtype=np.float64) - np.asarray(cmd, dtype=np.float64)
        live = np.ones(len(err), dtype=bool)
        live[list(self.ignore)] = False
        if not s.guard_gripper:
            live[GRIPPER] = False

        # runaway: far off, and moving further away (or grossly off)
        self._hist.append((t, np.array(meas, dtype=np.float64), err))
        while len(self._hist) > 2 and t - self._hist[1][0] >= s.runaway_window_s:
            self._hist.popleft()
        t0, m0, e0 = self._hist[0]
        absd = np.degrees(np.abs(err))
        gross = (absd > s.gross_deg) & self._lag_mask           # not the gripper
        away = np.zeros(len(err), dtype=bool)
        if t - t0 >= s.runaway_window_s:
            grow = absd - np.degrees(np.abs(e0))
            moved = np.degrees((np.asarray(meas) - m0) * np.sign(err))   # + = away from the goal
            away = (absd > s.runaway_deg) & (grow > s.runaway_grow_deg) & \
                   (moved > 0.5 * s.runaway_grow_deg)
        wild = np.flatnonzero((gross | away) & live)
        if wild.size:
            return Fault("runaway", [int(i) for i in wild],
                         f"joints {[int(i) + 1 for i in wild]} are "
                         f"{absd[wild].round(0).tolist()} deg off their command and moving away "
                         f"from it (not obeying the goal): their torque is off")

        # blocked: far behind for a long time
        if s.blocked and not self.recovering:
            late = (absd > s.lag_deg) & self._lag_mask & live
            self.lag_since[~late] = np.nan
            # Still moving: progress restarts the clock from here.
            moving = late & (np.degrees(np.abs(np.asarray(meas) - self.lag_from)) > s.stuck_deg)
            start = late & (np.isnan(self.lag_since) | moving)
            self.lag_since[start] = t
            self.lag_from[start] = np.asarray(meas)[start]
            stuck = np.flatnonzero(late & (t - self.lag_since > s.lag_s))
            if stuck.size:
                return Fault("blocked", [int(i) for i in stuck],
                             f"joints {[int(i) + 1 for i in stuck]} are more than "
                             f"{s.lag_deg:.0f} deg behind their command and have not moved for "
                             f"{s.lag_s:g} s (blocked?)")
        return None


__all__ = ["Fault", "Guard", "GuardSettings"]

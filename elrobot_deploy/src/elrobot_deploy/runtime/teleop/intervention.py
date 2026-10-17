"""The arithmetic of teleoperation. No I/O, no arena, no clock.

Pure functions of their arguments, so the whole geometry and timing of driving the arm from a
phone can be exercised with no phone, no robot and no IK. The phone source
(`runtime/processes/p11_phone.py`) supplies the packets, P6 the IK.

    `Console`    the clutch button -> is the operator driving, and the press / release edges.
    `RoomFrame`  the phone's pose since the press -> the TCP's motion, in the robot's axes.
    `Gripper`    the close button -> a jaw command, hold-to-close.
    `Pace`       the phone's capture clock -> a deadline on this machine's clock.
    `is_intervening`  the parity of one word.

THE ONE IDEA: THE CLUTCH

Only motion WHILE THE CLUTCH IS HELD is copied to the arm, and only RELATIVE to where the phone
was when it was pressed. At the press two things are frozen: the phone's pose (here) and the
TCP's pose (P6, which owns the model). Everything after is a delta from those anchors, so
nothing needs to know where the operator stands, and letting go and pressing again re-anchors
both halves: the remedy for every mapping problem, and how the operator walks the arm further
than their arm reaches.

THE MAPPING IS ROOM-FRAME (the user's decision, 2026-10-17)

"If I move the phone forward, the arm moves forward." Forward is the operator's forward, which
is the direction the phone's top points (the grip: screen facing left, top forward), taken
HORIZONTALLY at the press. Up is gravity: ARKit's world frame is gravity-aligned (+Y up). So
at each press `RoomFrame` builds the rotation `M` taking the room (forward, left, up) to the
robot base's (forward, left, up), and maps:

    translation   d_base = scale * M @ (p_now - p_press)                (world displacement)
    rotation      R_delta = M @ (R_now @ R_press^T) @ M^T               (world-frame rotation)

`R_delta` is applied to the TCP about its own position, in the base frame (P6:
`R_target = R_delta @ R_tcp_press`). So tilting the phone's top down pitches the gripper down
about the operator's left-right axis, and turning it left yaws the gripper left, whatever the
gripper's own orientation. (g1d's mapping was tool-frame, `legacy_g1d/.../intervention.py`.)

The phone's heading is the only thing taken from the phone at the press; its roll and pitch
there do not matter, so the operator need not hold it level, only point it forward.
"""

from __future__ import annotations

import numpy as np

from elrobot_deploy.robot.transforms import mat_from_quat, quat_from_mat

#: ARKit's world up (its world frame is gravity-aligned, +Y up).
WORLD_UP = np.array([0.0, 1.0, 0.0])


# --- the clutch -----------------------------------------------------------------------------


class Console:
    """One clutch button: the arm follows the phone while it is held.

    A hold, not a toggle: engagement is pressed and released many times within one motion, and
    what makes it usable is that releasing is free and re-pressing re-anchors.

    **Transitions are on the rising edge, and a fresh console is disarmed**: a button already
    down when the console starts (or when a guard forced a release) does not count until it has
    been seen up. So a phone lying on the clutch at startup never drives the arm, and a release
    forced by a guard needs a deliberate new press.
    """

    def __init__(self):
        self.engaged = False
        self._armed = False

    def update(self, pressed: bool) -> int:
        """One sample. Returns +1 if this engaged, -1 if this released, else 0."""
        pressed = bool(pressed)
        if not self._armed:
            if not pressed:
                self._armed = True
            return 0
        if self.engaged and not pressed:
            self.engaged = False
            return -1
        if not self.engaged and pressed:
            self.engaged = True
            return 1
        return 0

    def release(self) -> int:
        """Force a release (a guard fired, the stream went stale, the rollout ended).
        Returns -1 if that ended an engagement. The button must be seen up before it counts
        again."""
        edge = -1 if self.engaged else 0
        self.engaged = False
        self._armed = False
        return edge


# --- the room frame -------------------------------------------------------------------------


class HeadingUndefined(ValueError):
    """The phone's forward axis is (nearly) vertical at the press: no heading to take."""


class RoomFrame:
    """The phone's motion since the press, in the robot base's axes. See the module docstring.

    `phone_forward`: the device axis (ARKit camera frame) that points forward in the operator's
    grip, e.g. `(-1, 0, 0)`. `base_forward`, `base_up`: the robot base frame's forward and up.
    """

    #: How far from vertical the phone's forward axis must be at a press: below this its
    #: horizontal projection is too short to define a heading (sin 15 deg).
    MIN_HORIZONTAL = 0.26

    def __init__(self, phone_forward, base_forward, base_up, *, scale: float = 1.0):
        self.phone_forward = _unit(phone_forward)
        f, u = _unit(base_forward), _unit(base_up)
        #: Columns: the base frame's forward, left, up.
        self._base = np.c_[f, np.cross(u, f), u]
        self.scale = float(scale)
        self.M: np.ndarray | None = None          # room (ARKit world) -> base, set at a press
        self.p0: np.ndarray | None = None
        self.R0: np.ndarray | None = None

    def freeze(self, T_world_phone) -> None:
        """The press: remember the phone's pose and take the operator's heading from it."""
        T = np.asarray(T_world_phone, dtype=np.float64)
        R = T[:3, :3]
        fwd = R @ self.phone_forward
        horiz = fwd - WORLD_UP * float(fwd @ WORLD_UP)
        n = float(np.linalg.norm(horiz))
        if n < self.MIN_HORIZONTAL:
            raise HeadingUndefined("the phone's top points (nearly) straight up or down")
        f_w = horiz / n
        room = np.c_[f_w, np.cross(WORLD_UP, f_w), WORLD_UP]    # forward, left, up (world)
        self.M = self._base @ room.T
        self.p0, self.R0 = T[:3, 3].copy(), R.copy()

    def delta(self, T_world_phone) -> np.ndarray:
        """`[dx dy dz qw qx qy qz]`: the TCP motion, base frame, since `freeze`."""
        if self.M is None:
            raise RuntimeError("RoomFrame.delta before freeze")
        T = np.asarray(T_world_phone, dtype=np.float64)
        d = self.scale * (self.M @ (T[:3, 3] - self.p0))
        R_rel = self.M @ (T[:3, :3] @ self.R0.T) @ self.M.T
        return np.r_[d, quat_from_mat(R_rel)]

    def axes(self, T_world_phone) -> np.ndarray:
        """The phone's displacement since `freeze` as (forward, left, up) in the ROOM, metres,
        unscaled: what the bench tool prints to check the grip before anything moves."""
        T = np.asarray(T_world_phone, dtype=np.float64)
        return self._base.T @ (self.M @ (T[:3, 3] - self.p0))


def apply_delta(tcp_pose7, delta7) -> np.ndarray:
    """The TCP target: the press pose moved by `delta` (translation added, rotation applied
    about the TCP, both in the base frame)."""
    tcp = np.asarray(tcp_pose7, dtype=np.float64)
    d = np.asarray(delta7, dtype=np.float64)
    R = mat_from_quat(d[3:]) @ mat_from_quat(tcp[3:])
    return np.r_[tcp[:3] + d[:3], quat_from_mat(R)]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / np.linalg.norm(v)


# --- the gripper ----------------------------------------------------------------------------


class Gripper:
    """Hold-to-close: while the button is held the jaw command closes at `rate`; released, it
    opens at the same rate. Integrated on DEVICE time (the phone's capture clock), so the speed
    does not depend on how packets bunch up on WiFi; each step's `dt` is capped.

    `lo` is open and `hi` closed (servo angle, + closes). **The squeeze bound** (`measured`):
    the command may lead the measured jaw toward closed by at most `squeeze_margin`, and the
    bounded value is what is STORED (anti-windup), so releasing the button opens at once
    instead of first unwinding a command that ran on to "fully closed" against an object. On a
    position servo the grip force is about P x (command - measured), so this margin, not the
    command, sets how hard an object is held. The bus process applies the same bound every tick
    against the freshest read; this one keeps the integrator honest.

    **While the clutch is released the command holds** (the caller simply does not step it),
    so repositioning the hand never drops what is held.
    """

    MAX_DT = 0.1

    def __init__(self, rate: float, lo: float, hi: float, *, squeeze_margin: float = np.inf):
        self.rate = float(rate)
        self.lo, self.hi = float(lo), float(hi)
        self.squeeze_margin = float(squeeze_margin)
        self.cmd = self.lo

    def engage(self, commanded: float) -> None:
        """At each press: continue from the COMMANDED jaw (never the measured one, which an
        object holds open: anchoring there would drop the grip's force to zero)."""
        self.cmd = float(np.clip(commanded, self.lo, self.hi))

    def step(self, close: bool, dt: float, measured: float | None = None) -> float:
        dt = min(max(float(dt), 0.0), self.MAX_DT)
        cmd = self.cmd + (self.rate if close else -self.rate) * dt
        if measured is not None and np.isfinite(measured):
            cmd = min(cmd, float(measured) + self.squeeze_margin)
        self.cmd = float(np.clip(cmd, self.lo, self.hi))
        return self.cmd


# --- the clock ------------------------------------------------------------------------------


class Pace:
    """The phone's capture clock -> a deadline on this machine's clock.

    A pure shift, anchored on the first sample after `reset` (every press):

        due = t_anchor_local + (t_device - t_anchor_device),   t_anchor_local = arrival + lead

    so the motion is replayed at the pace it was made, whatever bursts the network delivers it
    in, and the lag is fixed for the engagement: `lead` plus the first sample's transport time.
    A sample later than the lead covers would be due in the past (and never executed), so a due
    time inside `MIN_MARGIN_S` of now slides the whole map later by just enough, once.
    """

    MIN_MARGIN_S = 0.005

    def __init__(self, lead: float):
        self.lead = float(lead)
        self.t_anchor_device: float | None = None
        self.t_anchor_local: float | None = None
        #: How far the map has slid since the press, s (latency spikes the lead didn't cover).
        self.slid = 0.0

    def reset(self) -> None:
        self.t_anchor_device = None
        self.t_anchor_local = None
        self.slid = 0.0

    def due(self, t_device: float, t_now: float) -> float:
        t_device = float(t_device)
        if self.t_anchor_local is None:
            self.t_anchor_device = t_device
            self.t_anchor_local = t_now + self.lead
        due = self.t_anchor_local + (t_device - self.t_anchor_device)
        floor = t_now + self.MIN_MARGIN_S
        if due < floor:
            self.t_anchor_local += floor - due
            self.slid += floor - due
            due = floor
        return float(due)


# --- the counter ----------------------------------------------------------------------------


def is_intervening(counter: int) -> bool:
    """Parity of `control.intervene`: odd means the operator is driving."""
    return bool(int(counter) & 1)


__all__ = [
    "WORLD_UP",
    "Console",
    "Gripper",
    "HeadingUndefined",
    "Pace",
    "RoomFrame",
    "apply_delta",
    "is_intervening",
]

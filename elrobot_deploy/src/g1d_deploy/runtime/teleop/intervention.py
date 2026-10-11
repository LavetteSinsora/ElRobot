"""The arithmetic of a human intervention. No I/O, no arena, no clock.

Five abstractions, each a pure function of its arguments, so the whole geometry
and timing of a takeover can be exercised on a laptop with no headset, no robot
and no GPU. P11 supplies the SDK, P6 supplies the IK, and neither contains any of
the reasoning below. For the operator:
`docs/deploying_a_policy/running_and_live_dashboard.md`, "`--intervene`: PICO
takeover"; the knobs are `docs/deploying_a_policy/deploy_toml.md`,
"`[intervene]`: the PICO takeover".

    `Anchor`   a frozen pair of poses; the operator's motion mapped onto the arm.
    `Console`  triggers -> who is driving what.
    `Gripper`  the thumbstick -> a jaw aperture.
    `Pace`     the PICO's device clock -> a deadline on this machine's clock.
    `is_intervening`  the parity of one word.

THE ONE IDEA

At the instant the operator squeezes a hand's trigger, two poses are frozen:
where their virtual gripper was, and where the robot's flange was.
Everything after that is a **delta between those two anchors**, so nothing here
ever needs to know where the operator is standing — and letting go and grabbing
again re-anchors both halves at once. That clutch is the remedy for every mapping
problem there is.

THE MAPPING IS TOOL-FRAME, AND THAT IS WHY THERE IS NO CALIBRATION HERE

The requirement is: if the operator's virtual gripper moves 10 cm along its own
out-of-the-jaws axis, the robot's gripper moves 10 cm along ITS own
out-of-the-jaws axis — whatever direction either of them happens to be pointing.
Write `A` for the operator's virtual TCP at the squeeze, `B` for it now, `C` for
the robot's FLANGE at the squeeze, `D` for the flange it is commanded to:

    p_D = p_C + s * R_C @ R_A.T @ (p_B - p_A)                   (1)
    R_D = R_C @ R_A.T @ R_B

`R_A.T @ (p_B - p_A)` is the operator's displacement **expressed in their own
tool frame** — a triple of numbers meaning "so far out of the jaws, so far across
them, so far along the hinge". `R_C @` puts that same triple back into the world
as the robot's tool frame reads it. **The XR world's orientation cancels between
the two**, which is why nothing in this file knows the OpenXR axis convention,
where the operator is standing, or which way the headset was facing when its
tracking initialised. There is no `R_align`, no headset yaw, and no per-engage
heading capture, because there is nothing for them to correct.

The rotation half is the same statement: `R_A.T @ R_B` is the operator's rotation
in their anchor tool frame, which is a body-frame rotation and therefore
frame-independent, and `R_C @` applies it about the robot's tool axes. Roll your
wrist about your controller's long axis and the gripper rolls about its own long
axis.

WHAT THIS DOES REQUIRE

(1) maps the operator's tool axes onto the robot's tool axes **by component
index**: axis 0 of one has to mean the same physical thing as axis 0 of the
other. If they disagreed you would need a fixed 3x3 between them — a constant of
the two rigs, not of where anybody stands.

The thing that makes them agree is `hand_tcp` (`DeployConfig.intervene_hand_tcp`,
controller -> the operator's virtual TCP) — which is why it takes a full pose:
its ROTATION must put the operator's tool axes in the robot flange's convention
(+X out of the jaws). A wrong rotation rotates every taken-over motion by the
error; `python -m g1d_deploy.runtime.processes.p11_pico` checks a candidate
with no robot powered.

WHY THE OPERATOR MUST NOT COMPOSE IN THE ROOM'S FRAME

The alternative reading of "move it forward" is the room's forward: rotate the
hand's displacement into base axes by a fixed room-to-robot rotation and add it.
That is a different feature, it needs the headset yaw that (1) does not, and it
is not what was asked for. Under it, "point the gripper down at the table and
push your hand forward" drives the arm forward in the room; under (1) it drives
the jaws further into the table, which is what "push the tool in" means to
somebody holding one.

The cost of (1) is that motion direction depends on how the ROBOT's gripper is
pointing at the squeeze — so the operator orients their controller to match the
gripper before taking hold. The anchor freezes at that instant, so from there the
correspondence is exact for as long as they stay engaged. If it ever feels wrong,
the fix is the clutch.

THE ROBOT SIDE IS THE FLANGE

The operator's side is their *virtual TCP* (`hand_tcp`, a measured
controller -> fingertip midpoint), whose ROTATION is what puts their
tool axes in the robot's convention. The robot's side is the **flange**, never
the robot file's TCP: the anchor freezes the flange pose and `follow` moves the
flange by what the operator's tool did.

So a rotation PIVOTS AT THE FLANGE. An operator who twists their wrist 90
degrees across the tool without moving their hand swings the Dex-1's fingertips
through ~26 cm — the chord of a quarter turn on its 18.5 cm arm — on top of
what their own virtual tool tip did. Turning a tool "in place" means moving the
hand too. This is a decision, not an oversight: the intervention drives the
same point on every end effector, whatever the robot file says its TCP is.

WHAT IS DELIBERATELY NOT HERE

**No release-on-stale.** Staleness is P11's business and it is a question about
one word (`control.intervene`), not about geometry. Nothing in this file has an
opinion about time except `Pace`, and `Pace` only maps one clock onto another.

**No jump filter and no envelope.** How far the arm may be driven is the IK's own
configuration and collision limits plus P6's velocity clamp against its seed —
both of which every policy target passes through too. A third opinion here would
be a bound with no measurement behind it.

**No quaternion convention conversion.** Everything in this file is g1d_deploy's
`[x y z qw qx qy qz]` (wxyz, `schema.POSE_DIM`). The SDK speaks xyzw, and
that conversion belongs at P11's boundary with the vendor, in exactly one place,
next to the code that knows it is talking to a vendor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.robot.kinematics import mat_from_quat, quat_from_mat

#: The identity pose, `[x y z qw qx qy qz]`. The default for both tool offsets,
#: so a caller that supplies neither gets flange-to-flange and controller-to-
#: controller — wrong for a real rig, but wrong in a way that is a pure identity
#: rather than a silent half-transform.
IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


# --- quaternions ------------------------------------------------------------
#
# Three lines of algebra rather than a dependency. `kin` already owns the
# matrix/quaternion conversions this system uses; what it does not have is the
# group operations, and importing a transform library for two of them would put a
# second quaternion convention one import away from a file whose entire job is to
# keep one straight.


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product, wxyz. `quat_mul(a, b)` is "rotate by b, then by a"."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q) -> np.ndarray:
    """The inverse of a UNIT quaternion, wxyz.

    Written as the conjugate rather than as a true inverse because every
    quaternion in this file has been normalised at P11's boundary, and the
    conjugate is exact where a division is not.
    """
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalise(q) -> np.ndarray:
    """Unit-length, and never silently zero.

    Raises rather than returning something plausible: a quaternion off unit
    length by a percent builds a matrix that is not quite a rotation, and every
    composition downstream compounds it. The one caller is P11's vendor boundary,
    where a degenerate value means the SDK returned nothing and the right answer
    is to drop the sample.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError(f"degenerate quaternion: {q}")
    return q / n


def pose_mul(a, b) -> np.ndarray:
    """Compose two poses, `a @ b`, both `[x y z qw qx qy qz]`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.empty(schema.POSE_DIM)
    out[:3] = a[:3] + mat_from_quat(a[3:]) @ b[:3]
    out[3:] = quat_mul(a[3:], b[3:])
    return out


def pose_inv(a) -> np.ndarray:
    """The inverse of a pose whose quaternion is unit length."""
    a = np.asarray(a, dtype=np.float64)
    qi = quat_conj(a[3:])
    out = np.empty(schema.POSE_DIM)
    out[:3] = -(mat_from_quat(qi) @ a[:3])
    out[3:] = qi
    return out


# --- 1. the anchor ----------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    """One hand's frozen pair, and the tool-frame map between them.

    `xr` is the OPERATOR's virtual TCP at the squeeze, in the XR world frame;
    `robot` is the ROBOT's FLANGE at the same instant, in the base frame. Both
    are frozen and neither is updated while the hand is engaged.

    `hand_tcp` is controller -> fingertip midpoint of the VIRTUAL gripper in the
    OPERATOR's hand, a measured calibration. Its
    ROTATION is the load-bearing half, because it is what puts the operator's
    tool axes in the same convention as the robot's — see the module
    docstring. Its translation decides whether a wrist twist swings their
    imagined tool tip or turns it in place.

    There is no robot-side TCP: a rotation pivots at the flange (module
    docstring, "THE ROBOT SIDE IS THE FLANGE").
    """

    xr: np.ndarray              # (7,) operator virtual TCP, XR world
    robot: np.ndarray           # (7,) robot flange, base frame
    hand_tcp: np.ndarray        # (7,) controller -> TCP, operator side

    @classmethod
    def freeze(cls, xr_controller7, robot_flange7, *,
               hand_tcp=IDENTITY_POSE) -> Anchor:
        """Freeze both halves. `robot_flange7` is a FLANGE pose because that is
        what the arena carries and what the IK consumes; `xr_controller7` is a
        CONTROLLER pose because that is what the SDK reports, and is put on the
        operator's virtual TCP here."""
        hand_tcp = np.asarray(hand_tcp, dtype=np.float64).reshape(schema.POSE_DIM)
        return cls(xr=pose_mul(xr_controller7, hand_tcp),
                   robot=np.asarray(robot_flange7, dtype=np.float64).reshape(
                       schema.POSE_DIM).copy(),
                   hand_tcp=hand_tcp)

    def follow(self, xr_controller7, scale: float = 1.0) -> np.ndarray:
        """The absolute FLANGE pose this hand should be at, `(7,)`.

        1. Put the operator's controller back on its virtual TCP.
        2. Move the robot's flange by what the operator's TCP did, **in tool
           coordinates**: `R_C @ R_A.T` carries a displacement out of the
           operator's tool frame and into the flange's, so the XR world's
           orientation never enters and no room-to-robot calibration exists.

        `scale` multiplies the TRANSLATION only. Scaling the rotation as well
        would make a full turn of the wrist a partial turn of the gripper, and the
        two would then disagree about which way is up.
        """
        now = pose_mul(xr_controller7, self.hand_tcp)
        r_c, r_a = mat_from_quat(self.robot[3:]), mat_from_quat(self.xr[3:])
        out = np.empty(schema.POSE_DIM)
        out[:3] = self.robot[:3] + scale * (r_c @ r_a.T @ (now[:3] - self.xr[:3]))
        out[3:] = quat_from_mat(r_c @ r_a.T @ mat_from_quat(now[3:]))
        return out


# --- 2. the triggers --------------------------------------------------------


class Console:
    """The operator's intent: one trigger per hand.

    **Each hand's trigger drives that arm while it is held.** A hold rather than
    a toggle because engagement is a clutch, pressed and released many times
    within one correction, and what makes it usable is that releasing is *free*
    and re-pressing re-anchors.

    **The intervention LEVEL is derived from the clutches**, not from a separate
    button: it opens when the first hand engages and closes when the last one
    lets go. So "the operator is driving" and "some hand is engaged" are the same
    proposition by construction, and there is no way for a session to be open with
    nobody holding anything.

    A hand that is inside an intervention but not engaged HOLDS — its arm keeps
    the pose it was last commanded — which is what makes single-arm correction
    work: grab the left, fix what went wrong, let go, and the right arm never
    moved.

    Transitions are on the RISING edge and a fresh `Console` is disarmed, one
    rule covering two situations. A trigger already down when the console is
    constructed does not read as a press, which matters at a rollout boundary
    where the operator may well be mid-squeeze, and at startup where a controller
    resting on a bench can have something against it.
    """

    #: Hysteresis on the analog triggers. Once engaged, stay engaged until the
    #: finger comes back below `RELEASE`: a finger resting near a single
    #: threshold would otherwise engage and disengage at the poll rate, and every
    #: one of those edges re-anchors the arm.
    PRESS = 0.6
    RELEASE = 0.4

    def __init__(self):
        self.engaged = np.zeros(schema.N_HANDS, dtype=bool)
        self._armed = np.zeros(schema.N_HANDS, dtype=bool)

    @property
    def active(self) -> bool:
        """Is an intervention running? Exactly "is any hand engaged"."""
        return bool(self.engaged.any())

    def reset(self) -> None:
        """Back to a fresh console: nothing engaged, and every trigger must be
        seen up before it may be pressed. What a rollout boundary calls, so that
        the next rollout starts exactly as the first one did."""
        self.engaged[:] = False
        self._armed[:] = False

    def update(self, trigger) -> tuple[int, np.ndarray]:
        """One sample of the triggers.

        Returns `(edge, grabbed)`: `+1` when this update opened an intervention,
        `-1` when it closed one, else `0`; and a `(2,)` mask of the hands that
        engaged on THIS update, which is what P11 re-anchors on.
        """
        was_active = self.active
        lever = np.asarray(trigger, dtype=np.float64).reshape(schema.N_HANDS)
        grabbed = np.zeros(schema.N_HANDS, dtype=bool)
        for h in range(schema.N_HANDS):
            v = float(lever[h]) if np.isfinite(lever[h]) else 0.0
            if not self._armed[h]:
                # Arming is on the LEVEL, not on a falling edge: a console built
                # while the trigger is already down has no falling edge to wait
                # for and would never arm at all.
                if v <= self.RELEASE:
                    self._armed[h] = True
                continue
            if self.engaged[h]:
                self.engaged[h] = v > self.RELEASE
            elif v >= self.PRESS:
                self.engaged[h] = True
                grabbed[h] = True
        active = self.active
        return (0 if active == was_active else (1 if active else -1)), grabbed

    def release_all(self) -> int:
        """Force every hand off, as a stale stream or a session exit does.

        Returns `-1` if that ended a running intervention, else `0` — the same
        vocabulary `update` speaks, so P11 has one handback path rather than two.
        """
        edge = -1 if self.active else 0
        self.reset()
        return edge


# --- 3. the thumbstick ------------------------------------------------------


class Gripper:
    """One hand's jaw: an anchor plus an integrated offset.

    The thumbstick is a VELOCITY control — hold it to drive the jaw, release it
    to stop wherever it got to — which is what makes a digital stick give
    continuous aperture.

    That stack commands the travel ENDPOINT and lets its own publisher ramp
    there. We cannot: P8 interpolates to the deadline, so an endpoint with a
    33 ms deadline slams the jaw across its whole 5.4 rad of travel in one action
    tick. So the ramp lives here instead, which is also the only place that can
    bound it.

    **`anchor` is the COMMANDED aperture at the squeeze, never the measured
    one**, and unlike the arm's anchor this is not a matter of a few millimetres.
    0 rad is fully closed, so a jaw squeezing an object sits ABOVE its command —
    you ask for 1.0, the object stops it at 1.4, and that difference IS the grip
    force through the driver's position loop. Anchoring on the measurement
    commands 1.4, the squeeze goes to zero, and the robot drops what it was
    holding at the very moment a human took over to save it.
    """

    #: Thumbstick hysteresis, three-state with a dead zone (press / release
    #: thresholds measured on a PICO controller).
    #: The dead zone is why `direction` is recorded beside `axis_y`: for most of
    #: every push the two disagree, and only one of them is what the jaw was
    #: asked for.
    PRESS = 0.55
    RELEASE = 0.25

    def __init__(self, rate: float, lo: float, hi: float, *,
                 squeeze_margin: float = np.inf, up_closes: bool = True,
                 close_is_max: bool = False):
        self.rate = float(rate)
        self.lo, self.hi = float(lo), float(hi)
        self.squeeze_margin = float(squeeze_margin)
        self.up_closes = bool(up_closes)
        #: Which end of `[lo, hi]` is closed. False for the Dex-1 (0 = closed),
        #: True for the BrainCo (1 = closed). It flips the stick's sign and
        #: turns the squeeze bound from a floor into a ceiling, and nothing else.
        self.close_is_max = bool(close_is_max)
        self.anchor = 0.0
        self.offset = 0.0
        self.direction = 0
        self._armed = False

    def engage(self, anchor: float) -> None:
        """Re-freeze at the commanded aperture and zero the offset.

        Called on every squeeze, so re-gripping never steps the jaw. The latch is
        disarmed here rather than merely reset: the stick is a velocity control,
        so a thumb already pushing when the trigger is squeezed would otherwise
        drive the jaw immediately. (The TRIGGER needs no such rule — its anchor
        freezes at the pose the arm already holds, so a zero delta commands zero
        motion — and the asymmetry is exactly the difference between a position
        map and a velocity one.)
        """
        self.anchor = float(anchor)
        self.offset = 0.0
        self.direction = 0
        self._armed = False

    def latch(self, axis_y: float) -> int:
        """The stick's three-state answer: `-1`, `0` or `+1`.

        Hysteresis in both directions, and arming on neutral: until the stick has
        been seen inside the dead zone once, it reads as 0 whatever it says.
        """
        v = float(axis_y) if np.isfinite(axis_y) else 0.0
        if not self._armed:
            if abs(v) <= self.RELEASE:
                self._armed = True
            self.direction = 0
            return 0
        if self.direction > 0:
            if v <= self.RELEASE:
                self.direction = 0
        elif self.direction < 0:
            if v >= -self.RELEASE:
                self.direction = 0
        elif v >= self.PRESS:
            self.direction = 1
        elif v <= -self.PRESS:
            self.direction = -1
        return self.direction

    def step(self, axis_y: float, dt: float, measured: float | None = None) -> float:
        """Advance the offset by one sample and return the jaw command.

        `dt` is DEVICE time since the previous sample, for the reason the
        deadlines use device time: the headset publishes over WiFi to the PC
        Service and samples arrive in bursts, so integrating on arrival would
        make three samples a millisecond apart move the jaw a ninth as far as
        the thumb asked for. The caller caps it; a gap must not dump a large
        offset in one step.

        **The offset is clamped, not the target**, and that is what decides
        whether this feels broken. Clamping only the target lets the offset wind
        past the stop — hold "close" for five seconds on a jaw already shut and
        it reaches -15 rad — and then pushing back does nothing at all for five
        seconds while it unwinds. Clamping the offset to the travel that remains
        either side of the anchor makes the target in range by construction and
        reverses instantly, and it also means the full 0..5.4 is reachable from
        ANY anchor: engaging mid-squeeze costs no travel, where a proportional map
        would have halved it.

        `measured` is THE SQUEEZE BOUND, and it is the one thing here that is
        about force rather than feel. g1d_deploy publishes jaw position straight
        to the wire at `GRIPPER_KP = 10.0` with no leash. A held stick is an
        integrator, so without a bound an operator can wind the command to 0
        against an object stopped at 1.4 rad and ask for 14 N·m. The bound
        is a floor under the target relative to the measurement rather than a
        limit on the offset, so it yields as the jaw actually closes and never
        becomes a dead zone.
        """
        direction = self.latch(axis_y)
        closing = 1.0 if self.close_is_max else -1.0
        sign = closing if self.up_closes else -closing
        self.offset = float(np.clip(
            self.offset + sign * direction * self.rate * float(dt),
            self.lo - self.anchor, self.hi - self.anchor))
        cmd = self.anchor + self.offset
        if measured is not None and np.isfinite(measured):
            if self.close_is_max:
                cmd = min(cmd, float(measured) + self.squeeze_margin)
            else:
                cmd = max(cmd, float(measured) - self.squeeze_margin)
        return float(np.clip(cmd, self.lo, self.hi))

    def hold(self) -> float:
        """The aperture a hand that is not engaged is commanded to: frozen where
        the operator last left it, with no integration and no squeeze bound."""
        return float(np.clip(self.anchor + self.offset, self.lo, self.hi))


# --- 4. the clock -----------------------------------------------------------


class Pace:
    """The PICO's device clock -> a deadline on this machine's clock.

    The map is a pure shift anchored once per intervention:

        due = t_anchor_local + (t_device - t_anchor_device)

    so the operator's motion is replayed at **the pace they made it**. That is
    the whole reason device time is used rather than arrival time: the headset
    publishes over WiFi to the PC Service on this box, and a network hop
    delivers in BURSTS — two samples a millisecond apart carrying 22 ms of hand
    motion between them — so arrival-timed deadlines would ask the arm to
    perform that 22 ms of travel in 1 ms, which P7 would faithfully attempt. The
    device clock is the only one that says how much motion a step contains.

    **`lead` is what makes the map survivable, and it is not decoration.** Anchor
    the map on the first sample with no lead and the margin on every later target
    is `L_1 - L_n`: the first sample's total latency minus this one's. That is a
    coin flip, so about half of all targets are published with a deadline already
    in the past, and P7 selects only on `deadline > now` — it discards them. With
    a lead the margin becomes `lead + L_1 - L_n`, positive whenever the lead
    covers the spread of the transport. On the deploy box that spread is measured
    at ~30 ms p1..p99 with P11 on an efficiency core, which is where
    `intervene_lead_s` comes from. It is also exactly the lag the operator feels,
    so responsiveness and robustness trade off one for one and this wants to be
    SET from a measurement rather than chosen.

    **`MIN_MARGIN_S` is the backstop that makes the map unable to die.** If the
    first sample of an intervention happened to be an unusually fast one, `L_1`
    is small and eats into the lead; and two monotonic clocks drift, so over
    minutes the map walks. Either way the deadlines approach the present and then
    pass it, and the arm stops responding for the rest of the intervention with
    nothing anywhere reporting why. So a due time that would land inside
    `MIN_MARGIN_S` of now slides the whole map later by just enough — permanently
    buying back the margin, at the cost of a few milliseconds of lag, once.

    It is deliberately much smaller than `lead`. Sliding whenever the margin fell
    short of the full lead would slide on every sample slower than the first,
    which at steady state is half of them, and the lag would ratchet up to the
    worst transport delay ever seen. At `MIN_MARGIN_S` it fires only when the
    margin has genuinely collapsed.
    """

    #: How close to `now` a deadline may be published before the map slides.
    #: Not a knob: it is "enough for P7 to see the slot at all", which is one
    #: servo period plus slack, and it has no bearing on feel.
    MIN_MARGIN_S = 0.005

    def __init__(self, lead: float):
        self.lead = float(lead)
        self.t_anchor_device: float | None = None
        self.t_anchor_local: float | None = None

    def reset(self) -> None:
        self.t_anchor_device = None
        self.t_anchor_local = None

    def due(self, t_device: float, t_now: float) -> float:
        t_device = float(t_device)
        if self.t_anchor_local is None:
            self.t_anchor_device = t_device
            self.t_anchor_local = t_now + self.lead
        due = self.t_anchor_local + (t_device - self.t_anchor_device)
        floor = t_now + self.MIN_MARGIN_S
        if due < floor:
            self.t_anchor_local += floor - due
            due = floor
        return float(due)


# --- the counter ------------------------------------------------------------


def is_intervening(counter: int) -> bool:
    """Parity of `control.intervene`: odd means the human is driving.

    The counter is monotonic and incremented once per transition, so its parity
    is the level and any CHANGE in it is "everything latched against the previous
    value is void". One word, two jobs, and they cannot disagree because every
    transition is both.

    A counter rather than a flag so that a reader which samples slowly cannot
    MISS a transition: a flag toggled twice between two reads is
    indistinguishable from no change at all, while a counter incremented twice is
    not — and one invalidation is the correct response to any number of
    transitions.
    """
    return bool(int(counter) & 1)


__all__ = [
    "IDENTITY_POSE",
    "Anchor",
    "Console",
    "Gripper",
    "Pace",
    "is_intervening",
    "pose_inv",
    "pose_mul",
    "quat_conj",
    "quat_mul",
    "quat_normalise",
]

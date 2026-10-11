"""P8 — hand control at 200 Hz.

Reads the target ring's `hand` column, writes the hand command topics
(`runtime/drivers/gripper.py`). Design: `docs/infra_logic/500hz_execution.md`
(the same walk as P7) and `docs/setup/robot_toml.md`, "`[end_effector]`".

Identical in structure to P7 over the hand column. The differences are all
subtractions: the payload is the `(2, 6)` command
rather than joint angles, so there is no gravity compensation, no torque and no
IK; it publishes no cursor and reads none; it may run on an E core.

The segment walk itself — the advance, the stall and the takeover preempt — is
`servo_segment.Segment`, the same object P7 walks with. That P8 filters its
advance on `control.intervene` exactly as P7 does is therefore not symmetry
kept up by hand: the body and hand columns share a row and a deadline,
so if the two servos disagreed about which side of a takeover a slot belongs
to, the jaw would go on executing the policy's aperture while the arm followed
the operator — precisely the configuration in which a gripped object is dropped.

**Why it needs no cursor at all**: P8 runs on the same deadlines as
P7, and every tick's hand slot carries the same `k` and the same `deadline` as
its body slot. So at any instant P8 sits in the same segment as P7 *by
construction*, and a second published cursor would be a second way to be wrong
about the same fact. The only divergence is sampling phase — P8 notices a
boundary up to one hand period (5 ms) after P7 does, a fifth of an action tick
against P6's two-tick write margin.

The stall matters here for exactly the reason it matters in P7, and the failure
is more visible: a jaw commanded to close across a 600 ms inference gap must not
be asked to do it in the 90 ms after the target lands.

One thing P7 has that this does not: the last-defence slew clamp. `qdot_max` in
the robot file is the 29 body joints; there is no configured velocity ceiling
for an end effector.
The Dex-1's own driver clamps every command against its measured position, which
is where that limit actually lives.

It does have a **range** clamp, which is a different thing.
`validate_actions` bounds the waist against the URDF's joint limits; the arm is
bounded three times over (the QP's `ConfigurationLimit`, P6's velocity clamp,
P7's slew clamp); without this the hand would be bounded nowhere — a policy has
been seen commanding 5.4304 rad against a top of travel at 5.4. The driver's own clamp does not cover this —
it bounds each STEP against the measured position, not the target, so a command
past the stop is pursued rather than refused. `hand_min` / `hand_max` are
facts about the end effector (`robot/end_effectors.py`), and the clamp sits here for the
same reason P7's does: this is the last point every command passes through,
whoever wrote it (P6 during a run, P0 during homing and the exit ramp).
"""

from __future__ import annotations

import numpy as np

from g1d_deploy.runtime import cpu_pinning as affinity, timing as clock
from g1d_deploy.runtime.shared_memory import Arena, build, open_all, schema
from g1d_deploy.config_loading.derive import derive
from g1d_deploy.runtime.execution.servo_segment import Phase, Segment


class HandServo:
    """One 200 Hz iteration, isolated from the process that spins it."""

    def __init__(self, *, ring, out, hw, cmd_last, t0: float, control):
        self.ring = ring
        self.out = out
        self.hw = hw
        self.control = control
        self.lo, self.hi = float(hw.hand_min), float(hw.hand_max)

        # The measured state is clamped too. It seeds `cmd_last`, so a gripper
        # resting a hair outside its own reported travel would otherwise make
        # the very first published command out of range.
        self.cmd_last = np.clip(
            np.asarray(cmd_last, dtype=np.float64), self.lo, self.hi)

        # Opens stalled at the measured end-effector state, the exact analogue
        # of P7's "q_cmd_last = measured posture at spawn".
        # `cursor=False`: P8 publishes no `k_exec` — see the module docstring.
        self.seg = Segment(ring, control, self.cmd_last, t0, cursor=False)

    stalled = property(lambda self: self.seg.stalled)
    t_segment_start = property(lambda self: self.seg.t_start)
    t_segment_end = property(lambda self: self.seg.t_end)
    cmd_segment_start = property(lambda self: self.seg.x_start)
    cmd_segment_end = property(lambda self: self.seg.x_end)

    def step(self, t_now: float) -> bool:
        """Advance one iteration at `t_now`.

        Returns True if the caller should wait for the next tick, False if this
        iteration advanced a segment — that branch publishes nothing and waits
        for nothing.
        """
        phase = self.seg.phase(t_now, self.cmd_last)
        if phase is Phase.HOLD:
            # The stall: republish the last COMMAND frozen rather
            # than anything measured, so a stalled jaw does not ratchet toward
            # wherever it has slipped.
            self.out.publish(self.cmd_last)
            return True
        if phase is Phase.ADVANCED:
            return False

        cmd = self.seg.at(self.seg.u_raw(t_now))
        # The range clamp. Applied to what is PUBLISHED and kept as `cmd_last`,
        # so an out-of-range target is truncated once rather than interpolated
        # toward forever — and so the next segment starts from a reachable
        # aperture rather than from one the jaw could never have been at.
        cmd = np.clip(cmd, self.lo, self.hi)
        self.out.publish(cmd)
        self.cmd_last = cmd
        return True


def run(shm_name: str, cfg, hw, *, out=None, ready=None) -> None:
    """The process. Runs from CHECK until STOPPED, through EXITING, so that
    P0's exit ramp on the hand ring is executed the same way as any other
    segment.

    `ready` is P0's homing gate, set after the first publish, exactly as in P7.
    P8 is gated for the same reason P7 is and not merely for symmetry:
    the homing ramp writes both rings, so a P0 that waited only on P7 would
    still hand a late P8 a ring full of apertures whose deadlines had passed."""
    arena = Arena.attach(shm_name, build(cfg, derive(cfg, hw)))
    r = open_all(arena, cfg)
    affinity.pin(hw, "p8_hand")
    affinity.set_realtime(hw, "p8_hand")
    control, state, ring = r["control"], r["state"], r["hand"]

    if out is None:
        from g1d_deploy.runtime.drivers.gripper import DdsHandCmd
        out = DdsHandCmd(hw)

    try:
        # The measured end-effector state at spawn, from the one place it
        # exists: P1's state ring (`grip_meas`). Zeros would be
        # outside a Dex-1's travel and would drive it into a hard stop.
        snap = state.latest()
        while snap is None and control.session < schema.STOPPED:
            clock.sleep_until(clock.now() + 1e-3)
            snap = state.latest()
        if snap is None:
            return
        cmd_last = np.asarray(snap["grip_meas"], dtype=np.float64).reshape(
            schema.N_HANDS, schema.HAND_DIM)

        out.connect()
        servo = HandServo(ring=ring, out=out, hw=hw, cmd_last=cmd_last,
                          t0=clock.now(), control=control)

        # The first command on the wire, before the loop — P7's rule, and the
        # same bound on the loop: `step` returns False only on a segment
        # advance, which publishes nothing.
        while not servo.step(clock.now()):
            pass
        if ready is not None:
            ready.set()

        pacer = clock.Pacer(hw.hand_hz)
        t = clock.now()
        while control.session < schema.STOPPED:
            t = pacer.wait() if servo.step(t) else clock.now()
    finally:
        out.close()
        arena.close()


__all__ = ["HandServo", "run"]

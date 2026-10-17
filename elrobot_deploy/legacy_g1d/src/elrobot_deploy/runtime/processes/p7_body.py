"""P7 — body control at 500 Hz.

Design: `docs/infra_logic/500hz_execution.md` ("What P7 does every 2 ms",
"What is not pluggable, and why").

Interpolates between whatever targets exist in the target ring and publishes to
`rt/lowcmd`. It owns `t_segment_start`, `q_segment_start`, `t_segment_end`,
`q_segment_end`, `stalled`, `q_cmd_last`, and the one published cursor
`k_exec`. It reads the target ring and the state ring (for gravity
compensation) and nothing else.

Each publish carries three numbers per joint, and this loop is the only place
two of them can come from:

* the **position**, from the interpolation;
* the **velocity**, which is that interpolation's own slope — free, exact, and
  known nowhere else in the system. Without it the firmware's `kd` term reads
  every commanded motion as velocity error and brakes it, leaving the joint
  `(kd/kp)` seconds of travel behind its target: 37.5 ms on every one of the
  fourteen arm joints at the shipped gains, measured at 42 ms. It is not a
  latency and there is no queue to drain — it is a damper being lied to;
* the **torque**, from `GravityComp` against the MEASURED joints.

**Shaping** (`runtime/execution/servo_filter.py`, `resolved.ServoShaping`) sits between the
walk and the wire: by default a critically damped filter reads the same target
line slightly ahead and publishes its own smooth position and velocity, and a
model feedforward adds the torque that motion needs. With `servo_filter.LINEAR` the
walk's line and slope go out unchanged, exactly as described above.

**P7 knows nothing.** It has no notion of homing, of sessions, of chunks, of
sync versus async, of the policy, of humans, or of safety. The one exception is
the rollout settings: it re-reads its servo settings (shaping and arm gains) when a
rollout starts under a new `rollout_generation` — a statement about which
settings are in force, taken between two publishes, never a branch in one.

That survives the intervention intact. P7 reads `control.intervene`, but never
its PARITY: it cannot tell a takeover from a handback and has no branch that
would care. All it knows is that when the word changes, every target it has
latched belongs to a world that no longer exists — which is a statement about
the ring, identical in kind to "that deadline has passed", and it is answered by
the mechanism P7 has anyway (the stall). The one behaviour the intervention
adds is that a segment in flight can be abandoned mid-way instead of only at
its deadline, which is what makes "stop" mean *now* to a person watching. Homing is ordinary target slots,
the exit ramp is ordinary target slots, and the damp is P0's.
Everything this loop does is "find the next deadline, walk toward it, publish".
That is why the code below is short, and it is what makes the two paths that
only ever run unattended — the first move of the day and the last — the *same*
path as every tick in between.

Three consequences of that purity are easy to undo by accident:

* **P7 never sends a damp.** That is P0's (`runtime/launcher/`). A servo loop that can
  stop the robot is a servo loop with a policy about when to, and that policy
  would run at 500 Hz on the one process that must never branch on anything.
* **P7 imports nothing from the IK / mujoco / mink stack**. The only
  kinematics it touches is `pin.rnea` gravity compensation, behind
  `runtime/drivers/arm_command.py::GravityComp`.
* **There is exactly one clamp**, the last-defence slew in the interpolate
  branch. There is no watchdog, no staleness test and no e-stop here: a dead
  upstream process means the arm holds its last command, decided deliberately.
"""

from __future__ import annotations

import numpy as np

from elrobot_deploy.runtime import cpu_pinning as affinity, timing as clock
from elrobot_deploy.runtime.shared_memory import Arena, build, open_all, schema
from elrobot_deploy.runtime.execution.servo_segment import Phase, Segment
from elrobot_deploy.runtime.execution.servo_filter import LINEAR, Shaping, line_ahead
from elrobot_deploy.runtime.execution.servo_filter import build as build_shaping
from elrobot_deploy.config_loading.derive import derive, servo_gains

class BodyServo:
    """One servo iteration, isolated from the process that spins it.

    Everything interesting about the servo is in `step()`: the interpolation and the
    clamp here, the segment advance and the stall in `Segment`. `run()` below
    only supplies a clock and a wire. Keeping them apart is what lets the stall
    be tested at microsecond resolution without sleeping.

    `shaping` (`runtime/execution/servo_filter.py`) says how the walked line becomes the
    published command. `LINEAR` — the default here, the unshaped servo —
    publishes the line and its slope; `run()` builds the config's.
    """

    def __init__(self, *, ring, control, state, out, gravity, hw,
                 q_cmd_last, t0: float, shaping: Shaping = LINEAR, command=None):
        self.ring = ring
        self.control = control
        self.state = state
        self.out = out
        self.gravity = gravity
        self.hw = hw
        self.shaping = shaping
        # P7's publish log (`schema.COMMAND`), read by P9. None in most tests.
        self.command = command

        # The largest step one servo period may command, per joint, in
        # body-target order (14 arm then 3 waist).
        self.qdot_max = np.asarray(hw.qdot_max_body, dtype=np.float64)
        self.slew = self.qdot_max / float(hw.body_hz)
        self.zero = np.zeros(schema.BODY_DOF)

        # Two "last" commands, equal on a linear servo: the LINE this loop walks
        # (today's command) and what was PUBLISHED, which the slew clamp bounds.
        self.q_cmd_last = np.asarray(q_cmd_last, dtype=np.float64).copy()
        self.q_line_last = self.q_cmd_last.copy()
        self.t_last = float(t0)
        self.h_max = 4.0 / float(hw.body_hz)

        # P7 opens stalled, holding the MEASURED posture, with the
        # ring empty. `cursor=True`: this is the one loop that publishes
        # `k_exec`.
        self.seg = Segment(ring, control, self.q_line_last, t0, cursor=True)

    # The walk's state, under stable names. Tests read these; nothing in the
    # loop writes them.
    stalled = property(lambda self: self.seg.stalled)
    t_segment_start = property(lambda self: self.seg.t_start)
    t_segment_end = property(lambda self: self.seg.t_end)
    q_segment_start = property(lambda self: self.seg.x_start)
    q_segment_end = property(lambda self: self.seg.x_end)

    def retune(self, shaping: Shaping, gains=None) -> None:
        """A new rollout's servo settings. Called between two steps.

        The filter is rebuilt seeded at the command last PUBLISHED, so the first
        shaped command continues from where the arm is being held rather than
        stepping. `gains` is `(kp, kd)` for the wire, when the wire can take
        them (`DdsLowCmd.set_gains`).
        """
        self.shaping = shaping
        if gains is not None and hasattr(self.out, "set_gains"):
            self.out.set_gains(*gains)

    # --- publishing ---------------------------------------------------------

    def _gravity_tau(self) -> np.ndarray:
        """Gravity compensation from the MEASURED joints, always.

        Even while stalled. The commanded posture is frozen during a stall
        but the arm is physically wherever it is, and a feedforward
        torque computed for a pose the arm is not in is the sag this term
        exists to remove.
        """
        snap = self.state.latest()
        if snap is None:
            return np.zeros(schema.BODY_DOF)
        return self.gravity.tau(snap["q"])

    # --- one iteration ------------------------------------------------------

    def step(self, t_now: float) -> bool:
        """Advance the loop by one iteration at `t_now`.

        Returns True if the caller should wait for the next servo tick, False
        if this iteration advanced a segment — that branch publishes nothing
        and waits for nothing, and falls straight into the next iteration,
        which publishes microseconds later against the new segment. No servo
        frame is dropped.
        """
        phase = self.seg.phase(t_now, self.q_line_last)
        if phase is Phase.ADVANCED:
            return False

        q_line, v_line = self._walk(t_now, phase)
        shaping = self.shaping
        if shaping.smoother is None:
            q, v, a = q_line, v_line, self.zero
        else:
            reference = line_ahead(self.seg, self.ring, t_now, shaping.lead_s)
            h = min(max(t_now - self.t_last, 0.0), self.h_max)
            q, v, a = shaping.smoother.step(reference, h)
        self.t_last = t_now

        # The last defence. It is the only clamp in this loop, and
        # in normal operation it never binds: P6 already clamped against the
        # seed over the tick span and homing's duration is derived
        # from the same ceiling. A bind here means one of those is wrong.
        q = np.clip(q, self.q_cmd_last - self.slew, self.q_cmd_last + self.slew)
        dq = np.clip(v * shaping.velocity_ff, -self.qdot_max, self.qdot_max)
        tau_ff = (shaping.feedforward.tau(v, a) if shaping.feedforward is not None
                  else self.zero)
        tau = self._gravity_tau() + tau_ff

        self.out.publish_body(q, tau, dq)
        if self.command is not None:
            self.command.write(t=t_now, k_exec=self.control.k_exec, q=q, dq=dq,
                               tau=tau, tau_ff=tau_ff, q_lin=q_line)
        self.q_cmd_last = q
        return True

    def _walk(self, t_now: float, phase: Phase) -> tuple[np.ndarray, np.ndarray]:
        """Today's command at `t_now`: the target line and its slope."""
        if phase is Phase.HOLD:
            # Zero velocity, and it is the honest value rather than a
            # convenient one: a stall is a HOLD, so the intended
            # velocity really is nothing. Carrying the expired segment's
            # slope here would tell the damper to expect motion that the
            # frozen command is not producing, and the joint would drift off
            # the hold under a feedforward nobody asked for.
            return self.q_line_last, self.zero

        u_raw = self.seg.u_raw(t_now)
        # The velocity the firmware is told to expect: **the derivative of the
        # line this loop is actually walking**, which is why it is computed
        # from `u_raw` and not from the clamped fraction.
        #
        # Inside the segment that derivative is the constant slope `delta/span`.
        # Outside it — the clamp's own territory, reachable on an inverted span
        # (`test_u_clamps_at_zero_on_an_inverted_span`) — the position is
        # pinned at an endpoint and is therefore not moving at all, so the slope
        # would be a velocity the position command is not producing.
        #
        # Clamped to the same ceiling the positions are, so the two channels
        # cannot disagree about what this machine may do: a span that collapses
        # to `EPS_SPAN` makes `delta/span` enormous, and a velocity feedforward
        # is a torque request — it must be bounded by something real.
        v = (np.clip(self.seg.delta / self.seg.span, -self.qdot_max, self.qdot_max)
             if u_raw >= 0.0 else self.zero)
        q = np.clip(self.seg.at(u_raw), self.q_line_last - self.slew,
                    self.q_line_last + self.slew)
        self.q_line_last = q
        return q, v


def relatch(servo: BodyServo, control, launch, hw, generation: int) -> int:
    """Take a new rollout's servo settings at its first tick.

    Returns the generation now in force. Two word loads when nothing changed,
    which is every period but one per rollout. Keyed on the session being
    ACTIVE, not on the generation alone: P0 bumps the generation at START and
    only then waits for P5, and a START the policy refuses leaves a generation
    that never runs — which this must not adopt. Anything the operator requests
    meanwhile is in `requested_config`, which this process never reads.
    """
    if (control.session != schema.ACTIVE
            or control.rollout_generation == generation):
        return generation
    from elrobot_deploy.config_loading.rollout_settings import latched

    cfg, _, generation = latched(control, launch)
    servo.retune(build_shaping(cfg.servo, hw, servo.q_cmd_last,
                               smoother=cfg.servo_smoother,
                               feedforward=cfg.servo_feedforward),
                 servo_gains(cfg, hw))
    print(f"[P7] rollout {generation}: servo {cfg.servo.describe()}", flush=True)
    return generation


def run(shm_name: str, cfg, hw, *, out=None, gravity=None, ready=None) -> None:
    """The process. Runs from CHECK until STOPPED.

    It stays alive through EXITING on purpose: P0's exit ramp is written into
    the same ring as everything else and needs an interpolator to execute it
    (the exit sequence). A P7 that stopped at ACTIVE would leave the robot stiff
    wherever the last policy tick put it.

    `ready` is P0's homing gate and is the ONLY thing this process is
    told about the session sequence — it sets a latch and never reads it, so
    "P7 knows nothing" still holds. It is set after the first publish rather
    than after `connect`, because what P0 is waiting for is a command on the
    wire: everything expensive here (the DDS handshake, pinocchio's URDF) sits
    between the two, and a gate that opened on `connect` would open on a P7 that
    still had a second of loading left.
    """
    from elrobot_deploy.config_loading.rollout_settings import latched

    arena = Arena.attach(shm_name, build(cfg, derive(cfg, hw)))
    r = open_all(arena, cfg)
    affinity.pin(hw, "p7_body")
    # Pinning says where this loop runs; SCHED_FIFO says it is not made to wait
    # there. Only asked for when the site file sets `rt_priority`.
    affinity.set_realtime(hw, "p7_body")
    control, state, ring = r["control"], r["state"], r["body"]
    # The servo settings in force: the active block's — the launch
    # config's before any START, the previous rollout's across a reset.
    launch = cfg
    cfg, _, generation = latched(control, launch)

    if out is None:
        from elrobot_deploy.runtime.drivers.arm_command import DdsLowCmd
        # The gains are the run's, not only the machine's: `servo_gains` applies
        # `cfg.servo.arm_gain_scale` (identity by default) and is the same
        # derivation P9 records.
        kp, kd = servo_gains(cfg, hw)
        out = DdsLowCmd(hw, kp=kp, kd=kd)
    if gravity is None:
        # No fallback to `ZeroComp` on a load failure. Without the feedforward
        # torque, kp alone holds the arm against gravity and kd brakes every
        # motion — the arm sags and sticks, plausibly, forever. A bad URDF path
        # should stop the run at startup, not degrade the tracking silently.
        from elrobot_deploy.runtime.drivers.arm_command import PinocchioComp
        gravity = PinocchioComp(hw)

    try:
        # The measured posture at spawn is both the legs' hold value and
        # `q_cmd_last` — which is what makes homing an ordinary
        # ramp from where the robot actually is rather than a jump.
        snap = state.latest()
        while snap is None and control.session < schema.STOPPED:
            clock.sleep_until(clock.now() + 1e-3)
            snap = state.latest()
        if snap is None:
            return
        q29 = np.asarray(snap["q"], dtype=np.float64)

        out.connect(q29[list(hw.joints_leg)])
        q_body = q29[list(hw.joints_arm) + list(hw.joints_waist)]
        servo = BodyServo(
            ring=ring, control=control, state=state, out=out, gravity=gravity,
            hw=hw, q_cmd_last=q_body, t0=clock.now(),
            shaping=build_shaping(cfg.servo, hw, q_body,
                                  smoother=cfg.servo_smoother,
                                  feedforward=cfg.servo_feedforward),
            command=r["command"])

        # The first command on the wire, before the loop, so `ready` means
        # exactly that. `step` returns False only on the frame that advances a
        # segment — which publishes nothing — so the loop runs at most twice and
        # never spins: the next call falls into the interpolate branch.
        while not servo.step(clock.now()):
            pass
        if ready is not None:
            ready.set()

        pacer = clock.Pacer(hw.body_hz)
        t = clock.now()
        while control.session < schema.STOPPED:
            generation = relatch(servo, control, launch, hw, generation)
            t = pacer.wait() if servo.step(t) else clock.now()
    finally:
        out.close()
        arena.close()


__all__ = ["BodyServo", "relatch", "run"]

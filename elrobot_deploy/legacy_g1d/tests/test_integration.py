"""P5, P6, P7 and P8 over ONE arena — the claims no single section can test.

Each section's own tests drive its loop against a ring it fills by hand. That
cannot catch the failures this system is actually built to avoid, because every
one of them lives *between* two loops: an epoch P6 stamps and P5 extrapolates
from, a deadline P6 computes and P7 interpolates over, a cursor P7 publishes and
P6 gates on.

So this file runs the real loops against a real arena under a virtual clock.
The clock is driven by **P5's own sleep**: when P5 waits for its trigger
instant, the wait pumps P6, P7 and P8 forward one servo period at a time. That
is the true dependency structure — P5 is the only loop that ever waits on
wall-clock time — and it makes a full sync cycle deterministic and instant.

Only the three things outside the system are fake: the policy (an echo), the IK
(an affine stand-in, so the timeline is tested independently of the QP), and the
wire (a recorder). Every ring, cursor, deadline and gate is the real one.
"""

from __future__ import annotations

import numpy as np
import pytest

from elrobot_deploy.runtime.execution import chunk_blend as blend
from elrobot_deploy import policy_contract as policymod
from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.processes.p5_policy import PolicyLoop
from elrobot_deploy.runtime.processes.p6_target import Target
from elrobot_deploy.runtime.processes.p7_body import BodyServo
from elrobot_deploy.runtime.processes.p8_hand import HandServo
from elrobot_deploy.runtime.drivers.arm_command import FakeLowCmd, ZeroComp
from elrobot_deploy.runtime.drivers.gripper import FakeHandCmd

T0 = 1000.0


# --- stand-ins for the three things outside the system ----------------------


class AffineSolver:
    """A stand-in for the QP: joint angles that are an affine function of the
    commanded flange position, so a target's identity survives into the ring and
    can be asserted on. Never fails, so `ok=False` never confounds a timing test.
    """

    def solve(self, pose, waist, seed_q14, budget=None):
        q = np.full(schema.ARM_DOF, float(pose[0, 0]))
        return q, 0.0, True


class BudgetlessSolver(AffineSolver):
    """Fails every solve the way an overrun does: writes nothing, leaves a hole."""

    def solve(self, pose, waist, seed_q14, budget=None):
        return np.zeros(schema.ARM_DOF), 9.9, False


class StubPolicy(policymod.Policy):
    """Produces a chunk whose action `i` is tagged with a value the solver turns
    into joint angles, so a target in the body ring can be traced back to the
    chunk and row it came from.

    It returns the whole horizon every time: truncation to `P` in sync is the
    infra's now. The deadline shift is off because the rig publishes its frames
    far in the future so the anchor never waits — a shift off that would be the
    clipped maximum and would smear the tags this file traces.
    """

    action_mode = "eef"
    end_effector = "dex1"
    anchor_camera = "rwrist"
    deadline_shift = False

    def __init__(self, cfg):
        super().__init__(None, cfg.prompt, None)
        self.cfg = cfg
        self.n_calls = 0

    def _actions(self, n):
        self.n_calls += 1
        tag = float(self.n_calls)
        pose = np.zeros((n, schema.N_HANDS, schema.POSE_DIM))
        pose[:, :, 0] = tag + np.arange(n)[:, None] / 100.0
        pose[:, :, 3] = 1.0                      # unit quaternion, wxyz
        # The gripper is inside the Dex-1's travel, as the contract requires;
        # the tag lives on the pose.
        return policymod.EefActions(pose=pose,
                                    hand=np.full((n, schema.N_HANDS), 1.0))

    def infer(self, obs, reader):
        return self._actions(self.cfg.H)


class Rig:
    """One arena, four real loops, one virtual clock."""

    def __init__(self, cfg, hw, rings, *, solver=None, inference_s=0.0):
        self.cfg, self.hw, self.rings = cfg, hw, rings
        self.t = T0
        self.servo_dt = 1.0 / hw.body_hz
        self.inference_s = inference_s

        rings["control"].session = schema.ACTIVE
        for region in rings["cameras"].values():
            region.spare()["pixels"][:] = 1
            region.publish(t_recv=T0 + 1e6)
        rings["state"].write(
            t_recv=T0 + 1e6, q=np.zeros(schema.N_JOINTS),
            dq=np.zeros(schema.N_JOINTS),
            flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
            grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=T0 + 1e6)
        # Homing has ended, so tick 0 holds the home posture and its
        # deadline is the instant the ramp finished.
        rings["targets"].write(0, T0, np.zeros(schema.BODY_DOF),
                               np.zeros((schema.N_HANDS, schema.HAND_DIM)), intervene=0)

        self.body_out = FakeLowCmd(hw)
        self.body_out.connect(np.zeros(12))
        self.hand_out = FakeHandCmd(hw)
        self.hand_out.connect()

        self.p6 = Target(rings, cfg, hw, solver or AffineSolver(),
                         blend_fn=blend.get(cfg.blend), now=self.now)
        self.p7 = BodyServo(ring=rings["body"], control=rings["control"],
                            state=rings["state"], out=self.body_out,
                            gravity=ZeroComp(), hw=hw,
                            q_cmd_last=np.zeros(schema.BODY_DOF), t0=T0)
        self.p8 = HandServo(ring=rings["hand"], out=self.hand_out, hw=hw,
                            cmd_last=np.zeros((schema.N_HANDS, schema.HAND_DIM)),
                            t0=T0, control=rings["control"])
        self.policy = StubPolicy(cfg)
        self.p5 = PolicyLoop(cfg, rings, self.policy, clock=self)

        self.hand_phase = 0.0
        self.k_exec_seen: list[int] = []

    # the clock interface PolicyLoop expects
    def now(self) -> float:
        return self.t

    def sleep_until(self, target: float) -> float:
        """P5's wait IS the simulation's time source (see the module docstring).

        The inference cost is added here rather than inside the stub policy
        because in sync mode it is the thing that makes `k` freeze while the
        clock keeps running — the whole reason the epoch has to be re-stamped.
        """
        while self.t < target:
            self.advance()
        self.t += self.inference_s
        while self.t_pumped < self.t:
            self.advance(pump_only=True)
        return self.t

    t_pumped = T0

    def advance(self, *, pump_only: bool = False) -> None:
        if not pump_only:
            self.t = round(self.t + self.servo_dt, 9)
        self.t_pumped = self.t
        # P6 runs as fast as it solves; once per servo period is a conservative
        # stand-in and is still far faster than the 30 Hz action rate.
        self.p6.step()
        self.p7.step(self.t)
        if self.t >= self.hand_phase:
            self.p8.step(self.t)
            self.hand_phase = self.t + 1.0 / self.hw.hand_hz
        self.k_exec_seen.append(self.rings["control"].k_exec)

    def pump(self, seconds: float) -> None:
        stop = self.t + seconds
        while self.t < stop:
            self.advance()


@pytest.fixture
def sync_cfg(cfg):
    return cfgmod.replace(cfg, sync=True, replan_period_ticks=4, action_horizon=8)


@pytest.fixture
def async_cfg(cfg):
    return cfgmod.replace(cfg, sync=False, replan_period_ticks=4, action_horizon=8)


def _rig(cfg, hw, arena, **kw):
    from elrobot_deploy.runtime.shared_memory import open_all
    return Rig(cfg, hw, open_all(arena, cfg), **kw)


# --- The mutual wait closes -------------------------------------


def test_sync_cycle_runs_end_to_end(sync_cfg, hw, arena):
    """The claim of the `+2` start: P5 waits on a tick P6 must write, from a chunk P5 must
    produce, and it is well-founded. If the `+2` on either side were wrong, this
    deadlocks rather than fails an assertion."""
    rig = _rig(sync_cfg, hw, arena, inference_s=0.05)
    rig.p5.first_inference()
    for _ in range(4):
        rig.pump(0.02)
        assert rig.p5.step() is not None, "P5 starved: the mutual wait did not close"
    assert rig.rings["control"].newest_chunk_id == 5
    assert rig.rings["control"].k_exec > 0
    assert len(rig.body_out.published) > 100


def test_the_first_solved_tick_is_exactly_where_p5_looks(sync_cfg, hw, arena):
    """P5 opens its window at `k_start_read = 2` and P6's first solve is
    `k_exec + 2 = 2`. The two `+2`s are the same `+2`."""
    rig = _rig(sync_cfg, hw, arena)
    rig.p5.first_inference()
    solved = []
    for _ in range(6):
        step = rig.p6.step()
        if step.written:
            solved.append(step.k)
    assert solved[0] == 2 == rig.p5.k_start_read


def test_tick_one_is_never_executed(sync_cfg, hw, arena):
    """The chunk's first action is dropped in every cycle."""
    rig = _rig(sync_cfg, hw, arena)
    rig.p5.first_inference()
    rig.pump(0.2)
    assert rig.rings["body"].get(1) is None
    assert rig.rings["body"].get(2) is not None


# --- The epoch ---------------------------------------------


def test_sync_restamps_the_epoch_so_deadlines_do_not_absorb_inference(
        sync_cfg, hw, arena):
    """The failure this prevents: without the re-stamp every deadline after the
    first chunk is early by the accumulated inference latency, so P7 is asked to
    execute a tick whose deadline has already passed and the arm lurches."""
    rig = _rig(sync_cfg, hw, arena, inference_s=0.2)
    rig.p5.first_inference()
    rig.pump(0.05)
    rig.p5.step()                       # 200 ms of virtual inference passes here
    written = None
    for _ in range(10):
        step = rig.p6.step()
        if step.written:
            written = step
            break
    assert written is not None
    assert written.deadline > rig.t, (
        "a freshly solved tick was given a deadline already in the past — the "
        "epoch was not re-stamped after inference")
    assert written.deadline == pytest.approx(rig.t + 2 * sync_cfg.dt, abs=1e-9)


def test_deadlines_are_affine_in_k_within_one_epoch(async_cfg, hw, arena):
    """The epoch rule and the basis of the window scan: any solved tick in the window extrapolates to
    the same trigger instant. Asserted directly on the ring."""
    rig = _rig(async_cfg, hw, arena)
    rig.p5.first_inference()
    rig.pump(0.3)
    entries = [rig.rings["body"].get(k) for k in range(2, 9)]
    entries = [e for e in entries if e is not None]
    assert len(entries) >= 3
    gaps = [(b.deadline - a.deadline) / (b.k - a.k)
            for a, b in zip(entries, entries[1:])]
    assert all(g == pytest.approx(async_cfg.dt, abs=1e-9) for g in gaps)


def test_p5_extrapolates_identically_from_any_tick_in_the_window(
        async_cfg, hw, arena):
    """The window scan's whole justification. Read the window from each solved tick in turn
    and assert the schedule is bit-identical — this is what makes a hole
    harmless."""
    rig = _rig(async_cfg, hw, arena)
    rig.p5.first_inference()
    rig.pump(0.3)
    body = rig.rings["body"]
    ticks = [k for k in range(rig.p5.k_start_read, rig.p5.k_next_inference + 1)
             if body.get(k) is not None]
    assert len(ticks) >= 2, "need several solved ticks to make the point"
    schedules = [rig.p5.schedule(body.get(k)) for k in ticks]
    # Identical to within floating-point rounding, not literally bit-identical:
    # `deadline + (k_next - k) * dt` rounds differently depending on which `k`
    # it started from. The spread is ~1 ULP — sub-picosecond against a 33 ms
    # tick — but it is not zero, by design.
    assert max(schedules) - min(schedules) < 1e-9


# --- P6 -> P7 handover ------------------------------------------------------


def test_both_rings_get_the_same_k_and_deadline(sync_cfg, hw, arena):
    """The jaw and the arm must not drift onto different timelines."""
    rig = _rig(sync_cfg, hw, arena)
    rig.p5.first_inference()
    rig.pump(0.3)
    common = 0
    for k in range(0, 12):
        b, h = rig.rings["body"].get(k), rig.rings["hand"].get(k)
        assert (b is None) == (h is None), f"tick {k}: one ring written, not both"
        if b is not None:
            assert b.deadline == h.deadline
            common += 1
    assert common >= 3


def test_no_target_is_written_onto_the_tick_p7_is_interpolating(sync_cfg, hw, arena):
    """Checked continuously: after every P6 step, the tick it wrote must
    never be `k_exec` or `k_exec - 1` as P7 currently holds them."""
    rig = _rig(sync_cfg, hw, arena)
    rig.p5.first_inference()
    for _ in range(400):
        step = rig.p6.step()
        k_exec = rig.rings["control"].k_exec
        if step.written:
            assert step.k not in (k_exec, k_exec - 1)
        rig.p7.step(rig.t)
        rig.t = round(rig.t + rig.servo_dt, 9)


def test_k_exec_is_monotonic_across_a_whole_run(sync_cfg, hw, arena):
    """`k_exec` monotonicity, including across the sync stall where it must simply stop moving."""
    rig = _rig(sync_cfg, hw, arena, inference_s=0.1)
    rig.p5.first_inference()
    for _ in range(3):
        rig.pump(0.02)
        rig.p5.step()
    seen = rig.k_exec_seen
    assert seen == sorted(seen)
    assert seen[-1] > 0


# --- P7 output --------------------------------------------------------------


def test_every_published_command_respects_the_slew_limit(sync_cfg, hw, arena):
    """The clamp holds across segment boundaries and across
    the stall, not just inside one segment."""
    rig = _rig(sync_cfg, hw, arena, inference_s=0.1)
    rig.p5.first_inference()
    for _ in range(3):
        rig.pump(0.02)
        rig.p5.step()
    pub = np.array([p[0] for p in rig.body_out.published])
    assert len(pub) > 100
    steps = np.abs(np.diff(pub, axis=0))
    assert np.all(steps <= hw.qdot_max_body / hw.body_hz + 1e-9)


def test_the_arm_actually_tracks_the_chunk(sync_cfg, hw, arena):
    """The end-to-end claim: a pose in a chunk becomes a joint angle on the wire.

    The affine solver makes chunk 1's tag (1.0) the joint value, so the arm must
    leave home and settle near it rather than merely moving smoothly.
    """
    rig = _rig(sync_cfg, hw, arena, inference_s=0.02)
    rig.p5.first_inference()
    for _ in range(3):
        rig.pump(0.05)
        rig.p5.step()
    arm = np.array([p[0][:schema.ARM_DOF] for p in rig.body_out.published])
    # The first published command is already one servo step into the motion —
    # the slew limit is what bounds it, not the segment start.
    assert arm[0].max() <= (hw.qdot_max_body[:schema.ARM_DOF] / hw.body_hz).max()
    assert arm[-1].max() > 1.0, "should have tracked the commanded chunk"


def test_a_stall_holds_position_and_does_not_creep(sync_cfg, hw, arena):
    """The stall across processes: during the sync inference gap P7 republishes a
    frozen command. Measured tracking error must not ratchet into the target."""
    rig = _rig(sync_cfg, hw, arena, inference_s=0.3)
    rig.p5.first_inference()
    rig.pump(0.02)
    rig.rings["state"].write(
        t_recv=rig.t, q=np.full(schema.N_JOINTS, 0.4), dq=np.zeros(schema.N_JOINTS),
        flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
        grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=rig.t)
    rig.pump(0.25)                              # long past the chunk's last tick
    tail = np.array([p[0] for p in rig.body_out.published[-40:]])
    assert np.allclose(tail, tail[0]), "the hold crept toward the measured q"


# --- holes ------------------------------------------------------------------


def test_a_hole_from_an_abandoned_solve_does_not_stop_p5(async_cfg, hw, arena):
    """An IK that abandons some ticks leaves holes, and
    the window scan is what keeps P5 running over them."""
    class Flaky(AffineSolver):
        def __init__(self):
            self.n = 0

        def solve(self, pose, waist, seed_q14, budget=None):
            self.n += 1
            if self.n % 2 == 0:
                return np.zeros(schema.ARM_DOF), 9.9, False
            return super().solve(pose, waist, seed_q14, budget)

    rig = _rig(async_cfg, hw, arena, solver=Flaky())
    rig.p5.first_inference()
    for _ in range(3):
        rig.pump(0.05)
        assert rig.p5.step() is not None, "a hole in the window starved P5"


def test_p5_waits_forever_when_p6_writes_nothing(async_cfg, hw, arena):
    """The empty-window block, asserted as ACCEPTED behaviour rather than fixed: if P6 produces
    nothing at all, P5 returns None indefinitely and never escalates. This test
    exists so that adding a timeout later is a deliberate, visible change."""
    rig = _rig(async_cfg, hw, arena, solver=BudgetlessSolver())
    rig.p5.first_inference()
    for _ in range(50):
        rig.pump(0.01)
        assert rig.p5.step() is None
    assert rig.rings["control"].newest_chunk_id == 1


# --- async ------------------------------------------------------------------


def test_async_never_freezes_k_and_stamps_the_epoch_once(async_cfg, hw, arena):
    """In async `k` does not freeze, so the epoch is stamped once, ever.
    A second stamp would shift every subsequent deadline."""
    rig = _rig(async_cfg, hw, arena, inference_s=0.05)
    rig.p5.first_inference()
    rig.pump(0.05)
    first = rig.rings["body"].get(2)
    for _ in range(3):
        rig.pump(0.05)
        rig.p5.step()
    later = [rig.rings["body"].get(k) for k in range(3, 14)]
    later = [e for e in later if e is not None]
    for e in later:
        assert e.deadline == pytest.approx(
            first.deadline + (e.k - first.k) * async_cfg.dt, abs=1e-9), (
            "async re-stamped its epoch")


# --- homing, P0 against P7 and P8 -------------------------


class HomingClock:
    """A clock whose passage of time IS the two servo loops stepping.

    The same trick as `Rig`, one phase earlier: during HOME there is no P5 and
    no P6, and the only loop that waits on wall-clock time is P0 itself — inside
    `write_ramp`'s pacing and the sleep to the last deadline. So P0's wait pumps
    P7 and P8, and a homing ramp runs end to end, deterministically, with no
    process and no robot.

    The "robot" tracks perfectly: every servo tick echoes the published command
    back into the state ring. That is what lets `_verify_home` run for real —
    the check that used to pass on a snap because it only ever looked at the
    endpoint.

    `late_by` is the bug. It delays the instant the servo loops begin stepping,
    which is exactly what P7's ~2.8 s of DDS and pinocchio startup did.
    """

    def __init__(self, rings, hw, *, q0, grip0, late_by: float = 0.0):
        self.rings, self.hw = rings, hw
        self.t = T0
        self.servo_dt = 1.0 / hw.body_hz
        self.start_at = T0 + late_by
        self.hand_phase = self.start_at

        self.body_out = FakeLowCmd(hw)
        self.body_out.connect(np.zeros(12))
        self.hand_out = FakeHandCmd(hw)
        self.hand_out.connect()
        self.p7 = BodyServo(ring=rings["body"], control=rings["control"],
                            state=rings["state"], out=self.body_out,
                            gravity=ZeroComp(), hw=hw, q_cmd_last=q0,
                            t0=self.start_at)
        self.p8 = HandServo(ring=rings["hand"], out=self.hand_out, hw=hw,
                            cmd_last=grip0, t0=self.start_at,
                            control=rings["control"])
        self._echo(q0, grip0)

    def _echo(self, q_body17, grip_2x6) -> None:
        """The robot: what was commanded is what is measured, one tick later."""
        q29 = np.zeros(schema.N_JOINTS)
        q29[list(self.hw.joints_arm) + list(self.hw.joints_waist)] = q_body17
        self.rings["state"].write(
            t_recv=self.t, q=q29, dq=np.zeros(schema.N_JOINTS),
            flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
            grip_meas=np.asarray(grip_2x6, dtype=float).ravel(),
            t_grip_recv=self.t)

    def now(self) -> float:
        return self.t

    def sleep_until(self, target: float) -> float:
        while self.t < target:
            self.t = round(self.t + self.servo_dt, 9)
            if self.t < self.start_at:
                continue
            while not self.p7.step(self.t):     # production's own loop shape
                pass
            if self.t >= self.hand_phase:
                while not self.p8.step(self.t):
                    pass
                self.hand_phase = self.t + 1.0 / self.hw.hand_hz
            self._echo(self.body_out.published[-1][0], self.hand_out.published[-1])
        return self.t

    # what the assertions are about
    def steps(self) -> np.ndarray:
        """Per-servo-tick command deltas, `(n-1, 17)`."""
        q = np.stack([p[0] for p in self.body_out.published])
        return np.abs(np.diff(q, axis=0))


def _homing_rig(cfg, hw, rings, derived, *, late_by=0.0):
    from elrobot_deploy.runtime.launcher import Launcher

    # A short floor keeps the whole ramp inside the fixture's 16-slot ring.
    short = cfgmod.replace(cfg, home_min_s=0.4)
    q0 = np.zeros(schema.BODY_DOF)
    q0[3] = 0.5                                     # 0.5 rad from home_q = 0
    grip0 = np.full((schema.N_HANDS, schema.HAND_DIM), 5.4)   # jaw wide open

    clock = HomingClock(rings, hw, q0=q0, grip0=grip0, late_by=late_by)
    lch = Launcher(short, hw, clock=clock, log=lambda *_: None)
    lch.rings, lch.derived = rings, derived
    return lch, clock, short


def test_homing_is_executed_as_a_ramp_when_the_servo_loops_are_up_first(
        rings, cfg, hw, derived):
    """The whole point of homing, asserted end to end for the first time.

    P7's slew clamp is a LAST DEFENCE: homing's duration is derived from
    the same `qdot_max` the clamp is built from, so during homing it must never
    bind. If it binds, the ramp was not executed as a ramp — and that is a
    silent failure, because the clamp turns the violation into a plausible,
    merely-fast motion that `_verify_home` then accepts on its endpoint.
    """
    lch, clock, short = _homing_rig(cfg, hw, rings, derived)

    lch.home()                                       # includes _verify_home

    slew = clock.p7.slew
    steps = clock.steps()
    assert np.all(steps.max(axis=0) < slew), "the slew clamp bound during homing"
    # Not merely "under" it — comfortably under, with the limit still a limit.
    # (This rig compresses `home_min_s` to 0.4 s to fit the fixture's ring, so
    # the peak sits near a third of the ceiling; the shipped 3 s floor uses 4%.)
    assert steps.max() < 0.5 * slew.max(), "a ramp leaves the limit headroom"

    # And it took the whole window rather than the end of it: half the travel at
    # the midpoint is the cosine's own signature.
    q = np.stack([p[0] for p in clock.body_out.published])[:, 3]
    progress = np.abs(q - q[0]) / abs(q[-1] - q[0])
    half = int(np.argmax(progress >= 0.5)) / len(progress)
    assert 0.35 < half < 0.65, f"half the motion done at t={half:.2f} of the ramp"

    # The gripper rode the same grid, and also as a ramp: it starts at the
    # measured aperture rather than at `home_hand`, and closes monotonically.
    grips = np.array([c[0, 0] for c in clock.hand_out.published])
    assert grips[0] > 0.8 * 5.4, "the jaw stepped to home_hand on tick one"
    assert np.all(np.diff(grips) <= 1e-12), "an aperture that only ever closes"
    # Not exactly 0.0: P8 samples at 200 Hz and notices the last boundary up to
    # one hand period after P7 does, so the stream stops one 5 ms slice
    # short of the final target. The TARGET is exact; the sampling is not.
    assert grips[-1] < 0.02
    half = int(np.argmax(np.abs(grips - grips[0]) >= 0.5 * 5.4)) / len(grips)
    assert 0.35 < half < 0.65, f"the jaw's half-travel is at t={half:.2f}"


def test_a_servo_loop_that_starts_late_turns_the_whole_ramp_into_one_step(
        rings, cfg, hw, derived):
    """The 2026-08-26 bug, kept so the gate has something to protect.

    `write_ramp` does not pace inside the ring's capacity, so the entire ramp is
    in the ring before a late P7 runs one iteration. Its first `next_after` then
    returns a tick near the END of the ramp, and it leaves the stall with a
    single segment in which to cover the whole distance — bounded only by the
    clamp the slew-clamp rule says never binds. On the real robot that was 0.44 rad in ~150 ms
    at 5.5 rad/s, five rollouts out of five.

    `_verify_home` does not save it, and that is the sharpest part of the claim:
    homing still "arrives", because arriving is all it checks.
    """
    lch, clock, _ = _homing_rig(cfg, hw, rings, derived, late_by=0.15)

    lch.home()                                       # does NOT raise

    steps = clock.steps()
    assert np.any(steps.max(axis=0) >= clock.p7.slew * 0.999), (
        "expected the slew clamp to saturate — the bug this gate prevents")
    # It was over almost before it began: a late start does not slow the motion
    # down to fit, it spends the ramp it missed at the ceiling instead.
    q = np.stack([p[0] for p in clock.body_out.published])[:, 3]
    progress = np.abs(q - q[0]) / abs(q[-1] - q[0])
    assert int(np.argmax(progress >= 0.5)) / len(progress) < 0.25

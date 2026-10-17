"""P0 — homing, the exit ramp and the damp.

Both ramps are exercised as pure functions over a real arena: no processes, no
robot, no DDS. What has to hold is what nobody would notice if it did not — a
homing ramp that ends one tick short of `home_q`, a step that quietly needs
twice `qdot_max`, an exit ramp written on top of the segment P7 is already
interpolating, a damp published while P7 is still commanding the arm.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.launcher import Launcher
from elrobot_deploy.runtime.launcher.ramps import (cosine_ramp, cosine_samples, home_ticks,
                                               ramp_duration, write_ramp)
from elrobot_deploy.runtime.launcher.state_machine import _ExitNow

T0 = 1000.0


def _hands(n: int, value: float = 0.0) -> np.ndarray:
    """`n` hand commands all reading `value` — a ramp that happens to be flat."""
    return np.full((n, schema.N_HANDS, schema.HAND_DIM), value)


class FakeClock:
    def __init__(self, t: float = T0):
        self.t = t
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep_until(self, t: float, **_) -> float:
        self.sleeps.append(t)
        self.t = max(self.t, t)
        return self.t


class CursorClock(FakeClock):
    """A clock that plants `k_exec` on a chosen tick just before P0 looks."""

    def __init__(self, control, k: int):
        super().__init__()
        self.control = control
        self.k = k

    def sleep_until(self, t: float, **_) -> float:
        self.control.k_exec = self.k
        return super().sleep_until(t)


class FakeProc:
    def __init__(self, name: str, events: list):
        self.name, self.events = name, events
        self.alive = True

    def terminate(self):
        self.events.append(f"stop:{self.name}")
        self.alive = False

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return self.alive


class RecorderProc:
    """A P9 stand-in that leaves on the session word, and remembers what it saw.

    `saw_session` is the whole point: the claim under test is not "P9 was
    joined" but "P9 was given STOPPED *before* anyone tried to kill it", and
    only the child's own view of the word can show that.
    """

    def __init__(self, control, events: list, *, obeys: bool = True):
        self.control, self.events, self.obeys = control, events, obeys
        self.alive = True
        self.saw_session = None

    def terminate(self):
        self.events.append("stop:p9_record")
        self.alive = False

    def join(self, timeout=None):
        if timeout is None:            # the join inside stop(), after SIGTERM
            return
        self.saw_session = self.control.session
        if self.obeys:
            self.events.append("p9_closed")
            self.alive = False

    def is_alive(self):
        return self.alive


class FakeWire:
    """The exit sequence wire. `publish_body` exists only so a test can prove P0 never
    calls it: P0 owns `rt/lowcmd` for the damp and for nothing else."""

    def __init__(self, hw, events: list):
        self.events = events

    def connect(self, leg_q_hold):
        self.events.append("connect")

    def publish_body(self, q_body17, tau_body17, dq_body17):
        self.events.append("publish_body")

    def damp(self):
        self.events.append("damp")

    def close(self):
        self.events.append("close")


def _q(value: float) -> np.ndarray:
    return np.full(schema.BODY_DOF, value)


# --- The ramp ---------------------------------------------------


def test_duration_is_the_velocity_limit_or_the_floor(cfg, hw):
    """Derived, not configured, so the ramp is inside the limit by
    construction and a slew clamp during homing is a genuine bug."""
    qdot = hw.qdot_max_body
    slow = ramp_duration(_q(0.0), _q(0.0), qdot_max=qdot, min_s=3.0, margin=0.5)
    assert slow == 3.0                          # nothing to move: the floor

    # Far enough that the velocity term beats the floor at the real limit
    # (2*pi rad/s, the vendor's max_pos_speed): 30 rad at half of that is ~9.5 s.
    far = _q(0.0).copy()
    far[0] = 30.0
    want = 30.0 / (qdot[0] * 0.5)
    assert want > 3.0, "the point of this case is that the limit, not the floor, binds"
    assert ramp_duration(_q(0.0), far, qdot_max=qdot, min_s=3.0,
                         margin=0.5) == pytest.approx(want)


def test_the_slowest_joint_sets_the_duration(cfg, hw):
    """qdot_max is per joint: a wrist that may move twice as fast must not
    hide a waist that may not."""
    qdot = np.full(schema.BODY_DOF, 4.0)
    qdot[-1] = 0.5                              # one slow joint
    q_to = _q(1.0)
    assert ramp_duration(_q(0.0), q_to, qdot_max=qdot, min_s=0.0,
                         margin=1.0) == pytest.approx(1.0 / 0.5)


def test_homing_ends_exactly_at_home_q(cfg, hw):
    qs = cosine_ramp(_q(0.3), _q(-0.4), dt=cfg.dt, qdot_max=hw.qdot_max_body,
                     min_s=cfg.home_min_s, margin=cfg.home_velocity_margin)
    assert np.array_equal(qs[-1], _q(-0.4))     # exactly, not nearly
    assert not np.array_equal(qs[0], _q(0.3))   # q_from is not a written tick


@pytest.mark.parametrize("delta", [0.0, 0.05, 1.5, 6.0])
def test_every_ramp_step_is_inside_the_velocity_limit(cfg, hw, delta):
    """Including the first step, which is measured from where the robot
    already is. The cosine's peak is pi/2 times its mean, which is what the
    0.5 velocity margin is there to absorb."""
    q_from = _q(0.0)
    q_to = _q(delta)
    qs = cosine_ramp(q_from, q_to, dt=cfg.dt, qdot_max=hw.qdot_max_body,
                     min_s=cfg.home_min_s, margin=cfg.home_velocity_margin)
    steps = np.diff(np.vstack([q_from, qs]), axis=0)
    assert np.all(np.abs(steps) <= hw.qdot_max_body * cfg.dt + 1e-12)


def test_the_profile_eases_in_and_out(cfg, hw):
    """Velocity continuous at both ends: the robot is stationary before the
    ramp and must be stationary at home_q."""
    qs = cosine_ramp(_q(0.0), _q(1.0), dt=cfg.dt, qdot_max=hw.qdot_max_body,
                     min_s=cfg.home_min_s, margin=cfg.home_velocity_margin)
    steps = np.diff(np.vstack([_q(0.0), qs]), axis=0)[:, 0]
    assert steps[0] < steps[len(steps) // 2] > steps[-1]
    assert np.all(steps >= 0)


def test_homing_occupies_negative_ticks_and_ends_at_zero(cfg):
    """Within a rollout tick numbering never restarts, k_exec stays
    monotonic, and the startup state falls out with no special case.

    `reset` restarts the numbering between rollouts, which is a different
    claim and is tested where it happens — this one is about the shape of a
    single homing ramp, and that shape is what reset reproduces.
    """
    ks = home_ticks(5)
    assert list(ks) == [-4, -3, -2, -1, 0]
    assert ks[-1] == 0 and np.all(ks[:-1] < 0)
    assert list(home_ticks(1)) == [0]


# --- writing a ramp into the rings ------------------------------------------


def test_a_ramp_writes_both_rings_with_the_same_k_and_deadline(rings, cfg):
    """A body slot and a hand slot with the same k carry the same
    deadline, and P0 writes both for a tick or neither."""
    qs = np.linspace(0.0, 1.0, 5)[:, None] * np.ones(schema.BODY_DOF)
    hands = cosine_samples(np.zeros((schema.N_HANDS, schema.HAND_DIM)),
                           np.full((schema.N_HANDS, schema.HAND_DIM), 0.25), 5)
    written = write_ramp(rings, home_ticks(5), qs, hands, t_first=T0,
                         dt=cfg.dt, lead_ticks=0, clock=FakeClock())
    assert [k for k, _ in written] == [-4, -3, -2, -1, 0]
    for i, (k, deadline) in enumerate(written):
        body, hand = rings["body"].get(k), rings["hand"].get(k)
        assert body is not None and hand is not None
        assert body.deadline == hand.deadline == deadline
        assert deadline == pytest.approx(T0 + i * cfg.dt)
        assert np.allclose(hand.payload, hands[i])
    assert np.allclose(rings["body"].get(0).payload, 1.0)
    assert np.allclose(rings["hand"].get(0).payload, 0.25)


def test_a_ramp_moves_the_hand_over_its_ticks_rather_than_on_the_first(rings,
                                                                       cfg):
    """The end effector used to be ONE held command written into all n ticks.

    That is a step wearing a ramp's clothing: P8 takes the final aperture in its
    first segment and drives there in a single action tick, while the arm eases
    over three seconds beside it. P8's loop has no slew clamp to absorb it, by
    decision — the Dex-1's driver bounds each step against the measurement, not
    the target — so nothing downstream turns it back into a motion.
    """
    n = 9
    hands = cosine_samples(np.zeros((schema.N_HANDS, schema.HAND_DIM)),
                           np.full((schema.N_HANDS, schema.HAND_DIM), 5.4), n)
    write_ramp(rings, home_ticks(n), np.zeros((n, schema.BODY_DOF)), hands,
               t_first=T0, dt=cfg.dt, lead_ticks=0, clock=FakeClock())

    got = np.array([rings["hand"].get(k).payload[0, 0] for k in home_ticks(n)])
    assert got[0] < 0.2 * 5.4, "the first tick is a step, not the start of a ramp"
    assert np.all(np.diff(got) > 0), "an aperture that only ever moves forward"
    assert got[-1] == pytest.approx(5.4), "and lands exactly on home_hand"
    # The EASE, not merely monotonicity — which a linear ramp also has. A cosine
    # is slow at both ends and fastest in the middle, so the jaw is stationary
    # where the ramp joins the hold before it and the hold after it.
    steps = np.diff(np.concatenate([[0.0], got]))
    assert steps[0] < steps[n // 2] and steps[-1] < steps[n // 2]


def test_tick_zero_is_written_even_though_k_exec_still_reads_zero(rings, cfg):
    """The cursor is stale until P7 takes its first homing target. A gate that
    believed that zero would refuse the one tick the startup state requires."""
    assert rings["control"].k_exec == 0
    qs = np.zeros((3, schema.BODY_DOF))
    written = write_ramp(rings, home_ticks(3), qs, _hands(3),
                         t_first=T0, dt=cfg.dt, lead_ticks=0, clock=FakeClock())
    assert [k for k, _ in written] == [-2, -1, 0]
    assert rings["body"].get(0) is not None


def test_the_gate_refuses_the_segment_p7_is_interpolating(rings, cfg):
    """Never write into {k_exec − 1, k_exec}. P0 is an ordinary
    producer here — the same discipline as P6, on the same rings."""
    ks = np.arange(10, 21)
    qs = np.zeros((len(ks), schema.BODY_DOF))
    written = write_ramp(rings, ks, qs, _hands(len(ks)),
                         t_first=T0, dt=cfg.dt, lead_ticks=0,
                         clock=CursorClock(rings["control"], 15))
    assert [k for k, _ in written] == [10, 11, 12, 13, 16, 17, 18, 19, 20]
    assert rings["body"].get(14) is None and rings["body"].get(15) is None


def test_the_ramp_is_paced_by_the_rings_own_capacity(rings, cfg, derived):
    """A ramp may be longer than the ring, so P0 may not write it all at once.
    The lead is the ring's size, not a configured number."""
    lead = derived.n_target - 2
    clock = FakeClock()
    ks = np.arange(0, 6)
    write_ramp(rings, ks, np.zeros((6, schema.BODY_DOF)), _hands(6),
               t_first=T0, dt=cfg.dt, lead_ticks=lead, clock=clock)
    assert clock.sleeps == [pytest.approx(T0 + i * cfg.dt - lead * cfg.dt)
                            for i in range(6)]


def test_a_ramp_whose_rings_disagree_in_length_is_refused(rings, cfg):
    """Body and hand are written together per tick. A short `hands` would not be a
    short ramp — `zip` would stop early and leave the arm ramping on into ticks
    whose hand slot still holds whatever P6 last wrote."""
    with pytest.raises(ValueError, match="ramp lengths disagree"):
        write_ramp(rings, home_ticks(6), np.zeros((6, schema.BODY_DOF)),
                   _hands(5), t_first=T0, dt=cfg.dt, lead_ticks=0,
                   clock=FakeClock())


# --- did homing arrive? -----------------------------------------


def _measured(rings, hw, q_body17, *, t=T0, grip=0.0):
    """Publish one state sample whose 17 body joints read `q_body17`.

    `grip` is where every hand dimension is measured — the aperture a homing
    hand ramp starts from, exactly as `q_body17` is the posture the body ramp
    starts from.
    """
    q29 = np.zeros(schema.N_JOINTS)
    q29[list(hw.joints_arm) + list(hw.joints_waist)] = q_body17
    rings["state"].write(t_recv=t, q=q29, dq=np.zeros(schema.N_JOINTS),
                         flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                         grip_meas=np.full(schema.GRIP_DIM, grip),
                         t_grip_recv=t)


def test_homing_that_arrives_is_accepted(rings, cfg, hw, derived):
    lch, _ = _launcher(cfg, hw, rings, derived)
    q_to = np.zeros(schema.BODY_DOF)
    _measured(rings, hw, q_to + 1e-4)
    lch._verify_home(q_to)                              # does not raise


def test_arm_droop_and_the_structural_bow_are_tolerated(rings, cfg, hw, derived):
    """The measured numbers this robot actually produces, and why both pass.

    Left-arm droop reached 0.091 rad on the 2026-08-20 rollouts — the arm carries
    the context camera and sags under `kp`. That is honest: the observation
    carries the measured joints, so the policy sees where the arm really is.

    The bow — waist slot 1, and it IS a bow rather than a roll (`kin.WAIST_JOINT_NAMES`)
    — sits at 0.039 rad on every rollout, moving under 0.005 rad across a whole
    run while commanded to zero throughout. It is ~11.7 Nm of upper-body gravity
    against kp=300, which the joint has no authority to close. This test exists
    because it used to assert the opposite: it demanded a refusal at 0.0389, and
    `home_tol_waist = 0.01` duly refused every run the machine can produce.
    """
    lch, _ = _launcher(cfg, hw, rings, derived)
    q_to = np.zeros(schema.BODY_DOF)

    droop = q_to.copy()
    droop[3] = 0.091                                    # L_elb, the worst seen
    _measured(rings, hw, droop)
    lch._verify_home(q_to)                              # accepted

    real = droop.copy()
    real[schema.ARM_DOF + 1] = 0.0389                   # the bow, as measured
    _measured(rings, hw, real)
    lch._verify_home(q_to)                              # accepted: structural


def test_a_waist_that_actually_moved_is_still_refused(rings, cfg, hw, derived):
    """Loosening the tolerance to clear the bow must not disarm the check.

    The failure worth catching is a trunk somewhere it was not commanded —
    knocked, mis-homed, or moved by a checkpoint that unfreezes the waist. P6's
    IK pins the trunk at the COMMANDED angles and `PinocchioComp` locks it at
    zero, so a trunk elsewhere biases every flange pose in the run and nothing
    else reports it. 0.05 clears the 0.039 droop and still refuses this.
    """
    lch, _ = _launcher(cfg, hw, rings, derived)
    q_to = np.zeros(schema.BODY_DOF)

    moved = q_to.copy()
    moved[schema.ARM_DOF + 1] = 0.20                     # well past the droop
    _measured(rings, hw, moved)
    with pytest.raises(ConfigError) as raised:
        lch._verify_home(q_to)

    # It must identify WHICH waist joint, and it must not call slot 1 a roll —
    # that is the humanoid enum's name for it and this robot has no roll axis.
    # What the label actually reads ("bow", "pitch") is the model map's business;
    # pinning the exact word here would just re-hardcode what was hardcoded
    # before.
    message = str(raised.value)
    assert "waist joint 1" in message
    assert "roll" not in message


def test_an_arm_that_never_arrived_is_refused(rings, cfg, hw, derived):
    """Droop is bounded; an arm blocked or homed to the wrong posture is not."""
    lch, _ = _launcher(cfg, hw, rings, derived)
    q_to = np.zeros(schema.BODY_DOF)
    stuck = q_to.copy()
    stuck[10] = 0.4                                     # R_elb, 23 degrees out
    _measured(rings, hw, stuck)
    with pytest.raises(ConfigError, match="arm joint 10"):
        lch._verify_home(q_to)


def test_home_runs_the_check_itself(rings, cfg, hw, derived):
    """The wiring, not the predicate: `home()` used to end by sleeping to the
    last deadline and returning — commanding a posture and never confirming it.

    Here the measured posture never moves (there is no P7 in this test), so the
    ramp is written, the sleep happens, and the robot is still where it started
    — which is what an arm that did not follow looks like from P0.
    """
    lch, _ = _launcher(cfg, hw, rings, derived)
    start = np.zeros(schema.BODY_DOF)
    start[3] = 0.8                                      # far from home_q = 0
    _measured(rings, hw, start)

    with pytest.raises(ConfigError, match="homing did not arrive"):
        lch.home()

    assert rings["control"].session == schema.HOME      # it got that far
    assert rings["body"].get(0) is not None             # the ramp WAS written


def test_home_accepts_a_robot_that_was_already_there(rings, cfg, hw, derived):
    lch, _ = _launcher(cfg, hw, rings, derived)
    _measured(rings, hw, np.zeros(schema.BODY_DOF))     # already at home_q
    lch.home()                                          # does not raise


def test_homing_ramps_the_gripper_from_where_it_is_to_home_hand(rings, cfg, hw,
                                                                derived):
    """Both rings ramp, on one tick grid.

    The arm's velocity limit sets the duration and the hand rides it. A gripper
    has no `qdot_max` to derive one from — P8's loop has no clamp, and inventing one
    would be an unmeasured knob — but it does not need one: a tick's
    two payloads share a deadline, so they share a duration, and the arm's is
    the one derived from a real limit.

    `home_hand` is zeros in the fixture, so a jaw measured OPEN has somewhere to
    travel. The 2026-08-26 commit that homes the grippers open made this live:
    before it, both apertures happened to already be at 5.4 on every rollout, so
    the step was invisible in the records.
    """
    # A short floor keeps the whole ramp inside the 16-slot ring so the test can
    # read it back, as in the exit-ramp test: 3 s at 30 Hz wraps it five times.
    short = cfgmod.replace(cfg, home_min_s=0.4)
    lch, _ = _launcher(short, hw, rings, derived)
    _measured(rings, hw, np.zeros(schema.BODY_DOF), grip=5.4)   # jaw wide open

    lch.home()

    n = int(np.ceil(short.home_min_s / short.dt))
    got = np.array([rings["hand"].get(k).payload[0, 0] for k in home_ticks(n)])
    assert got[0] > 0.8 * 5.4, "the first tick is a step, not the start of a ramp"
    assert np.all(np.diff(got) < 0), "an aperture that only ever closes"
    assert got[-1] == pytest.approx(0.0), "and lands exactly on home_hand"
    # And the body rode the same grid: one duration, one deadline per tick.
    assert rings["body"].get(0).deadline == rings["hand"].get(0).deadline


def test_the_tolerances_are_the_experiments_to_set(rings, cfg, hw, derived):
    """An offset that is understood and accepted is a config change, not a code
    change — so the refusal names the knob and the knob is per-experiment."""
    loose = cfgmod.replace(cfg, home_tol_waist=0.05)
    lch, _ = _launcher(loose, hw, rings, derived)
    q_to = np.zeros(schema.BODY_DOF)
    known = q_to.copy()
    known[schema.ARM_DOF + 1] = 0.0389
    _measured(rings, hw, known)
    lch._verify_home(q_to)                              # accepted, deliberately


# --- warmup -----------------------------------------------------


class WarmingClock(FakeClock):
    """A clock that lets P5 finish warming after a few polls of P0's wait."""

    def __init__(self, control, *, after: int, count: int = 1):
        super().__init__()
        self.control, self.after, self.count = control, after, count
        self.polls = 0

    def sleep_until(self, t: float, **kw) -> float:
        self.polls += 1
        if self.polls == self.after:
            self.control.warm = self.count
        return super().sleep_until(t, **kw)


def test_warmup_waits_for_p5s_warm_word(rings, cfg, hw, derived):
    """The 2026-08-20 fix: the operator must not be offered READY until P5 has
    actually compiled. Every one of those rollouts paid a 29.9 s XLA compile
    inside its first REAL inference — on an observation that was 29.9 s stale
    by the time the arm executed it — because Enter was accepted first."""
    control = rings["control"]
    clock = WarmingClock(control, after=4)
    lch, _ = _launcher(cfg, hw, rings, derived, clock=clock)
    assert control.warm == 0

    lch.warmup()

    assert clock.polls >= 4, "returned without waiting for P5"
    assert control.warm == 1
    assert control.session == schema.WARMUP


def test_warmup_refuses_a_p5_that_died_before_it_warmed(rings, cfg, hw, derived):
    """A bad checkpoint, a missing GPU, an OOM. `_wait` has no timeout by
    decision, so the only thing that can end this wait other than success is
    the fact that the process is gone — never a clock."""
    lch, _ = _launcher(cfg, hw, rings, derived)
    lch.children.handles["p5_policy"].alive = False

    with pytest.raises(ConfigError, match="exited before it warmed"):
        lch.warmup()


def test_a_spawn_backend_that_cannot_answer_is_treated_as_alive(rings, cfg, hw,
                                                                derived):
    """`alive` must never turn an unanswerable question into "the child died" —
    that would refuse a healthy run. It degrades to the old behaviour: wait."""
    control = rings["control"]
    clock = WarmingClock(control, after=2)
    lch, _ = _launcher(cfg, hw, rings, derived, clock=clock)

    class Mute:
        def is_alive(self):
            raise OSError("no such backend")

    lch.children.handles["p5_policy"] = Mute()
    lch.warmup()                                  # waits, then succeeds

    assert control.warm == 1


def test_p5_is_spawned_by_check_so_it_warms_through_homing(cfg, hw):
    """Loading a checkpoint and compiling an executable needs nothing this
    system produces, so it must overlap the homing ramp rather than queue
    behind it.

    Structural, like `test_policy_env_is_applied_before_any_import`: `check`
    verifies DDS rates and hand topics that a unit test cannot supply, so the
    claim is asserted where it is decided — which method contains the spawn.
    `warmup` must contain no spawn at all, because a `warmup` that both starts
    P5 and waits on its `warm` word would deadlock on any run whose spawn
    failed silently.
    """
    import inspect

    assert '"p5_policy"' in inspect.getsource(Launcher.check)
    assert "children.start" not in inspect.getsource(Launcher.warmup)


# --- The homing gate --------------------------------------------


class FakeEvent:
    """A `ready` latch that a test sets after a chosen number of waits."""

    def __init__(self, *, after: int = 0):
        self.after, self.waits, self.flag = after, 0, after == 0

    def set(self) -> None:
        self.flag = True

    def wait(self, timeout=None) -> bool:
        self.waits += 1
        if self.waits >= self.after:
            self.flag = True
        return self.flag


def _gated(cfg, hw, rings, derived, events):
    lch, _ = _launcher(cfg, hw, rings, derived)
    lch.children.ready = events
    for name in events:
        lch.children.handles.setdefault(name, FakeProc(name, []))
    return lch


def test_p0_waits_for_the_servo_loops_before_it_writes_the_homing_ramp(
        rings, cfg, hw, derived):
    """The 2026-08-26 finding, and the reason `_await_publishing` exists.

    P0 spawned P7 and called `home()` ~30 ms later, while P7 needs ~2.8 s to
    import the DDS bindings and load pinocchio's URDF. `write_ramp` does not
    pace inside the ring's capacity, so all 90 ticks were in the ring before P7
    ran one iteration — and its first `next_after` returned a tick 98.9% of the
    way to `home_q`, to be reached in a single segment. Every rollout on record
    did it: still for 2.77 s, then the whole move in ~150 ms at 5.5 rad/s
    against a slew clamp the slew-clamp rule says never binds.
    """
    ready = {"p7_body": FakeEvent(after=4), "p8_hand": FakeEvent(after=2)}
    lch = _gated(cfg, hw, rings, derived, ready)

    lch._await_publishing("p7_body", "p8_hand")

    assert ready["p7_body"].waits == 4, "returned before P7 was on the wire"
    assert ready["p8_hand"].flag


def test_the_gate_refuses_a_servo_loop_that_died_before_it_published(
        rings, cfg, hw, derived):
    """A bad `dds_iface`, a URDF `PinocchioComp` cannot load. No timeout, for
    `warmup`'s reason: the only thing that may end this wait other than success
    is the process being gone. A gate that timed out would resume homing into a
    ring nobody reads — exactly the bug it was added to close."""
    ready = {"p7_body": FakeEvent(after=99)}
    lch = _gated(cfg, hw, rings, derived, ready)
    lch.children.handles["p7_body"].alive = False

    with pytest.raises(ConfigError, match="exited before it set its gate"):
        lch._await_publishing("p7_body")


def test_the_gate_names_the_cause_that_fits_the_child_that_died(rings, cfg, hw,
                                                                derived):
    """A wrong hint costs more than no hint. A servo loop that never reached its
    loop is a DDS or URDF fault; a dashboard that never listened is almost
    always a port already in use, and telling its operator to check `dds_iface`
    sends them to the one place the answer is not."""
    ready = {"p10_dashboard": FakeEvent(after=99)}
    lch = _gated(cfg, hw, rings, derived, ready)
    lch.children.handles["p10_dashboard"].alive = False

    with pytest.raises(ConfigError, match="port already being in use"):
        lch._await_publishing("p10_dashboard")


# --- CHECK's producers: never silent, never waiting on the dead -------------


def _producer_launcher(cfg, hw, rings, derived, logged, name="cam_rwrist"):
    lch, events = _launcher(cfg, hw, rings, derived, log=logged.append)
    lch.children.handles[name] = FakeProc(name, events)
    return lch


def test_check_says_which_producer_it_is_still_waiting_on(rings, cfg, hw,
                                                          derived):
    """A `--camera normal` run printed the PICO banner and then nothing, arm at
    rest. CHECK has no timeout by decision, so the wait itself must say what it
    is waiting on — the full hint once, then a short line, so a live wait is
    not mistaken for a wedge without repeating a paragraph."""
    logged: list[str] = []
    lch = _producer_launcher(cfg, hw, rings, derived, logged)

    lch._await_producer("cam_rwrist", lambda: lch.clock.now() >= T0 + 65.0,
                        "its first frame", "HINT")

    notes = [m for m in logged if "still waiting on cam_rwrist" in m]
    assert len(notes) == 2, "the full note at 30 s, a short line at 60 s"
    assert "its first frame" in notes[0] and "HINT" in notes[0]
    assert "HINT" not in notes[1] and "(60 s)" in notes[1]
    assert logged[-1] == "[P0] cam_rwrist: producing"


def test_a_normal_startup_wait_prints_nothing(rings, cfg, hw, derived):
    """~20 s for the first lowstate sample is a normal fisheye startup; four
    copies of a network diagnosis there read like a fault."""
    logged: list[str] = []
    lch = _producer_launcher(cfg, hw, rings, derived, logged)

    lch._await_producer("cam_rwrist", lambda: lch.clock.now() >= T0 + 25.0,
                        "its first frame", "HINT")

    assert logged == ["[P0] cam_rwrist: producing"]


def test_check_refuses_a_producer_that_died_instead_of_waiting_forever(
        rings, cfg, hw, derived):
    """A P2 that raised — a wrong-shape frame is refused there by design — used
    to leave CHECK polling an arena region nobody would ever write again."""
    logged: list[str] = []
    lch = _producer_launcher(cfg, hw, rings, derived, logged)
    lch.children.handles["cam_rwrist"].alive = False

    with pytest.raises(ConfigError, match="cam_rwrist exited before its first"):
        lch._await_producer("cam_rwrist", lambda: False, "its first frame", "")


@pytest.mark.parametrize("robot, port, server", [
    ("g1d_fisheye_dex1", 55565, "fish_eye_image_server"),
    ("g1d_normal_dex1", 55555, "teleimager"),
])
def test_the_camera_hint_names_the_server_this_rig_is_read_from(
        robot, port, server, cfg):
    """The fix is opposite for the two rigs — start teleimager, or make sure it
    is NOT running — so one generic hint would be wrong for one of them."""
    hw = cfgmod.load_hardware(robot)
    zmq_cfg = cfgmod.replace(
        cfg, cameras={"head": cfgmod.CameraSpec(8, 8, source="zmq")},
        primary_camera="head")
    lch = Launcher(zmq_cfg, hw, log=lambda *_: None)
    hint = lch._camera_hint("head")
    assert f"{hw.camera_host}:{port}" in hint
    assert server in hint


def test_check_waits_on_p1_and_every_camera_by_name():
    """Structural, as above: `check` needs a robot to run. The two waits that
    were silent must go through `_await_producer`."""
    import inspect

    flat = "".join(inspect.getsource(Launcher.check).split())
    assert 'self._await_producer("p1_state",' in flat
    assert 'self._await_producer(f"cam_{cam}",' in flat


def test_check_gates_both_servo_loops_and_waits_after_every_spawn(cfg, hw):
    """Structural, like the P5 claim above: `check` needs DDS rates a unit test
    cannot supply, so the claim is asserted where it is decided.

    The wait must come AFTER the P5 and P9 spawns. P7's ~2.8 s of DDS and
    pinocchio is dead time that should overlap the checkpoint load, exactly as
    the ramp used to — so the gate costs a run nothing it was not already
    paying. It must also be in `check` and not `home`, so that "confirm they
    produce" covers the two servo loops the way it already covers P1.

    Three latches, not two: P10 carries one as well, because START arrives
    through it and a READY prompt nobody can answer is the same class of
    bug as a homing ramp written into a ring nobody reads.
    """
    import inspect

    src = inspect.getsource(Launcher.check)
    assert src.count("gated=True") == 3, \
        "both servo loops and the dashboard carry a ready latch"
    assert src.index('"p5_policy"') < src.index("self._await_publishing(")
    assert "self._await_publishing(" not in inspect.getsource(Launcher.home)


# --- The exit sequence ------------------------------------------


def _launcher(cfg, hw, rings, derived, *, clock=None, events=None, log=None):
    events = events if events is not None else []
    lch = Launcher(cfg, hw, clock=clock or FakeClock(),
                   lowcmd=lambda h: FakeWire(h, events),
                   log=log or (lambda *_: None))
    lch.rings = rings
    lch.derived = derived
    lch.children.handles = {name: FakeProc(name, events) for name in
                            ("p5_policy", "p6_target", "p7_body", "p8_hand")}
    return lch, events


def test_the_exit_ramp_starts_two_ticks_ahead_of_the_cursor(rings, cfg, hw,
                                                            derived):
    """`deadline(k_exec + 2) = now + 2·dt` — P7 is already committed to the
    segment ending at deadline(k_exec), so the first tick P0 may own is two
    out."""
    control = rings["control"]
    control.k_exec = 5
    # One row, because a tick's joints and its hand command are now the same
    # record — the hand column is what the exit ramp seeds from.
    rings["targets"].write(5, T0, _q(0.2),              # the commanded posture
                           np.full((schema.N_HANDS, schema.HAND_DIM), 3.0), intervene=0)
    clock = FakeClock()
    # A short floor keeps the whole ramp inside the ring, so the test can read
    # it back; the ring is 16 slots and a 3 s ramp at 30 Hz wraps it five times.
    short = cfgmod.replace(cfg, home_min_s=0.1)
    lch, _ = _launcher(short, hw, rings, derived, clock=clock)

    written = lch._exit_ramp()

    assert [k for k, _ in written] == list(range(7, 7 + len(written)))
    assert rings["body"].get(6) is None                 # nothing at k_exec + 1
    assert written[0][1] == pytest.approx(T0 + 2 * cfg.dt)
    first = rings["body"].get(7)
    # It ramps from the commanded posture, not from the measurement.
    assert 0.0 < first.payload[0] < 0.2
    last = written[-1][0]
    assert np.allclose(rings["body"].get(last).payload, 0.0)     # at exit_q
    # The hand ramps too, seeded from the last COMMANDED aperture rather than
    # the measurement: a run that ends mid-grasp is holding a jaw against an
    # object, so the measurement lags the command by whatever the object is
    # worth and starting there would step.
    first_hand = rings["hand"].get(7).payload[0, 0]
    assert 0.0 < first_hand < 3.0, "the hand stepped instead of ramping"
    assert np.allclose(rings["hand"].get(last).payload, 0.0)     # at exit_hand


def test_the_damp_is_published_after_p7_stops(rings, cfg, hw, derived):
    """The exit sequence in order: P5 and P6 stop writing, the ramp executes, P7 and P8
    stop, and only then the damp — which is the one thing that actually stops
    the firmware."""
    rings["control"].k_exec = 3
    rings["targets"].write(3, T0, _q(0.0), intervene=0)
    lch, events = _launcher(cfg, hw, rings, derived)

    lch.exit()

    assert events == ["stop:p5_policy", "stop:p6_target",
                      "stop:p7_body", "stop:p8_hand", "connect", "damp"]
    assert rings["control"].session == schema.STOPPED


def test_a_second_interrupt_skips_the_ramp_and_damps(rings, cfg, hw, derived):
    """An immediate damp drops the arm into gravity, so it is the operator's
    explicit second decision — but once made it is not negotiable."""
    rings["control"].k_exec = 3
    lch, events = _launcher(cfg, hw, rings, derived)
    lch._skip_ramp = True

    lch.exit()

    assert rings["body"].latest_below(1 << 30) is None      # no ramp written
    assert events[-1] == "damp"
    assert "publish_body" not in events


def test_nothing_is_published_after_the_damp_latches(rings, cfg, hw, derived):
    """The latch is set before the publish, not after it, so no second
    path can slip a command in behind it."""
    lch, events = _launcher(cfg, hw, rings, derived)
    lch._damp()
    lch._damp()
    lch._damp()
    assert events == ["connect", "damp"]


def test_exit_is_a_no_op_without_an_arena(cfg, hw):
    """A refusal at INIT must not try to damp a robot nothing ever connected
    to."""
    Launcher(cfg, hw, log=lambda *_: None).exit()


# --- shutdown: the session word before the signal ----------


def test_stopped_is_published_before_the_recorder_is_terminated(rings, cfg, hw,
                                                                derived):
    """P9 loops on `session < STOPPED` and writes `manifest.json` from its own
    `finally`. SIGTERM runs no `finally`, so terminating it first discarded
    every stream still inside the 1 MB write buffer — `chunks.bin` among them,
    which is the only record of what each inference cost."""
    lch, events = _launcher(cfg, hw, rings, derived)
    p9 = RecorderProc(rings["control"], events)
    lch.children.handles["p9_record"] = p9

    lch.exit()

    assert p9.saw_session == schema.STOPPED
    assert "p9_closed" in events
    assert "stop:p9_record" not in events       # it left on its own


def test_the_recorder_closes_after_the_damp_not_before(rings, cfg, hw, derived):
    """The damp is the last thing that happens to the robot, so it is the last
    thing worth recording. STOPPED goes out after it, never instead of it."""
    lch, events = _launcher(cfg, hw, rings, derived)
    lch.children.handles["p9_record"] = RecorderProc(rings["control"], events)

    lch.exit()

    assert events.index("damp") < events.index("p9_closed")


def test_a_recorder_that_will_not_exit_is_still_terminated(rings, cfg, hw,
                                                           derived):
    """The wait is bounded. A wedged child must not keep the arena mapped and
    the launcher alive — and the operator is told what it cost."""
    lines = []
    lch, events = _launcher(cfg, hw, rings, derived, log=lines.append)
    lch.children.handles["p9_record"] = RecorderProc(rings["control"], events,
                                                     obeys=False)

    lch.exit()

    assert "stop:p9_record" in events
    assert any("p9_record" in line and "buffered" in line for line in lines)


# --- The end-effector stream ------------------------------


def _write_state(rings, *, t, t_grip):
    rings["state"].write(t_recv=t, q=np.zeros(schema.N_JOINTS),
                         dq=np.zeros(schema.N_JOINTS),
                         flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                         grip_meas=np.zeros(schema.GRIP_DIM),
                         t_grip_recv=t_grip)


def test_a_silent_gripper_topic_is_a_startup_refusal(rings, cfg, hw, derived):
    """`t_grip_recv = 0` is P1's "never received", and it is indistinguishable
    downstream from a gripper that is closed and holding still. Left to run, it
    pins this family's entire proprioceptive input to a constant."""
    lch, _ = _launcher(cfg, hw, rings, derived)
    _write_state(rings, t=T0, t_grip=0.0)

    with pytest.raises(ConfigError, match="t_grip_recv"):
        lch._verify_hand_state()


def test_a_live_gripper_topic_passes_the_check(rings, cfg, hw, derived):
    """One real arrival is the whole test — this stream has no declared rate to
    measure against, unlike lowstate."""
    lch, _ = _launcher(cfg, hw, rings, derived)
    _write_state(rings, t=T0, t_grip=T0 - 0.001)

    lch._verify_hand_state()            # must not raise


def test_the_refusal_names_the_topic_and_the_domain(rings, cfg, hw, derived):
    """The message is the diagnosis. lowstate arriving proves the domain but
    not these topics, so both have to be in front of whoever reads it — and
    BOTH grippers' topics, since P1 pairs them and the refusal cannot say which
    one went quiet."""
    lch, _ = _launcher(cfg, hw, rings, derived)
    _write_state(rings, t=T0, t_grip=0.0)

    with pytest.raises(ConfigError) as exc:
        lch._verify_hand_state()
    for topic in hw.topic_hand_state:
        assert topic in str(exc.value)
    assert hw.dds_iface in str(exc.value)


# --- transit drops, reported at CHECK ---------------------------------------


class _Counters:
    """A camera region stub exposing only what `_drop_note` reads."""

    def __init__(self, counters):
        self._counters = counters

    def peek_counters(self):
        return self._counters


def _note(before, after):
    return Launcher._drop_note("rwrist", before, _Counters(after))


def test_check_reports_frames_lost_between_the_server_and_p2():
    """`frame_id` counts what P2 received, `source_seq` what the board sent.
    Their difference over the window is what ZMQ discarded at its high-water
    mark — silently, by design, since the newest frame wins."""
    # 100 sent, 90 received.
    assert "10/100 frames dropped" in _note((0, 0), (90, 100))


def test_check_says_so_when_nothing_was_dropped():
    assert "no drops" in _note((0, 0), (100, 100))


def test_check_is_silent_for_a_source_that_publishes_no_counter():
    """The vendor `image_server` sends a bare JPEG and the fakes have no
    counter. A fabricated zero would read as 'measured, and fine'."""
    assert _note((0, -1), (100, -1)) == ""


def test_check_is_silent_when_a_camera_had_not_published_yet():
    assert _note(None, (100, 100)) == ""
    assert _note((0, 0), None) == ""


# --- The board image server, started only when something needs it ------------


class _NoSSH:
    """A launcher stub carrying just what `_ensure_board_imager` reads."""

    def __init__(self, cfg, host="192.168.123.164", image_server="fisheye"):
        self.cfg = cfg
        self.hw = type("_HW", (), {"camera_host": host,
                                   "camera_ports": {"rwrist": 55557},
                                   "image_server": image_server,
                                   "image_server_dir": "/srv/fe",
                                   "image_server_conda": "/opt/conda",
                                   "image_server_conda_env": "tv"})()
        self.logged = []

    def log(self, msg):
        self.logged.append(msg)


def _cameras(**specs):
    return type("_Cfg", (), {"cameras": specs})()


def test_a_usb_only_rig_never_reaches_for_the_robot_board(monkeypatch):
    """A rig whose cameras are all on this machine must not need SSH to the
    board to be able to start — reaching for it would make an unrelated network
    fault a refusal for a run that does not depend on it."""
    import elrobot_deploy.runtime.drivers.board as board

    def explode(*a, **k):                       # noqa: ANN002, ANN003
        raise AssertionError("SSHed to the board for a USB-only rig")

    monkeypatch.setattr(board, "ensure_running", explode)
    stub = _NoSSH(_cameras(
        exo=cfgmod.CameraSpec(8, 8, source="realsense"),
        rwrist=cfgmod.CameraSpec(8, 8, source="v4l2", device="/dev/x")))
    Launcher._ensure_board_imager(stub)
    assert stub.logged == []


def test_a_zmq_camera_starts_the_board_server_before_p2_connects(monkeypatch):
    """P2's first act is to connect a SUB socket and block. ZMQ connects to an
    unbound port quite happily, so a server nobody started reads as a dead
    camera, not as a missing server."""
    import elrobot_deploy.runtime.drivers.board as board

    seen = {}

    def fake(host, **kwargs):
        seen["host"] = host
        seen.update(kwargs)
        return False                            # we started it

    monkeypatch.setattr(board, "ensure_running", fake)
    stub = _NoSSH(_cameras(rwrist=cfgmod.CameraSpec(8, 8, source="zmq")))
    Launcher._ensure_board_imager(stub)
    assert seen["host"] == "192.168.123.164"
    # Where the server lives on the board is the robot file's [image_server].
    assert (seen["remote_dir"], seen["conda_root"], seen["conda_env"]) == (
        "/srv/fe", "/opt/conda", "tv")
    assert "started" in stub.logged[0]


def test_camera_normal_never_touches_the_board(monkeypatch):
    """`--camera normal` is a robot with the stock wrists and teleimager serving
    them. The autostart STOPS teleimager — the only server those cameras have —
    so it must not run, even though the cameras are zmq ones."""
    import elrobot_deploy.runtime.drivers.board as board

    def explode(*a, **k):                       # noqa: ANN002, ANN003
        raise AssertionError("SSHed to the board under --camera normal")

    monkeypatch.setattr(board, "ensure_running", explode)
    stub = _NoSSH(_cameras(rwrist=cfgmod.CameraSpec(8, 8, source="zmq")),
                  image_server="teleimager")
    Launcher._ensure_board_imager(stub)
    assert len(stub.logged) == 1 and "teleimager" in stub.logged[0]


# --- The board process pattern ----------------------------------------------


def test_the_process_pattern_cannot_match_the_shell_that_runs_it():
    """MEASURED on the board: `pgrep -f` scans EVERY process's command line,
    including the `bash -lc` wrapper running the pgrep, whose argv contains the
    pattern verbatim. Unanchored, `_is_running()` answered True forever — so
    `ensure_running` would never start the server and `stop()` would kill its
    own shell.

    The publisher's argv begins with `python`; every shell wrapping it begins
    with `bash`. The `^` anchor is the entire fix, so it is pinned here.
    """
    import re

    from elrobot_deploy.runtime.drivers.board.image_server_launch import PROCESS_PATTERN

    pattern = re.compile(PROCESS_PATTERN)
    assert pattern.search("python publisher.py")
    assert pattern.search("python3.10 publisher.py")
    # The shapes that must NOT match: the launcher, and an ssh/pgrep wrapper.
    assert not pattern.search(
        "bash -lc source /home/unitree/miniconda3/etc/profile.d/conda.sh && "
        "conda activate tv && cd /home/unitree/fish_eye_image_server && "
        "nohup python publisher.py > /tmp/fish_eye_image_server.log 2>&1")
    assert not pattern.search(
        "bash -lc pgrep -f '^python[0-9.]* publisher\\.py' || true")


def test_zero_does_not_mean_skip_the_board_autostart(monkeypatch):
    """`"0"` is a non-empty string and therefore truthy, so a bare truth test
    made ELROBOT_NO_BOARD_AUTOSTART=0 mean "skip" — the opposite of what anyone
    typing 0 intends — and then reported the board as "already running" without
    having looked at it."""
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    for falsy in ("0", "false", "no", "off", "", "  "):
        monkeypatch.setenv(imager.SKIP_ENV, falsy)
        with pytest.raises(imager.BoardImagerError, match="could not SSH"):
            imager.ensure_running("192.0.2.1", connect_timeout=0.05, **_WHERE)

    for truthy in ("1", "true", "yes"):
        monkeypatch.setenv(imager.SKIP_ENV, truthy)
        assert imager.ensure_running("192.0.2.1", **_WHERE) is True   # skipped


_WHERE = {"remote_dir": "/home/unitree/fish_eye_image_server",
          "conda_root": "/home/unitree/miniconda3", "conda_env": "tv"}


def test_the_image_server_location_comes_from_the_robot_file(tmp_path):
    """No board path is hard-coded: a fisheye `[image_server]` must say where
    the server lives, and the loader carries it into `Hardware`."""
    hw = cfgmod.load_hardware("g1d_fisheye_dex1")
    assert (hw.image_server, hw.image_server_dir, hw.image_server_conda,
            hw.image_server_conda_env) == (
        "fisheye", "/home/unitree/fish_eye_image_server",
        "/home/unitree/miniconda3", "tv")
    assert cfgmod.load_hardware("g1d_normal_dex1").image_server == "teleimager"
    f = tmp_path / "r.toml"
    f.write_text('extends = "g1d_normal_dex1"\n[image_server]\ntype = "fisheye"\n')
    with pytest.raises(ConfigError, match="dir"):
        cfgmod.load_hardware(str(f))
    g = tmp_path / "g.toml"
    g.write_text('extends = "g1d_normal_dex1"\n[board]\nimage_server = "teleimager"\n')
    with pytest.raises(ConfigError, match="image_server"):
        cfgmod.load_hardware(str(g))


# --- stopping the vendor image server ---------------------------------------


class _FakeStdout:
    def __init__(self, text=""):
        self._text = text

    def read(self):
        return self._text.encode()


class FakeSSH:
    """Records what `_run` sends, and scripts `systemctl is-active`.

    `active_for` is how many `is-active` polls answer "active" before the unit
    reports gone — which is the only signal `_stop_vendor` acts on.
    """

    def __init__(self, active_for: int, state: str = "active"):
        self.commands: list[str] = []
        self.active_for = active_for
        self.state = state
        self.asked = 0

    def exec_command(self, command, timeout=None):
        self.commands.append(command)
        out = ""
        if "is-active" in command:
            self.asked += 1
            out = self.state if self.asked <= self.active_for else "failed"
        return None, _FakeStdout(out), _FakeStdout()


def test_the_vendor_stop_never_blocks_on_systemds_own_timeout():
    """The bug this replaced: a plain `systemctl stop teleimager` takes NINETY
    seconds and was given twenty, so it failed on every single launch.

    The unit's `ExecStart` is a compound command, so bash cannot exec into the
    server and the MainPID is bash; `teleimager-server` never exits on the
    SIGTERM, and systemd waits out `TimeoutStopSec` before the SIGKILL that was
    always going to be needed. Measured from the board's journal 2026-09-02:
    `Stopping` at 10:00:10, `Failed with result 'timeout'` at 10:01:40.
    """
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    ssh = FakeSSH(active_for=2)
    imager._stop_vendor(ssh, "123", timeout=10.0)
    sent = "\n".join(ssh.commands)

    assert "systemctl stop --no-block teleimager" in sent
    # The blocking form is the whole defect; it must not come back.
    assert "sudo -S systemctl stop teleimager" not in sent


def test_the_vendor_is_stopped_before_it_is_killed():
    """Order, and it is not cosmetic. `teleimager` is `Restart=always` with
    `RestartSec=5`, so a kill on its own buys five seconds and then loses the
    cameras again — the board's journal shows exactly that, stopped at 10:00:05
    and back up by 10:00:10. Only a manual stop cancels the restart, so the
    stop must be in flight before anything is killed."""
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    ssh = FakeSSH(active_for=2)
    imager._stop_vendor(ssh, "123", timeout=10.0)

    stop_at = next(i for i, c in enumerate(ssh.commands)
                   if "stop --no-block" in c)
    kills = [i for i, c in enumerate(ssh.commands) if "kill -s KILL" in c]
    assert kills, "the SIGTERM is what the server ignores; a kill must follow"
    assert min(kills) > stop_at


def test_a_vendor_that_is_already_gone_costs_one_poll_and_no_kill():
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    ssh = FakeSSH(active_for=0)
    imager._stop_vendor(ssh, "123", timeout=10.0)
    assert not [c for c in ssh.commands if "kill -s KILL" in c]


def test_a_vendor_that_never_dies_raises_rather_than_hanging():
    """The ceiling is ours now, not systemd's — but there still is one, and it
    names the unit rather than reporting a bare socket timeout."""
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    ssh = FakeSSH(active_for=10_000)
    with pytest.raises(imager.BoardImagerError, match="still active"):
        imager._stop_vendor(ssh, "123", timeout=0.2)


def test_a_vendor_mid_restart_still_counts_as_holding_the_cameras():
    """`is-active` is not a yes/no. teleimager is `Restart=always` with
    `RestartSec=5` and crash-loops on this board — sampled 2026-09-02 it read
    `active` -> `activating` -> `active` on a five second cycle, because systemd
    reports the restart wait as `activating`.

    `== "active"` therefore answers False for half of every cycle. Land there
    and the stop is skipped, our publisher starts, and teleimager returns
    seconds later to take the devices and `modprobe -r uvcvideo` on its way in.
    """
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    for holding in ("active", "activating", "deactivating", "reloading"):
        assert imager._vendor_active(FakeSSH(active_for=1, state=holding)), holding
    for gone in ("inactive", "failed", "unknown", ""):
        assert not imager._vendor_active(FakeSSH(active_for=1, state=gone)), gone


def test_a_vendor_stuck_activating_is_still_killed():
    """The consequence of the above for `_stop_vendor`: it polls the same
    predicate, so a unit looping through `activating` must not read as done."""
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    ssh = FakeSSH(active_for=2, state="activating")
    imager._stop_vendor(ssh, "123", timeout=10.0)
    assert [c for c in ssh.commands if "kill -s KILL" in c]


# --- keeping the board's copy of the server current -------------------------


class FakeBoard:
    """The board as `ensure_running` sees it: the server directory's files, the
    publisher's pids, the stamp, and SFTP. Commands are matched by substring,
    as `FakeSSH` does."""

    def __init__(self, files: dict[str, bytes] | None, *, pids=(), stamp=None,
                 drop_writes: bool = False):
        self.files = files                  # None: the directory does not exist
        self.pids = list(pids)
        self.stamp = stamp
        self.drop_writes = drop_writes
        self.puts: list[str] = []
        self.kills = 0
        self.launches = 0
        self.commands: list[str] = []

    def exec_command(self, command, timeout=None):
        import hashlib
        import re

        self.commands.append(command)
        out = ""
        if "is-active" in command:
            out = "inactive"
        elif "sha256sum" in command:
            out = "\n".join(f"{hashlib.sha256(b).hexdigest()}  {n}"
                            for n, b in sorted((self.files or {}).items()))
        elif "mkdir -p" in command:
            self.files = {} if self.files is None else self.files
        elif "nohup python publisher.py" in command:
            self.launches += 1
            self.pids = [str(1000 + self.launches)]
        elif "pgrep -f" in command:
            out = " ".join(self.pids)
        elif "cat /tmp/fish_eye_image_server.stamp" in command:
            out = self.stamp or ""
        elif "> /tmp/fish_eye_image_server.stamp" in command:
            m = re.search(r"echo (\S+) (\S+) >", command)
            self.stamp = f"{m.group(1)} {m.group(2)}"
        elif re.search(r"\bkill \d+", command):
            self.kills += 1
            self.pids = []
        return None, _FakeStdout(out), _FakeStdout()

    def open_sftp(self):
        board = self

        class _SFTP:
            def put(self, local, remote):
                board.puts.append(remote)
                self.pending = (remote, Path(local).read_bytes())

            def posix_rename(self, src, dst):
                remote, data = self.pending
                assert remote == src and src.endswith(".g1d-tmp")
                if not board.drop_writes:
                    board.files[dst.rsplit("/", 1)[1]] = data

            def close(self):
                pass

        return _SFTP()

    def close(self):
        pass


def _ours():
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    return {n: (imager.LOCAL_DIR / n).read_bytes()
            for n in imager.local_manifest()}


def _ensure(monkeypatch, board):
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    monkeypatch.delenv(imager.SKIP_ENV, raising=False)
    monkeypatch.setattr(imager, "_connect", lambda *a, **k: board)
    monkeypatch.setattr(imager.time, "sleep", lambda s: None)
    said: list[str] = []
    return imager.ensure_running("192.0.2.1", log=said.append, **_WHERE), said


def _digest():
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    return imager.manifest_digest(imager.local_manifest())


def test_the_manifest_is_the_code_the_board_runs():
    """Every file the publisher imports or reads, and nothing it does not."""
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    names = set(imager.local_manifest())
    assert {"photometry.py", "publisher.py", "cameras.yaml"} <= names
    assert "README.md" not in names


def test_a_stale_board_copy_is_replaced_and_its_publisher_restarted(monkeypatch):
    """THE BUG THIS EXISTS FOR, measured 2026-09-30: the board ran the Sep-21
    `photometry.py` (the gamma/under-relax law) a week after the fix merged,
    because nothing ever copied it over. Its live log flipped the right wrist
    between gain 61 and 96, p99.5 194 <-> 255 at 30% clipped, every tick."""
    stale = dict(_ours(), **{"photometry.py": b"UNDER_RELAX = 0.70\n"})
    board = FakeBoard(stale, pids=["11470"], stamp=None)

    was_running, said = _ensure(monkeypatch, board)

    assert board.files == _ours()
    assert board.puts == ["/home/unitree/fish_eye_image_server/photometry.py.g1d-tmp"]
    assert board.kills == 1 and board.launches == 1 and was_running is False
    assert board.stamp == f"1001 {_digest()}"
    assert "photometry.py" in said[0] and "restarting" in said[1]


def test_matching_code_on_disk_is_not_enough_if_the_process_predates_it(monkeypatch):
    """What the board looked like right after the 2026-09-30 hand copy: every
    file identical, and the publisher from before it still serving the old law
    from memory. No stamp, so it is restarted — without copying anything."""
    board = FakeBoard(_ours(), pids=["11470"], stamp=None)

    was_running, said = _ensure(monkeypatch, board)

    assert board.puts == [] and board.kills == 1 and board.launches == 1
    assert was_running is False and len(said) == 1 and "restarting" in said[0]


def test_a_stamp_from_another_process_does_not_vouch_for_this_one(monkeypatch):
    """A publisher restarted by hand after a stamped launch: right digest,
    wrong pid. The stamp describes a process that is gone."""
    board = FakeBoard(_ours(), pids=["2222"], stamp=f"1111 {_digest()}")

    was_running, _ = _ensure(monkeypatch, board)

    assert board.kills == 1 and board.launches == 1 and was_running is False


def test_a_current_board_is_left_exactly_alone(monkeypatch):
    """Same files, and the running publisher is the one stamped with this
    code: no copy, no kill, no launch, nothing said."""
    board = FakeBoard(_ours(), pids=["1001"], stamp=f"1001 {_digest()}")

    was_running, said = _ensure(monkeypatch, board)

    assert was_running is True and said == []
    assert board.puts == [] and board.kills == 0 and board.launches == 0


def test_an_empty_board_is_installed_then_started(monkeypatch):
    """No directory at all used to be a refusal that printed an rsync."""
    board = FakeBoard(None)

    was_running, _ = _ensure(monkeypatch, board)

    assert board.files == _ours() and board.launches == 1 and board.kills == 0
    assert was_running is False and board.stamp == f"1001 {_digest()}"


def test_a_copy_that_does_not_land_is_an_error_not_a_start(monkeypatch):
    """Verified by hashing on the board again: a full disk or a permissions
    problem must not end in a publisher running whatever was left there."""
    from elrobot_deploy.runtime.drivers.board import image_server_launch as imager

    board = FakeBoard(dict(_ours(), **{"photometry.py": b"old\n"}),
                      pids=["11470"], drop_writes=True)

    with pytest.raises(imager.BoardImagerError, match="still differs"):
        _ensure(monkeypatch, board)
    assert board.kills == 0 and board.launches == 0


# --- reset ------------------------------------------------------


class RequestClock(FakeClock):
    """Plants a request after `after` sleeps — the operator arriving late.

    A poll loop that returns on its first pass proves nothing about the loop:
    the request has to arrive while P0 is already waiting.
    """

    def __init__(self, control, request, *, after: int = 1):
        super().__init__()
        self.control, self.request, self.after = control, request, after

    def sleep_until(self, t: float, **_) -> float:
        self.after -= 1
        if self.after == 0:
            self.control.request = self.request
        _play_p5(self.control)
        return super().sleep_until(t)


def _play_p5(control, *, refuse_prompt: str | None = None,
             fail_settings: bool = False) -> None:
    """What P5's `RolloutLatch` does between two dummies: judge a
    pending prompt request, and take a new rollout generation."""
    seq, text = control.prompt_request()
    if seq and control.verdict()[0] != seq:
        if refuse_prompt is not None and refuse_prompt in text:
            control.judge(seq, schema.PROMPT_REFUSED, "too many tokens")
        else:
            control.judge(seq, schema.PROMPT_ACCEPTED)
    gen = control.rollout_generation
    if gen and (control.settings_applied & ~schema.APPLIED_FAILED) != gen:
        control.settings_applied = (gen | schema.APPLIED_FAILED if fail_settings
                                    else gen)


def _spawning_launcher(cfg, hw, rings, derived, *, clock=None, on_spawn=None):
    """A launcher that can START children, not only stop them.

    `_launcher` plants handles directly, which is all `exit` needs — it never
    spawns. `reset` does, so this one supplies a spawn and a latch.
    """
    events: list[str] = []

    def spawn(name, dotted, args, kwargs):
        if on_spawn is not None:
            on_spawn(name)
        events.append(f"start:{name}")
        return FakeProc(name, events)

    # `after=0` is a latch that is already set: `_await_publishing` has its own
    # tests above, and here it would only ever wait on a process that is a stub.
    lch = Launcher(cfg, hw, clock=clock or FakeClock(), spawn=spawn,
                   event=lambda: FakeEvent(after=0),
                   lowcmd=lambda h: FakeWire(h, events),
                   log=lambda *_: None)
    lch.rings = rings
    lch.derived = derived
    lch._seed_settings()
    lch.children.handles = {name: FakeProc(name, events) for name in
                            ("p1_state", "p5_policy", "p6_target", "p7_body",
                             "p8_hand", "p9_record")}
    return lch, events


def test_reset_restarts_the_interpolators_and_never_touches_the_policy(
        rings, cfg, hw, derived):
    """The whole point of reset: a second rollout must not reload a checkpoint.

    P5 is what a relaunch is expensive for — tens of seconds of load and XLA
    compile — so the one thing reset may never do is stop it. P1, P9 and the
    cameras stay for their own reasons: P5 goes on warming against live
    observations, and `home` needs a measured posture the instant it is called.
    """
    lch, events = _spawning_launcher(cfg, hw, rings, derived)
    _measured(rings, hw, np.zeros(schema.BODY_DOF))     # already at home_q

    lch.reset()

    for name in ("p6_target", "p7_body", "p8_hand"):
        assert f"stop:{name}" in events, name
    for name in ("p7_body", "p8_hand"):
        assert f"start:{name}" in events, name

    for name in ("p5_policy", "p1_state", "p9_record"):
        assert f"stop:{name}" not in events, name
        assert f"start:{name}" not in events, name

    # P6 is `active`'s to spawn. A reset ends at READY, and a target
    # process running there would be solving against a policy nobody started.
    assert "start:p6_target" not in events


def test_reset_zeroes_the_cursor_before_a_new_p7_exists(rings, cfg, hw,
                                                        derived):
    """`k_exec` is P7's word, so reset may store it only in the window
    where P7 is dead. By the time p7_body is spawned it must already read 0 —
    otherwise the new P7 comes up onto the cursor the old rollout left."""
    control = rings["control"]
    control.k_exec = 50_000
    seen: dict[str, int] = {}
    lch, _ = _spawning_launcher(
        cfg, hw, rings, derived,
        on_spawn=lambda name: seen.setdefault(name, control.k_exec))
    _measured(rings, hw, np.zeros(schema.BODY_DOF))

    lch.reset()

    assert seen["p7_body"] == 0, "P7 was spawned onto the old rollout's cursor"
    assert seen["p8_hand"] == 0


def test_reset_wipes_the_previous_rollout_out_of_the_target_rings(
        rings, cfg, hw, derived):
    """"Everything about the previous rollout is gone" is the operator's whole
    mental model of Reset, and `TargetRing.clear` is what makes it true.

    Stale slots would not in fact break P7 — `next_after` selects on deadline
    and theirs are all in the past — but a ring still holding a rollout that
    ended is one the dashboard would draw and a reader would have to reason
    about.
    """
    rings["targets"].write(50_000, T0, _q(0.4),
                           np.full((schema.N_HANDS, schema.HAND_DIM), 0.4), intervene=0)
    lch, _ = _spawning_launcher(cfg, hw, rings, derived)
    _measured(rings, hw, np.zeros(schema.BODY_DOF))

    lch.reset()

    assert rings["body"].get(50_000) is None
    assert rings["hand"].get(50_000) is None
    # And homing really ran on top of the wipe: homing's ramp ends at tick 0.
    assert rings["body"].get(0) is not None


def test_reset_leaves_the_session_alive_and_the_cursor_at_zero(rings, cfg, hw,
                                                               derived):
    """Reset ends in HOME; `ready` publishes READY on the next pass of the
    rollout loop. What matters here is that it did NOT end in EXITING — the
    session is alive, and the arm is standing at home_q."""
    lch, _ = _spawning_launcher(cfg, hw, rings, derived)
    _measured(rings, hw, np.zeros(schema.BODY_DOF))

    lch.reset()

    assert rings["control"].session == schema.HOME
    assert rings["control"].k_exec == 0


# --- The request word -------------------------------------------


def test_a_request_is_consumed_by_the_process_that_acts_on_it(rings, cfg, hw,
                                                              derived):
    """P0 is the request's consumer, not a second observer of it. A request left
    in the word would be taken again on the next pass of whichever loop is
    polling — one press of Reset would reset forever."""
    lch, _ = _spawning_launcher(cfg, hw, rings, derived)
    rings["control"].request = schema.REQ_RESET

    assert lch._take_request() == schema.REQ_RESET
    assert rings["control"].request == schema.REQ_NONE
    assert lch._take_request() == schema.REQ_NONE


def test_ready_waits_for_start_and_a_reset_there_is_a_no_op(rings, cfg, hw,
                                                            derived):
    """READY is a poll now, not `input()`: a process asleep in a stdin read
    polls nothing, and START arrives through the arena.

    A RESET arriving at READY asks for the state P0 is already in. It is
    consumed and ignored rather than refused — there is nothing to refuse.
    """
    control = rings["control"]
    control.request = schema.REQ_RESET
    clock = RequestClock(control, schema.REQ_START, after=3)
    lch, _ = _spawning_launcher(cfg, hw, rings, derived, clock=clock)

    lch.ready()                                 # returns only on START

    assert control.session == schema.READY
    assert control.request == schema.REQ_NONE
    assert len(clock.sleeps) >= 3, "returned without ever waiting"


def test_exit_from_the_dashboard_takes_the_same_path_as_the_first_ctrl_c(
        rings, cfg, hw, derived):
    """One implementation of the exit sequence, reached two ways. `_ExitNow` is the first
    Ctrl-C's exception, and an EXIT request raises the same one, so the shutdown
    sequence cannot drift between the terminal and the dashboard."""
    for phase in ("ready", "active"):
        control = rings["control"]
        control.request = schema.REQ_EXIT
        lch, _ = _spawning_launcher(cfg, hw, rings, derived)

        with pytest.raises(_ExitNow):
            getattr(lch, phase)()

        assert control.request == schema.REQ_NONE, phase


def test_active_returns_on_reset_so_a_rollout_is_not_a_process_lifetime(
        rings, cfg, hw, derived):
    """`active` used to be a `while True` that only an interrupt could leave,
    which is exactly what made one rollout the life of one process."""
    control = rings["control"]
    clock = RequestClock(control, schema.REQ_RESET, after=2)
    lch, events = _spawning_launcher(cfg, hw, rings, derived, clock=clock)

    lch.active()                                # returns, does not raise

    assert control.session == schema.ACTIVE
    assert "start:p6_target" in events          # the run really was handed over
    assert control.request == schema.REQ_NONE


# --- CHECK refuses a NIC that cannot hear the robot -------------------------


def test_check_refuses_before_p1_when_the_nic_holds_no_robot_subnet_address(
        rings, cfg, hw, derived, monkeypatch):
    """The wait on p1_state cannot succeed without an address on 192.168.123.0/24,
    so CHECK says so at once — with the nmcli line — instead of waiting forever
    on a robot that pings fine (ssh routes by IP; DDS binds the NIC's address)."""
    from elrobot_deploy.runtime.drivers.dds import domain as dds

    monkeypatch.setattr(dds, "has_nic", lambda iface: True)
    monkeypatch.setattr(dds, "ipv4_addresses", lambda iface: ["192.168.1.111"])
    monkeypatch.setattr(dds, "nm_connection", lambda iface: "Wired connection 1")
    monkeypatch.setattr(os, "name", "posix")
    lch, _ = _launcher(cfg, hw, rings, derived)

    with pytest.raises(ConfigError, match="192.168.1.111") as exc:
        lch._verify_robot_subnet()
    assert 'nmcli con mod "Wired connection 1"' in str(exc.value)


def test_check_passes_the_subnet_gate_without_pinging(rings, cfg, hw, derived,
                                                      monkeypatch):
    """A robot still booting is a reason to wait, not to refuse: the gate reads
    the NIC only, and the ping belongs to the wait's note."""
    from elrobot_deploy.runtime.drivers.dds import domain as dds

    monkeypatch.setattr(dds, "has_nic", lambda iface: True)
    monkeypatch.setattr(dds, "ipv4_addresses", lambda iface: ["192.168.123.118"])
    monkeypatch.setattr(dds, "answers", lambda robot, src: pytest.fail("pinged"))
    monkeypatch.setattr(os, "name", "posix")
    logged: list[str] = []
    lch, _ = _launcher(cfg, hw, rings, derived, log=logged.append)

    lch._verify_robot_subnet()
    assert any("binds 192.168.123.118" in m for m in logged)


def test_the_p1_hint_is_evaluated_when_the_note_fires(rings, cfg, hw, derived):
    """The answer can change during the wait (the robot gets switched on), so
    a callable hint is asked when the note fires, not frozen at the start."""
    logged: list[str] = []
    lch = _producer_launcher(cfg, hw, rings, derived, logged)
    asked_at = []

    def hint():
        asked_at.append(lch.clock.now())
        return "robot silent"

    lch._await_producer("cam_rwrist", lambda: lch.clock.now() >= T0 + 35.0,
                        "its first frame", hint)

    notes = [m for m in logged if "still waiting on cam_rwrist" in m]
    assert "robot silent" in notes[0]
    assert asked_at and asked_at[0] >= T0 + 30.0

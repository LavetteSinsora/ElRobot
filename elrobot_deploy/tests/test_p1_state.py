"""P1 — the state reader.

The failures worth catching here are the ones a real run cannot show you. P1
produces plausible numbers under every one of them: a `t_recv` stamped after
the FK is only ~0.5 ms late and nothing reports it; a flange computed from a
zeroed waist is wrong by centimetres and looks like a badly tracking policy; a
gripper topic that died an hour ago still fills `grip_meas` with its last value.
So each test below pins the *relationship* between fields of one record, not
the field values.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from elrobot_deploy.runtime.processes import p1_state
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.timing import now
from elrobot_deploy.robot.kinematics import ForwardKinematics
from elrobot_deploy.runtime.drivers.robot_state import FakeHandState, FakeLowState


class RecordingFK:
    """Stands in for the mujoco FK: records what it was asked and when it ran.

    `delay` makes the FK expensive enough that a stamp taken after it is
    unmistakably later than one taken before — the difference the loop's
    ordering is about is otherwise tens of microseconds.
    """

    def __init__(self, delay: float = 0.0):
        self.delay = float(delay)
        self.spans: list[tuple[float, float]] = []
        self.args: list[tuple[np.ndarray, np.ndarray]] = []

    def flange(self, arm_q14, waist_q3):
        t0 = now()
        if self.delay:
            time.sleep(self.delay)
        self.args.append((np.array(arm_q14), np.array(waist_q3)))
        self.spans.append((t0, now()))
        return np.tile(np.arange(float(schema.POSE_DIM)), (schema.N_HANDS, 1))


def samples(arena) -> list[np.ndarray]:
    """Every written state slot, in write order.

    Valid only while the ring has not wrapped, which every test here keeps
    true: `index` runs 1..n against cursor 0..n-1.
    """
    rec = arena.view("state_rec")
    rows = [s for s in rec if int(s["index"]) > 0]
    assert [int(s["index"]) for s in rows] == list(range(1, len(rows) + 1))
    return rows


# --- The FK carries the joints it was computed from --------


def test_flange_is_the_fk_of_the_q_in_the_same_record(arena, cfg, hw):
    """The whole reason FK lives in P1: the flange pose and the joints it came
    from are one record, so P5's anchor and P6's solve cannot disagree.

    The waist assertion mirrors the IK's. P1 must place the **measured**
    waist where P6's IK pins the commanded one; solving with a zeroed waist
    instead is a silent bias in metres, so the test also proves it would notice.
    """
    p1_state.run(arena.shm.name, cfg, hw,
                 lowstate=FakeLowState(n=8), handstate=FakeHandState(n=0))

    fk = ForwardKinematics()
    arm = np.asarray(hw.joints_arm)
    waist = np.asarray(hw.joints_waist)
    rows = samples(arena)
    assert len(rows) == 8

    for s in rows:
        q = s["q"]
        assert np.allclose(s["flange"], fk.flange(q[arm], q[waist]), atol=1e-12)
        # and the measured waist is load-bearing: zeroing it moves the answer.
        assert not np.allclose(s["flange"], fk.flange(q[arm], np.zeros(3)),
                               atol=1e-6)


# --- The stamp comes first ---------------------------------


def test_t_recv_is_stamped_before_the_fk_not_after(arena, cfg, hw):
    """`t_recv` is every consumer's notion of when the measurement happened.
    Work done before the stamp is a bias no field records, so the stamp must
    sit between the sample arriving and the first thing done with it."""
    fk = RecordingFK(delay=5e-3)
    p1_state.run(arena.shm.name, cfg, hw, fk=fk,
                 lowstate=FakeLowState(n=4), handstate=FakeHandState(n=0))

    rows = samples(arena)
    assert len(rows) == len(fk.spans) == 4
    for s, (enter, leave) in zip(rows, fk.spans):
        t_recv = float(s["t_recv"])
        assert t_recv < enter                     # before the FK started
        assert enter - t_recv < 1e-3              # and immediately before it
        assert leave - t_recv > 4e-3              # a post-FK stamp would land here
        assert np.array_equal(s["flange"], np.tile(
            np.arange(float(schema.POSE_DIM)), (schema.N_HANDS, 1)))


# --- The two subscriptions are independent ------------------------


def test_a_dead_hand_topic_freezes_grip_but_not_the_arm_stream(arena, cfg, hw):
    """Losing the end-effector topic must not interrupt the arm stream, which
    gravity compensation depends on — and the staleness must be *checkable*,
    because a frozen `grip_meas` is otherwise indistinguishable from a gripper
    that is simply not moving. `t_grip_recv` is the only thing that tells them
    apart: there is no staleness flag and no timeout anywhere."""
    hand = FakeHandState(n=2)                     # two samples, then silence
    p1_state.run(arena.shm.name, cfg, hw, fk=RecordingFK(),
                 lowstate=FakeLowState(n=40, delay=1e-3), handstate=hand)

    rows = samples(arena)
    assert len(rows) == 40                        # the arm stream never stopped
    assert hand.i == 2

    last_grip = np.zeros(schema.GRIP_DIM)
    last_grip[0] = 1.2 + 0.01
    last_grip[schema.HAND_DIM] = 1.2 + 0.02

    tail = rows[10:]                              # past the latch's own startup
    frozen = {float(s["t_grip_recv"]) for s in tail}
    assert len(frozen) == 1                       # one arrival, held
    t_grip = frozen.pop()
    assert t_grip > 0.0
    for s in tail:
        assert np.allclose(s["grip_meas"], last_grip)
        assert t_grip < float(s["t_recv"])        # stale, and visibly so
    assert float(rows[-1]["t_recv"]) - t_grip > 0.02


# --- session gating -----------------------------------------


def test_the_loop_never_runs_once_the_session_has_stopped(arena, cfg, hw, rings):
    rings["control"].session = schema.STOPPED
    source = FakeLowState(n=5)
    p1_state.run(arena.shm.name, cfg, hw, fk=RecordingFK(),
                 lowstate=source, handstate=FakeHandState(n=0))

    assert source.i == 0
    assert samples(arena) == []
    assert source.closed                          # closed on the way out


def test_a_hand_topic_that_never_spoke_is_distinguishable_from_a_still_one(
        arena, cfg, hw):
    """`t_grip_recv == 0` is "never received". Without it, a gripper topic that
    never came up at all would publish a confident 12 zeros — a fully open hand
    — into every state sample from the first tick."""
    p1_state.run(arena.shm.name, cfg, hw, fk=RecordingFK(),
                 lowstate=FakeLowState(n=6), handstate=FakeHandState(n=0))

    for s in samples(arena):
        assert float(s["t_grip_recv"]) == 0.0
        assert not s["grip_meas"].any()
        assert float(s["t_recv"]) > 0.0           # the arm stream is fine


# --- The Dex-1 binding: one topic per hand, one motor per topic -------


class FakeSub:
    """A `ChannelSubscriber` that yields a fixed jaw angle, then EOF.

    `q=None` is the gripper that never speaks: `Read()` returning None is how
    the SDK reports a closed reader, and it is what a silent topic looks like
    from inside `read()`.
    """

    def __init__(self, topic, idl):
        self.topic, self.idl = topic, idl
        self.q, self.n = None, 0

    def Init(self, handler=None, queueLen=0):
        pass

    def Read(self, timeout=None):
        if self.q is None:
            return None
        self.n += 1
        motor = type("M", (), {"q": self.q})()
        return type("Msg", (), {"states": [motor]})()

    def Close(self):
        pass


def _dds_hand(monkeypatch, hw, jaws):
    """A connected `DdsHandState` whose two subscribers return `jaws`."""
    pytest.importorskip("cyclonedds")      # the message types are real
    from elrobot_deploy.runtime.drivers.dds import channel

    from elrobot_deploy.runtime.drivers import robot_state as wire

    monkeypatch.setattr(channel, "ChannelSubscriber", FakeSub)
    monkeypatch.setattr(channel, "ChannelFactoryInitialize",
                        lambda *a, **k: None)

    hand = wire.DdsHandState(hw)
    hand.connect()
    for sub, q in zip(hand._subs, jaws):
        sub.q = q
    return hand


def test_each_hand_reads_its_own_topic(monkeypatch, hw):
    """The Dex-1 gives every gripper its own topic carrying its own single
    motor, so hand `h`'s jaw is `topic_hand_state[h]`'s `states[0]` — NOT
    `states[h]` of one shared topic, which is what this used to read off the
    silent `rt/unitree_actuator/state`. Getting that wrong is survivable-looking:
    the acting gripper's aperture lands in the idle hand's slot and the policy's
    only proprioception reads a constant zero.
    """
    from elrobot_deploy.runtime.drivers import robot_state as wire

    hand = _dds_hand(monkeypatch, hw, [5.1316, 3.8305])   # measured, 2026-08-20
    rec = wire.record(hand.dtype)

    assert hand.read(rec)

    block = rec["grip"].reshape(schema.N_HANDS, schema.HAND_DIM)
    assert [s.topic for s in hand._subs] == list(hw.topic_hand_state)
    assert block[0, 0] == 5.1316 and block[1, 0] == 3.8305
    assert not block[:, 1:].any()          # the Brainco five stay zero


def test_one_silent_gripper_stops_the_whole_latch(monkeypatch, hw):
    """P1 pairs the topics deliberately. Returning the live hand alone would
    keep `t_grip_recv` moving while the other froze at its last value — a
    *plausible* reading that the session sequence's CHECK could no longer catch, reintroduced
    one hand at a time."""
    from elrobot_deploy.runtime.drivers import robot_state as wire

    hand = _dds_hand(monkeypatch, hw, [5.1316, None])      # right never speaks
    rec = wire.record(hand.dtype)

    assert not hand.read(rec)              # EOF, not a half-filled sample
    assert not rec["grip"].any()


def test_every_end_effector_has_one_topic_per_hand(hw):
    """The one-topic form (`rt/unitree_actuator/state`) ran a whole rollout on a
    constant; the built-in table carries `[left, right]` for every kind."""
    from elrobot_deploy.robot.end_effectors import END_EFFECTORS

    for ee in END_EFFECTORS.values():
        assert len(ee.topic_hand_cmd) == len(ee.topic_hand_state) == schema.N_HANDS
        assert "left" in ee.topic_hand_state[0] and "right" in ee.topic_hand_state[1]

    assert len(hw.topic_hand_state) == schema.N_HANDS
    assert len(hw.topic_hand_cmd) == schema.N_HANDS


# --- The robot's clock rides along (2026-09-16) ------------------------------


def test_the_robots_tick_is_recorded_beside_t_recv(arena, cfg, hw):
    """`LowState_.tick` is the robot's own millisecond clock at the sample.
    Beside `t_recv` it is the delivery delay: a stall in the stream is either
    the robot not producing samples (tick jumps) or samples delivered late
    (tick keeps its 2 ms cadence). The fake ticks at 2 ms per sample."""
    p1_state.run(arena.shm.name, cfg, hw,
                 lowstate=FakeLowState(n=6), handstate=FakeHandState(n=0))
    rows = samples(arena)
    assert [int(s["tick"]) for s in rows] == [0, 2, 4, 6, 8, 10]

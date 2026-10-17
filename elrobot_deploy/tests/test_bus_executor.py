"""The ElRobot executor: the 8-joint ring, `BusServo`, the drivers, the guard, the process.

No hardware: the real ST3215 driver runs against `ElRobotRepo/tests/fake_bus.py`.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import sys
from multiprocessing import resource_tracker
from pathlib import Path

import numpy as np
import pytest

from elrobot_deploy.cli import executor_test as et
from elrobot_deploy.robot.elrobot_joints import (
    GRIPPER,
    N_JOINTS,
    JointLimits,
    urdf_limits,
)
from elrobot_deploy.runtime import timing as clock
from elrobot_deploy.runtime.drivers.st3215_driver import (
    DriverError,
    SimDriver,
    St3215Driver,
)
from elrobot_deploy.runtime.execution.bus_servo import BusServo, ExecutorSettings
from elrobot_deploy.runtime.processes import p7_bus
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.shared_memory.bus_arena import bus_layout, open_bus_rings
from elrobot_deploy.runtime.shared_memory.layout import Arena

REPO = Path(__file__).resolve().parents[2]

WIDE = JointLimits(lower=np.full(N_JOINTS, -3.0), upper=np.full(N_JOINTS, 3.0),
                   velocity=np.full(N_JOINTS, 4.71))
LINEAR = ExecutorSettings(filter_hz=0.0)


@pytest.fixture
def rings():
    name = f"elxt{os.getpid()}{np.random.randint(1 << 30)}"
    arena = Arena.create(name, bus_layout())
    yield name, arena, open_bus_rings(arena)
    arena.close()
    arena.unlink()


def servo(r, *, settings=LINEAR, limits=WIDE, q0=None, t0=0.0):
    q0 = np.zeros(N_JOINTS) if q0 is None else q0
    return BusServo(ring=r["targets"], control=r["control"], limits=limits, settings=settings,
                    q0=q0, t0=t0)


def step(s, t, q_meas=None):
    """One published command: iterate through any ADVANCED branch, as the loop does."""
    for _ in range(4):
        q = s.step(t, q_meas)
        if q is not None:
            return q
    raise AssertionError("step never published")


def run_to(s, t0, t1, q_meas=None, hz=250.0):
    """Every tick from t0 to t1 inclusive, as the loop runs (the slew clamp is per tick)."""
    for i in range(round((t1 - t0) * hz) + 1):
        q = step(s, t0 + i / hz, q_meas)
    return q


# --- the ring ----------------------------------------------------------------------------


def test_ring_carries_8_joints_and_selects_by_deadline_and_world(rings):
    _, _, r = rings
    ring = r["targets"]
    ring.write(0, 1.0, np.arange(8.0), intervene=0)
    ring.write(1, 2.0, np.arange(8.0) + 1, intervene=0)
    ring.write(2, 1.5, np.full(8, 9.0), intervene=1)
    nxt = ring.next_after(0.5, 0)
    assert nxt.k == 0 and nxt.payload.shape == (8,)
    assert ring.next_after(1.0, 0).k == 1                  # strictly after
    assert ring.next_after(0.5, 1).k == 2                  # other world, other row
    assert [e.k for e in ring.line_after(0.5, 5.0, 0)] == [0, 1]


# --- the walk ----------------------------------------------------------------------------


def test_linear_walk_interpolates_between_deadlines(rings):
    _, _, r = rings
    a, b = np.full(8, 0.01), np.full(8, 0.03)
    r["targets"].write(0, 0.1, a, intervene=0)
    r["targets"].write(1, 0.2, b, intervene=0)
    s = servo(r)
    assert np.allclose(step(s, 0.0), 0.0)
    assert np.allclose(run_to(s, 0.004, 0.048), 0.0048)    # 48 % of the way to a
    assert np.allclose(run_to(s, 0.052, 0.152), 0.0204)    # 52 % of the way from a to b
    assert np.allclose(run_to(s, 0.156, 0.2), b)


def test_holds_the_last_command_when_the_ring_runs_dry(rings):
    _, _, r = rings
    r["targets"].write(0, 0.1, np.full(8, 0.02), intervene=0)
    s = servo(r)
    run_to(s, 0.0, 0.1)
    assert np.allclose(run_to(s, 0.104, 0.5), 0.02) and s.phase == "hold"


def test_intervene_change_preempts_and_ignores_the_old_world(rings):
    _, _, r = rings
    r["targets"].write(0, 1.0, np.full(8, 0.5), intervene=0)
    s = servo(r)
    q_mid = run_to(s, 0.0, 0.2).copy()
    r["control"]._c[0]["intervene"] = 1                   # a clutch press
    assert np.allclose(run_to(s, 0.204, 0.5), q_mid)       # world 0's target is gone: hold
    r["targets"].write(1, 0.6, q_mid + 0.01, intervene=1)
    assert np.allclose(run_to(s, 0.504, 0.6), q_mid + 0.01)


# --- the clamps --------------------------------------------------------------------------


def test_slew_clamp_bounds_every_tick(rings):
    _, _, r = rings
    r["targets"].write(0, 0.004, np.full(8, 1.0), intervene=0)   # 1 rad in one tick
    s = servo(r)
    step(s, 0.0)
    q = step(s, 0.004)
    assert np.allclose(q, 4.71 * 0.5 / 250.0)


def test_limits_clip_the_command(rings):
    _, _, r = rings
    lim = JointLimits(lower=np.full(8, -0.01), upper=np.full(8, 0.01), velocity=np.full(8, 4.71))
    r["targets"].write(0, 0.1, np.full(8, 0.5), intervene=0)
    s = servo(r, limits=lim)
    for t in np.arange(0.0, 0.2, 0.004):
        q = step(s, t)
    assert np.allclose(q, 0.01)


def test_squeeze_bound_caps_how_far_the_jaw_command_leads_the_measured_jaw(rings):
    _, _, r = rings
    goal = np.zeros(8)
    goal[GRIPPER] = 2.0                                    # "close fully", at a feasible rate
    r["targets"].write(0, 1.0, goal, intervene=0)
    st = ExecutorSettings(filter_hz=0.0, squeeze_margin=0.1)
    s = servo(r, settings=st)
    meas = np.zeros(8)
    meas[GRIPPER] = 0.5                                    # an object stops the jaw here
    q = run_to(s, 0.0, 1.2, meas)
    assert q[GRIPPER] == pytest.approx(0.6)                # measured + margin, not 2.0
    meas[GRIPPER] = 0.55                                   # it compresses: follow, same force
    assert run_to(s, 1.204, 1.3, meas)[GRIPPER] == pytest.approx(0.65)


def test_friction_compensation_leads_the_line_while_moving_and_not_while_holding(rings):
    _, _, r = rings
    goal = np.full(8, 0.2)
    r["targets"].write(0, 1.0, goal, intervene=0)          # 0.2 rad/s on every joint
    st = ExecutorSettings(filter_hz=0.0, friction=math.radians(0.5))
    s = servo(r, settings=st)
    q = run_to(s, 0.0, 0.5)
    lead = q - s.q_line_last
    want = math.radians(0.5) * math.tanh(0.2 / st.friction_v)
    assert np.allclose(lead[:7], want) and lead[GRIPPER] == 0.0   # the jaw is the squeeze's
    q = run_to(s, 0.504, 1.5)                              # ring dry: holding, no lead
    assert np.allclose(q, goal)


# --- shaping, with rows arriving like the phone's ----------------------------------------


def _ramp_error(r, *, filter_lead: bool, v=0.3, lead=0.04, hz=250.0, row_hz=60.0):
    st = ExecutorSettings(hz=hz, filter_hz=15.0, filter_lead=filter_lead)
    s = servo(r, settings=st)
    ring, k, t_row, errs = r["targets"], 0, 0.0, []
    for i in range(int(2.0 * hz)):
        t = i / hz
        while t_row <= t:                                  # rows captured at 60 Hz, due +lead
            ring.write(k, t_row + lead, np.full(8, v * t_row), intervene=0)
            k, t_row = k + 1, t_row + 1 / row_hz
        q = step(s, t)
        if t > 1.0:
            errs.append(abs(q[0] - v * (t - lead)))        # the line: capture time + lead
    return max(errs)


def test_filter_with_look_ahead_adds_no_lag_on_a_ramp(rings):
    _, _, r = rings
    # What remains is the filter holding its reference over each 4 ms step: v * h / 2
    # = 0.6 mrad at 0.3 rad/s (0.03 deg), against the 6.4 mrad of the filter's own lag below.
    assert _ramp_error(r, filter_lead=True) < 1.2 * 0.3 * (1 / 250) / 2


def test_filter_without_look_ahead_lags_2_over_w(rings):
    _, _, r = rings
    lag = 2.0 / (2 * math.pi * 15.0)
    assert _ramp_error(r, filter_lead=False) == pytest.approx(0.3 * lag, rel=0.15)


def test_sine_plan_through_ring_servo_and_sim_driver_with_a_fake_clock(rings):
    """The executor_test pipeline in one thread: 60 Hz rows -> 250 Hz goals -> lagging servos."""
    _, _, r = rings
    now = [0.0]
    limits = urdf_limits()
    plan = et.make_plan(limits.clip(np.radians(et.REST_DEG)), limits, [2, 3], 5.0, 0.5, 2, 15.0)
    drv = SimDriver(limits.clip(np.radians(et.REST_DEG)), clock=lambda: now[0], sleep_io=False)
    q0 = drv.connect()
    s = servo(r, settings=ExecutorSettings(), limits=limits, q0=q0)
    ring, k, t_row, cmds, errs = r["targets"], 0, 0.0, [], []
    s0, s1 = plan.sine_window
    for i in range(int(plan.duration * 250)):
        t = now[0] = i / 250.0
        while t_row <= t:
            ring.write(k, t_row + 0.04, plan.at(t_row), intervene=0)
            k, t_row = k + 1, t_row + 1 / 60.0
        q = step(s, t, drv.read())
        drv.write(q)
        cmds.append(q.copy())
        if s0 + 0.3 < t - 0.04 < s1:
            errs.append(np.abs(drv.read() - plan.at(t - 0.04))[[2, 3]].max())
    cmds = np.array(cmds)
    peak = math.radians(5.0) * 2 * math.pi * 0.5
    assert np.abs(np.diff(cmds, axis=0)).max() <= 1.2 * peak / 250.0   # no 60 Hz staircase
    assert max(errs) < math.radians(1.0)                  # a 30 ms servo lag on a 16 deg/s sine
    assert np.allclose(cmds[-1], plan.end, atol=1e-3)


# --- the guard ---------------------------------------------------------------------------


def test_guard_lag_runaway_and_bus_failures():
    g = p7_bus.Guard()
    cmd = np.zeros(8)
    late = np.zeros(8)
    late[2] = np.radians(10)
    assert g.check(0.0, cmd, late) is None                 # behind, but not for long yet
    assert "joints [3]" in g.check(0.6, cmd, late)
    g = p7_bus.Guard()
    wild = np.zeros(8)
    wild[4] = np.radians(30)
    assert "not obeying" in g.check(0.0, cmd, wild) and g.runaway == [4]
    g = p7_bus.Guard()
    msgs = [g.check(0.0, cmd, None) for _ in range(p7_bus.BUS_FAILS)]
    assert msgs[-1] is not None and all(m is None for m in msgs[:-1])


# --- the real driver, on the fake serial bus ---------------------------------------------


@pytest.fixture
def fake_bus_mod():
    sys.path.insert(0, str(REPO / "tests"))
    try:
        import fake_bus
        yield fake_bus
    finally:
        sys.path.remove(str(REPO / "tests"))


def _fake_arm(fake_bus, cals, readings, **servo_kw):
    from elrobot.st3215 import registers as R
    from elrobot.st3215.bus import Bus

    servos = []
    for j in range(1, 9):
        # FakeServo reports `u - offset`, like the real one after the centring offset.
        s = fake_bus.FakeServo(j, position=readings[j - 1] + cals[j].offset, **servo_kw)
        s.set(R.OFFSET, cals[j].offset)
        s.set(R.PRESENT_POSITION, readings[j - 1])
        servos.append(s)
    return Bus(serial_port=fake_bus.FakeSerial(servos)), servos


def test_st3215_driver_converts_and_enables_like_move(fake_bus_mod):
    from elrobot.calibration import load
    from elrobot.st3215 import registers as R

    cals = load()
    readings = [cals[j].zero_reading for j in range(1, 9)]
    bus, servos = _fake_arm(fake_bus_mod, cals, readings)
    sim = fake_bus_mod.SimClock()
    for s in servos:
        s.set(R.P_COEF, 32)
    drv = St3215Driver(bus=bus, cals=cals, clock=sim.now, sleep=sim.sleep, p_coef={6: 48},
                       goal_speed={6: 120})
    q0 = drv.connect()
    assert np.allclose(q0, 0.0)                            # every joint at its zero reading
    assert [s.get(R.P_COEF) for s in servos] == [32] * 5 + [48] + [32] * 2
    speeds = [s.get(R.GOAL_SPEED) for s in servos]
    assert speeds[5] == 120 and all(v > 1000 for i, v in enumerate(speeds) if i != 5)
    assert all(s.get(R.TORQUE_ENABLE) == 1 and s.get(R.ACCELERATION) == 0 for s in servos)
    assert all(s.get(R.GOAL_SPEED) > 0 for s in servos)   # 0 would mean "full speed"
    q = np.radians([10, -10, 20, 5, 0, 0, 0, 30])
    drv.write(q)
    goals = [s.get(R.GOAL_POSITION) for s in servos]
    assert np.allclose(drv.to_rad(np.array(goals)), q, atol=drv.rad_per_step)
    assert goals[7] < cals[8].zero_reading                 # gripper: + closes, reading falls
    drv.write(np.full(8, 10.0))                            # far past every limit
    assert [s.get(R.GOAL_POSITION) for s in servos] == [int(drv.hi[i]) if drv.sign[i] > 0
                                                        else int(drv.lo[i]) for i in range(8)]
    pos, _speed, _load, temp = drv.read_full()
    assert np.allclose(pos, 0.0) and temp.shape == (8,)
    drv.release()
    assert all(s.get(R.TORQUE_ENABLE) == 0 for s in servos)
    assert all(s.get(R.P_COEF) == 32 for s in servos)      # the session's P is undone


def test_st3215_driver_rewrite_every_sends_only_changed_goals(fake_bus_mod):
    from elrobot.calibration import load
    from elrobot.st3215 import protocol

    cals = load()
    bus, _ = _fake_arm(fake_bus_mod, cals, [cals[j].zero_reading for j in range(1, 9)])
    ser = bus.ser
    drv = St3215Driver(bus=bus, cals=cals, rewrite_every=4)

    def servos_in_last_write():
        p = ser.sent[-1]
        assert p[4] == protocol.SYNC_WRITE
        params = p[5:-1]
        return [params[2 + k * (params[1] + 1)] for k in range((len(params) - 2) // (params[1] + 1))]

    q = np.zeros(8)
    drv.write(q)                                           # first: everyone
    assert servos_in_last_write() == list(range(1, 9))
    n = len(ser.sent)
    drv.write(q)                                           # unchanged: nothing on the wire
    assert len(ser.sent) == n
    q2 = q.copy()
    q2[5] = math.radians(1)
    drv.write(q2)                                          # one changed: only servo 6
    assert servos_in_last_write() == [6]
    drv.write(q2)                                          # 4th write: the refresh, everyone
    assert servos_in_last_write() == list(range(1, 9))


def test_st3215_driver_release_confirms_torque_off_and_resends(fake_bus_mod):
    """2026-10-17: a torque-off sync_write sent just before the port closed was lost."""
    from elrobot.calibration import load
    from elrobot.st3215 import protocol
    from elrobot.st3215 import registers as R

    cals = load()
    bus, servos = _fake_arm(fake_bus_mod, cals, [cals[j].zero_reading for j in range(1, 9)])
    sim = fake_bus_mod.SimClock()
    drv = St3215Driver(bus=bus, cals=cals, clock=sim.now, sleep=sim.sleep)
    drv.connect()
    ser, write = bus.ser, bus.ser.write
    dropped = []

    def lossy(packet, always=False):
        if (packet[4] == protocol.SYNC_WRITE and packet[5] == R.TORQUE_ENABLE.addr
                and (always or not dropped)):
            dropped.append(packet)
            return
        write(packet)

    ser.write = lossy                                      # the first torque-off is lost
    drv.release()
    assert len(dropped) == 1 and all(s.get(R.TORQUE_ENABLE) == 0 for s in servos)

    for s in servos:
        s.set(R.TORQUE_ENABLE, 1)
    drv.torque_on = True
    ser.write = lambda p: lossy(p, always=True)            # every torque-off is lost
    with pytest.raises(DriverError, match="still on"):
        drv.release()


def test_bus_process_never_releases_an_arm_it_did_not_enable(rings, monkeypatch):
    """A refused connect may mean another program holds the arm: turning it off would drop it."""
    import signal

    name, _, _ = rings
    calls = []

    class Refuses:
        def connect(self):
            raise DriverError("torque is already on for [1, 2]")

        def release(self, joints=None):
            calls.append("release")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(p7_bus, "make_driver", lambda kind, **kw: Refuses())
    status, sigint = queue.Queue(), signal.getsignal(signal.SIGINT)
    try:
        p7_bus.run(name, driver="x", driver_kw={}, settings=ExecutorSettings(), limits=WIDE,
                   status=status)
    finally:
        signal.signal(signal.SIGINT, sigint)
    msgs = [status.get_nowait() for _ in range(status.qsize())]
    assert calls == ["close"]
    assert msgs[0][0] == "fault" and "already on" in msgs[0][1] and msgs[-1][0] == "done"


def test_st3215_driver_refuses_a_held_arm_and_aborts_a_runaway(fake_bus_mod):
    from elrobot.calibration import load
    from elrobot.st3215 import registers as R

    cals = load()
    readings = [cals[j].zero_reading for j in range(1, 9)]
    bus, servos = _fake_arm(fake_bus_mod, cals, readings)
    servos[2].set(R.TORQUE_ENABLE, 1)
    with pytest.raises(DriverError, match="already on"):
        St3215Driver(bus=bus, cals=cals).connect()

    sim = fake_bus_mod.SimClock()
    bus, servos = _fake_arm(fake_bus_mod, cals, readings, stops=(-10_000, 10_000), clock=sim)
    servos[4].runaway = +1                                 # the multi-turn offset trap
    drv = St3215Driver(bus=bus, cals=cals, clock=sim.now, sleep=sim.sleep)
    with pytest.raises(DriverError, match="on their own"):
        drv.connect()
    assert all(s.get(R.TORQUE_ENABLE) == 0 for s in servos)


# --- the process -------------------------------------------------------------------------


def test_bus_process_follows_the_ring_and_logs(tmp_path):
    name = f"elxp{os.getpid()}"
    arena = Arena.create(name, bus_layout())
    r = open_bus_rings(arena)
    r["control"].session = schema.ACTIVE
    ctx = mp.get_context("spawn")
    status = ctx.Queue()
    limits = urdf_limits()
    q_rest = limits.clip(np.radians(et.REST_DEG))
    log = tmp_path / "exec.csv"
    proc = ctx.Process(target=p7_bus.run, args=(name,), kwargs={
        "driver": "sim", "driver_kw": {"q0": q_rest.tolist()}, "settings": ExecutorSettings(),
        "limits": limits, "status": status, "log_path": str(log)})
    proc.start()
    try:
        kind, q0 = status.get(timeout=20)
        resource_tracker.register(arena.shm._name, "shared_memory")
        assert kind == "ready" and np.allclose(q0, q_rest, atol=1e-3)
        target = q_rest.copy()
        target[5] += math.radians(5)
        t_end = clock.now() + 0.6
        k = 0
        while clock.now() < t_end:                         # 60 Hz rows, all the same target
            r["targets"].write(k, clock.now() + 0.04, target, intervene=0)
            k += 1
            clock.sleep_until(clock.now() + 1 / 60)
        clock.sleep_until(clock.now() + 0.3)
        assert np.allclose(r["state"].latest()["q"], target, atol=math.radians(0.3))
    finally:
        r["control"].session = schema.STOPPED
        done = None
        while done is None:
            try:
                msg = status.get(timeout=10)
            except queue.Empty:
                break
            if msg[0] == "done":
                done = msg[1]
        proc.join(timeout=5)
        arena.close()
        arena.unlink()
    a = p7_bus.load_log(Path(done))
    assert len(a) > 100 and np.median(a["dt_ms"][5:]) == pytest.approx(4.0, abs=0.2)

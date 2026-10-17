"""P0 (`runtime/launcher/`) end to end on simulated servos: real processes, a real arena.

Each test is a whole session (spawn the bus process, home, READY/ACTIVE, exit), a few seconds.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from elrobot_deploy.config_loading.config import load
from elrobot_deploy.runtime import timing as clock
from elrobot_deploy.runtime.launcher import Launcher
from elrobot_deploy.runtime.launcher.ramps import cosine_ramp, ramp_duration
from elrobot_deploy.runtime.shared_memory import schema
from tests.conftest import arena_name


@pytest.fixture
def fast():
    """The shipped profile on simulated servos, with short ramps."""
    return load(sets=["bus.driver=sim", "poses.min_s=0.3", "poses.velocity_margin=0.6"])


class Recorder(Launcher):
    """A launcher that remembers what it saw and can be told to request things."""

    def __init__(self, cfg, **kw):
        self.lines = []
        super().__init__(cfg, phone=False, arena_name=arena_name(),
                         log=self.lines.append, **kw)
        self.seen = {}
        self.final = None

    def _set(self, state):
        super()._set(state)
        snap = self.rings["state"].latest()
        if snap is not None:
            self.seen[schema.SESSION_NAMES[state]] = np.asarray(snap["q"]).copy()
        if state == schema.STOPPED:
            self.final = (self.rings["control"].release_torque, self.rings["state"].latest())


def request_when(launcher, state, request, delay=0.2):
    def go():
        while launcher.rings is None or launcher.rings["control"].session != state:
            time.sleep(0.01)
        time.sleep(delay)
        launcher.rings["control"].request = request
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def test_a_session_homes_hands_over_and_exits_to_the_exit_pose(fast, tmp_path):
    p0 = Recorder(fast, log_dir=tmp_path)
    request_when(p0, schema.ACTIVE, schema.REQ_EXIT)
    assert p0.run() == 0
    assert np.allclose(p0.seen["READY"], fast.poses.home, atol=math.radians(1.0))
    release, last = p0.final
    assert release, "a completed exit ramp lets the bus release torque"
    assert np.allclose(last["q"], fast.poses.exit, atol=math.radians(1.0))
    assert list(tmp_path.glob("*_bus.csv")), "the bus process saved its tick log"


def test_reset_ramps_home_and_ticks_keep_counting(fast):
    p0 = Recorder(fast)
    k_seen = []

    def drive():
        while p0.rings is None or p0.rings["control"].session != schema.ACTIVE:
            time.sleep(0.01)
        # Move the arm off home by hand (as P6 would), then RESET.
        c, t = p0.rings["control"], p0.rings["targets"]
        q = fast.poses.home.copy()
        q[0] += 0.3
        k0 = c.k_exec
        k_seen.append(k0)
        now = clock.now()
        for i in range(30):
            t.write(k0 + 2 + i, now + 0.05 + i * fast.dt, q, intervene=c.intervene)
        time.sleep(0.05 + 30 * fast.dt + 0.4)
        c.request = schema.REQ_RESET
        while c.session == schema.ACTIVE:       # HOME (the reset ramp), READY, ACTIVE again
            time.sleep(0.001)
        while c.session != schema.ACTIVE:
            time.sleep(0.001)
        k_seen.append(c.k_exec)
        c.request = schema.REQ_EXIT

    threading.Thread(target=drive, daemon=True).start()
    assert p0.run() == 0
    assert k_seen[1] > k_seen[0] + 30, "k_exec never rewinds"
    assert np.allclose(p0.seen["READY"], fast.poses.home, atol=math.radians(1.0))


def test_a_bus_fault_ends_the_session_and_leaves_the_arm_held(fast):
    """The bus process reporting a fault mid-homing ends the session: no exit ramp, torque
    left on. (The guards themselves are `test_bus_executor.py`'s.)"""
    p0 = Recorder(fast)

    def fault_after_ready():
        while p0.rings is None or p0.rings["control"].session != schema.HOME:
            time.sleep(0.01)
        # The bus process is the only writer of these words; standing in for its guard here.
        p0.rings["control"].set_fault("joints [3] are 25 deg off their command (test)")
        p0.rings["control"].bus = schema.BUS_HOLDING

    threading.Thread(target=fault_after_ready, daemon=True).start()
    assert p0.run() == 1
    release, _ = p0.final
    assert not release, "after a fault the arm is not released"
    assert any("25 deg off" in line for line in p0.lines)


def test_an_unexpected_failure_skips_the_ramp_and_holds(fast):
    p0 = Recorder(fast)

    def boom():
        raise RuntimeError("something nobody understands")
    p0.ready = boom
    with pytest.raises(RuntimeError, match="nobody understands"):
        p0.run()
    release, last = p0.final
    assert not release
    assert np.allclose(last["q"], fast.poses.home, atol=math.radians(1.0)), "held where it was"


def test_ramps_respect_the_velocity_limit():
    q0, q1 = np.zeros(8), np.r_[1.0, np.zeros(7)]
    qdot = np.full(8, 2.0)
    assert ramp_duration(q0, q1, qdot_max=qdot, min_s=0.1, margin=0.5) == pytest.approx(1.0)
    qs = cosine_ramp(q0, q1, dt=0.01, qdot_max=qdot, min_s=0.1, margin=0.5)
    v = np.abs(np.diff(np.vstack([q0, qs]), axis=0)).max() / 0.01
    assert v <= 2.0 * 0.5 * math.pi / 2 * 1.01
    assert np.array_equal(qs[-1], q1)

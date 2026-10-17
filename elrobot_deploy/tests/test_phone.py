"""The phone source (`runtime/processes/p11_phone.py`) and the packet format
(`runtime/teleop/phone.py`), driven packet by packet without a socket."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from elrobot_deploy.runtime.processes.p11_phone import PhoneLoop
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.teleop import phone
from elrobot_deploy.runtime.teleop.phone import GRIP_IN_WORLD

FWD, LEFT, UP = np.array([0, 0, -1.0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0])


def pose(p=(0, 0, 0), R=GRIP_IN_WORLD):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class Phone:
    """Packets as the app sends them, 60 Hz on its own clock."""

    def __init__(self, loop):
        self.loop = loop
        self.seq = 0
        self.t = 100.0
        self.recv = 500.0
        self.epoch = 0

    def send(self, p=(0, 0, 0), *, engage=False, grip=False, track="normal", R=GRIP_IN_WORLD,
             dt=1 / 60, seq=None):
        self.seq = self.seq + 1 if seq is None else seq
        self.t += dt
        self.recv += dt
        pkt = phone.parse(phone.packet(self.seq, self.t, pose(p, R), engage=engage, grip=grip,
                                       track=track, world_epoch=self.epoch))
        return self.loop.on_packet(pkt, self.recv)


@pytest.fixture
def active(rings, cfg):
    rings["control"].session = schema.ACTIVE
    q = cfg.poses.home
    rings["state"].write(t=0.0, q=q, q_cmd=q)
    lines = []
    loop = PhoneLoop(rings, cfg, log=lines.append)
    loop.lines = lines
    return loop, Phone(loop), rings


def engage(ph):
    ph.send(engage=False)
    return ph.send(engage=True)


# --- packets --------------------------------------------------------------------------------


def test_a_packet_round_trips_and_v1_has_no_epoch():
    T = pose((0.1, 0.2, 0.3))
    p = phone.parse(phone.packet(7, 12.5, T, engage=True, grip=False, world_epoch=3))
    assert p.seq == 7 and p.t == 12.5 and np.allclose(p.T, T) and p.engage and not p.grip
    assert p.world_epoch == 3 and p.track == phone.TRACK_NORMAL
    v1 = json.loads(phone.packet(1, 1.0, T))
    del v1["world_epoch"]
    v1["v"] = 1
    assert phone.parse(json.dumps(v1).encode()).world_epoch == 0


@pytest.mark.parametrize("bad", [b"not json", b"{}", json.dumps({"seq": 1, "t": 1.0,
                                 "pose": [0] * 16}).encode()])
def test_bad_packets_are_refused_whole(bad):
    with pytest.raises(phone.BadPacket):
        phone.parse(bad)


# --- the clutch and the rows ----------------------------------------------------------------


def test_nothing_happens_until_a_press_during_active(active):
    loop, ph, rings = active
    rings["control"].session = schema.READY
    assert ph.send(engage=True) == "inactive"
    rings["control"].session = schema.ACTIVE
    assert ph.send(engage=True) == "idle", "a press held from before ACTIVE does not count"
    assert rings["control"].intervene == 0
    assert ph.send(engage=False) == "idle"
    assert ph.send(engage=True) == "engaged"
    assert rings["control"].intervene == 1


def test_rows_carry_the_motion_since_the_press_in_robot_axes(active):
    loop, ph, rings = active
    engage(ph)
    ph.send(0.02 * FWD + 0.01 * UP, engage=True)
    row = rings["intervene"].newest(1)
    assert np.allclose(row.delta[:3], [0.0, 0.02, 0.01], atol=1e-9)   # base: +y fwd, +z up
    # Pinned at the press (one frame earlier) + the lead, then the phone's own clock.
    assert row.deadline == pytest.approx(ph.recv + loop.cfg.teleop.lead_s, abs=1e-6)


def test_a_release_hands_back_on_an_even_counter_and_stops_rows(active):
    loop, ph, rings = active
    engage(ph)
    n = rings["intervene"].meta()["newest"]
    assert ph.send(engage=False) == "idle"
    assert rings["control"].intervene == 2
    ph.send(engage=False)
    assert rings["intervene"].meta()["newest"] == n


def follow_jaw(rings, cfg, jaw):
    q = cfg.poses.home.copy()
    q[schema.GRIPPER] = jaw
    rings["state"].write(t=0.0, q=q, q_cmd=q)


def test_the_gripper_closes_while_held_and_holds_when_released(active):
    loop, ph, rings = active
    engage(ph)
    for _ in range(30):                                  # half a second squeezing, free air
        ph.send(engage=True, grip=True)
        follow_jaw(rings, loop.cfg, rings["intervene"].newest(1).grip)
    closed = rings["intervene"].newest(1).grip
    lo = loop.cfg.limits.lower[schema.GRIPPER]
    assert closed == pytest.approx(lo + 30 / 60 * loop.cfg.teleop.gripper_rate, rel=1e-3)
    ph.send(engage=False)                                # clutch released: no rows, jaw held
    assert loop.gripper.cmd == pytest.approx(closed)


def test_on_an_object_the_jaw_command_leads_the_jaw_by_the_squeeze_margin(active):
    loop, ph, rings = active
    engage(ph)
    follow_jaw(rings, loop.cfg, 0.5)                     # an object stops the jaw at 0.5 rad
    for _ in range(60):
        ph.send(engage=True, grip=True)
    assert rings["intervene"].newest(1).grip == pytest.approx(
        0.5 + loop.cfg.executor.squeeze_margin)


# --- the guards -----------------------------------------------------------------------------


@pytest.mark.parametrize("glitch, why", [
    (dict(track="limited:excessiveMotion"), "tracking"),
    (dict(p=0.2 * FWD), "jumped"),
    (dict(R=GRIP_IN_WORLD @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])), "jumped"),
])
def test_a_guard_releases_and_needs_a_fresh_press(active, glitch, why):
    loop, ph, rings = active
    engage(ph)
    assert ph.send(engage=True, **glitch) == "released"
    assert why in loop.why and rings["control"].intervene == 2
    assert ph.send(engage=True) == "released", "still held: stays released"
    ph.send(engage=False)
    assert ph.send(engage=True) == "engaged"


def test_a_world_reset_releases(active):
    loop, ph, rings = active
    engage(ph)
    ph.epoch += 1
    assert ph.send(engage=True) == "released" and "world origin" in loop.why


def test_a_stale_stream_releases(active):
    loop, ph, rings = active
    engage(ph)
    loop.on_idle(ph.recv + loop.cfg.teleop.stale_s / 2)
    assert loop.engaged
    loop.on_idle(ph.recv + loop.cfg.teleop.stale_s + 0.01)
    assert not loop.engaged and "no packet" in loop.why
    assert rings["control"].intervene % 2 == 0


def test_the_end_of_the_rollout_releases(active):
    loop, ph, rings = active
    engage(ph)
    rings["control"].session = schema.HOME
    loop.on_idle(ph.recv)
    assert not loop.engaged and rings["control"].intervene % 2 == 0


def test_a_phone_pointing_up_at_the_press_is_not_engaged(active):
    loop, ph, rings = active
    up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]) @ GRIP_IN_WORLD     # top points up
    assert np.allclose(up @ [-1, 0, 0], UP)
    ph.send(engage=False, R=up)
    assert ph.send(engage=True, R=up) == "released"
    assert rings["control"].intervene == 0


def test_reordered_packets_are_dropped_and_a_restart_releases(active):
    loop, ph, rings = active
    engage(ph)
    n = rings["intervene"].meta()["newest"]
    ph.send(engage=True, seq=ph.seq - 1)                 # late duplicate: ignored
    assert rings["intervene"].meta()["newest"] == n and loop.engaged
    ph.seq = 50_000
    ph.send(engage=True)
    ph.seq = 0
    assert ph.send(engage=True) == "released" and "restarted" in loop.why


def test_a_counter_left_odd_by_a_dead_source_is_handed_back(rings, cfg):
    rings["control"].intervene = 5
    PhoneLoop(rings, cfg, log=lambda *a: None)
    assert rings["control"].intervene == 6


def test_closing_hands_back(active):
    loop, ph, rings = active
    engage(ph)
    loop.close()
    assert rings["control"].intervene % 2 == 0


def test_latency_is_reported_per_engagement(active):
    loop, ph, rings = active
    engage(ph)
    for i in range(10):
        ph.send(engage=True, dt=1 / 60)
    ph.recv += 0.03                                      # one late packet
    ph.send(engage=True)
    ph.send(engage=False)
    report = [x for x in loop.lines if "engagement" in x]
    assert report and "max +30.0 ms" in report[0]


def test_lead_covers_a_late_packet_without_sliding(active):
    loop, ph, rings = active
    engage(ph)
    ph.recv += 0.03                                      # 30 ms late, inside the 40 ms lead
    ph.send(engage=True)
    assert loop.pace.slid == 0.0
    ph.recv += 0.05                                      # 80 ms late in total: slides
    ph.send(engage=True)
    assert loop.pace.slid > 0.0
    assert rings["intervene"].newest(1).deadline > ph.recv
    assert math.isfinite(rings["intervene"].meta()["latency_s"])

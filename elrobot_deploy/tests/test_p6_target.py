"""P6 (`runtime/processes/p6_target.py`): operator rows in, joint targets out.

Most tests use a fake solver whose "TCP position" is the first three joints, so every number a
test checks is visible by eye; one runs the real `tcp_qp` to check the anchor round trip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from elrobot_deploy.runtime.processes.p6_target import Target
from elrobot_deploy.runtime.shared_memory import schema

T0 = 1000.0


class FakeSolver:
    """fk: position = q[:3], identity rotation. solve: the seed with q[:3] = the pose's position."""

    def __init__(self, ok=True):
        self.ok = ok
        self.nominal = None
        self.calls = 0

    def fk(self, q7):
        return np.r_[np.asarray(q7, float)[:3], 1.0, 0.0, 0.0, 0.0]

    def solve(self, pose, seed, budget=None):
        self.calls += 1
        q = np.array(seed, dtype=float)
        q[:3] = pose[:3]
        return q, 0.001, self.ok

    def set_nominal(self, q7):
        self.nominal = np.array(q7, dtype=float)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def wide(cfg):
    """The shipped config with a huge workspace and fast joints, so only the test's own bound
    acts."""
    from dataclasses import replace
    teleop = replace(cfg.teleop, workspace_min=(-10.0, -10.0, -10.0),
                     workspace_max=(10.0, 10.0, 10.0))
    limits = replace(cfg.limits, lower=np.full(8, -3.0), upper=np.full(8, 3.0),
                     velocity=np.full(8, 100.0))
    return replace(cfg, teleop=teleop, limits=limits)


def held(rings, q, t=T0):
    rings["state"].write(t=t, q=q, q_cmd=q)


def press(rings):
    rings["control"].intervene = rings["control"].intervene + 1


def row(rings, deadline, delta_xyz=(0, 0, 0), grip=0.1, quat=(1, 0, 0, 0)):
    return rings["intervene"].write(intervene=rings["control"].intervene, deadline=deadline,
                                    delta=np.r_[delta_xyz, quat], grip=grip)


def make(rings, cfg, solver=None, clock=None):
    return Target(rings, cfg, solver or FakeSolver(), now=clock or Clock())


Q0 = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.05])


def test_nothing_is_written_while_nobody_holds_the_clutch(rings, wide):
    held(rings, Q0)
    p6 = make(rings, wide)
    assert p6.step().reason == "released"
    row(rings, T0 + 0.04)                  # a row of an even world is ignored
    assert p6.step().reason == "released"
    assert rings["targets"].newest_full() is None


def test_a_press_anchors_on_the_held_command_and_sets_the_nominal_pose(rings, wide):
    held(rings, Q0)
    solver = FakeSolver()
    p6 = make(rings, wide, solver)
    press(rings)
    assert p6.step().reason == "idle"
    assert np.allclose(p6.anchor_pose[:3], Q0[:3])
    assert np.allclose(solver.nominal, Q0[:7])


def test_a_row_becomes_the_anchor_moved_by_its_delta(rings, wide):
    held(rings, Q0)
    rings["control"].k_exec = 40
    p6 = make(rings, wide)
    press(rings)
    row(rings, T0 + 0.04, delta_xyz=(0.01, -0.02, 0.0), grip=0.3)
    s = p6.step()
    assert s.written and s.k == 42 and s.deadline == T0 + 0.04
    t = rings["targets"].full(42)
    assert np.allclose(t["q"][:3], Q0[:3] + [0.01, -0.02, 0.0])
    assert t["q"][schema.GRIPPER] == pytest.approx(0.3)
    assert int(t["intervene"]) == rings["control"].intervene
    assert np.allclose(t["pose"][:3], Q0[:3] + [0.01, -0.02, 0.0])
    assert t["err"] == pytest.approx(0.001)


def test_rows_are_decimated_onto_the_grid(rings, wide):
    """120 Hz rows on a 60 Hz grid: every other row becomes a target, ticks consecutive."""
    held(rings, Q0)
    p6 = make(rings, wide)
    press(rings)
    dt = wide.dt
    written = []
    for i in range(12):
        row(rings, T0 + 0.04 + i * dt / 2, delta_xyz=(0.001 * i, 0, 0))
        s = p6.step()
        if s.written:
            written.append((s.k, i))
    assert [i for _, i in written] == [0, 2, 4, 6, 8, 10]
    ks = [k for k, _ in written]
    assert ks == list(range(ks[0], ks[0] + 6))


def test_a_late_row_refounds_the_grid_instead_of_firing_a_backlog(rings, wide):
    held(rings, Q0)
    p6 = make(rings, wide)
    press(rings)
    row(rings, T0 + 0.04)
    assert p6.step().written
    row(rings, T0 + 0.04 + 5 * wide.dt)     # the stream paused for four frames
    assert p6.step().written
    row(rings, T0 + 0.04 + 5.5 * wide.dt)   # half a frame later: decimated, grid is the new row
    assert p6.step().reason == "decimated"


def test_the_velocity_clamp_spans_the_deadline_difference(rings, wide):
    from dataclasses import replace
    slow = replace(wide, limits=replace(wide.limits, velocity=np.full(8, 1.0)),
                   velocity_scale=1.0)                          # 1 rad/s
    held(rings, Q0)
    p6 = make(rings, slow)
    press(rings)
    row(rings, T0 + 0.05, delta_xyz=(1.0, 0, 0))                # asks for 1 rad at once
    assert p6.step().written
    first = rings["targets"].newest_full()
    assert first["q"][0] == pytest.approx(Q0[0] + 0.05)         # 50 ms since the anchor
    row(rings, T0 + 0.05 + 0.1, delta_xyz=(1.0, 0, 0))          # the next frame 100 ms later
    assert p6.step().written
    second = rings["targets"].newest_full()
    assert second["q"][0] == pytest.approx(first["q"][0] + 0.1)


def test_the_workspace_box_clamps_but_widens_to_contain_the_anchor(rings, wide):
    from dataclasses import replace
    boxed = replace(wide, teleop=replace(wide.teleop, workspace_min=(0.0, 0.0, 0.0),
                                         workspace_max=(0.2, 0.25, 0.25)))
    held(rings, Q0)                         # anchor x 0.1 (inside), y 0.2, z 0.3 (above the box)
    p6 = make(rings, boxed)
    press(rings)
    row(rings, T0 + 0.04, delta_xyz=(0.5, 0.0, 0.0))
    assert p6.step().written
    q = rings["targets"].newest_full()["q"]
    assert q[0] == pytest.approx(0.2)       # clamped to the box
    assert q[2] == pytest.approx(0.3)       # the anchor's own height is allowed: no jump
    row(rings, T0 + 0.04 + wide.dt, delta_xyz=(0.0, 0.0, 0.1))
    assert p6.step().written
    assert rings["targets"].newest_full()["q"][2] == pytest.approx(0.3)


def test_an_abandoned_solve_writes_nothing(rings, wide):
    held(rings, Q0)
    p6 = make(rings, wide, FakeSolver(ok=False))
    press(rings)
    row(rings, T0 + 0.04)
    assert p6.step().reason == "abandoned"
    assert rings["targets"].newest_full() is None


def test_a_release_drops_the_engagement_and_the_next_press_reanchors(rings, wide):
    held(rings, Q0)
    p6 = make(rings, wide)
    press(rings)
    row(rings, T0 + 0.04, delta_xyz=(0.05, 0, 0))
    assert p6.step().written
    press(rings)                            # even: released
    assert p6.step().reason == "released" and p6.anchor_pose is None
    moved = Q0.copy()
    moved[0] += 0.05
    held(rings, moved, t=T0 + 0.1)
    press(rings)                            # odd again: a new anchor where the arm is held
    assert p6.step().reason == "idle"
    assert p6.anchor_pose[0] == pytest.approx(moved[0])


def test_the_real_ik_reproduces_the_anchor_for_a_zero_delta(rings, cfg):
    pytest.importorskip("mink")
    from elrobot_deploy.runtime.ik import build
    solver = build(cfg)
    q = cfg.poses.home
    held(rings, q)
    p6 = make(rings, cfg, solver)
    press(rings)
    row(rings, T0 + 0.04, grip=q[schema.GRIPPER])
    assert p6.step().written
    t = rings["targets"].newest_full()
    assert t["err"] < 1e-3
    assert np.allclose(t["q"], q, atol=math.radians(0.5))

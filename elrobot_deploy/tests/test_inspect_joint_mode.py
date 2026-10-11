"""Joint-mode rollouts: no commanded pose, except where an operator intervened.

A joint policy commands angles; its ticks carry a NaN pose and there is no
gripper ghost to draw (nor its TCP marker and trail, which are part of it), and
nothing for the IK bench to re-solve. A PICO takeover writes EEF poses whatever
the policy's mode, so those stretches keep the ghost. The commanded joints are
drawn throughout, on the trunk the record says P6 solved in.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1d_deploy.dashboards.inspect import loader, scene, timeline
from g1d_deploy.dashboards.inspect.modes import ik as ik_mode

from . import inspect_fixtures as fx

pytest.importorskip("mujoco")


def _scene(run):
    ro = loader.load(run)
    tl = timeline.build(ro)
    return ro, tl, scene.build(ro, tl)


def test_an_eef_rollout_draws_its_pose_everywhere_it_was_commanded(tmp_path):
    ro, tl, sc = _scene(fx.write(tmp_path / "r", example="sine_eef"))
    try:
        m = sc.meta()
        assert m["action_mode"] == "eef" and ro.action_mode == "eef"
        assert m["ghost_frames"] > 0 and m["intervention_frames"] == 0
        assert m["ghost_frames"] == m["solved_frames"]
    finally:
        ro.close()


def test_a_joint_rollout_has_no_commanded_pose(tmp_path):
    ro, tl, sc = _scene(fx.write(tmp_path / "r", example="sine_joint"))
    try:
        m = sc.meta()
        assert m["action_mode"] == "joint"
        assert m["ghost_frames"] == 0          # the page greys the switch
        assert m["intervention_frames"] == 0
        # ...while the commanded joints are drawn wherever a tick was in force.
        assert m["solved_frames"] > 0
        assert not sc.ghost_valid.any()
        assert ro.summary()["action_mode"] == "joint"
        ok, why = ik_mode.applicable(ro)
        assert not ok and "joint-mode" in why
    finally:
        ro.close()


def test_an_intervention_is_drawn_in_a_joint_rollout(tmp_path):
    """PICO stretches carry EEF poses: the ghost is drawn there and only there,
    and the IK bench has something to re-solve."""
    ro, tl, sc = _scene(fx.write(tmp_path / "r", example="sine_joint",
                                 intervene=[(30, 45)]))
    try:
        m = sc.meta()
        assert m["action_mode"] == "joint"
        assert m["intervention_frames"] > 0
        assert m["ghost_frames"] == m["intervention_frames"]
        rows = tl.index["command"]
        on = np.flatnonzero(sc.ghost_valid)
        ks = ro.body_targets["k"][rows[on]]
        assert ((ks >= 30) & (ks < 45)).all()
        assert ro.summary()["intervened_ticks"] == 15
        assert ik_mode.applicable(ro) == (True, "")
    finally:
        ro.close()


def test_the_commanded_joints_stand_on_the_measured_trunk(tmp_path):
    """No switch any more: the record's `waist_ik` (the measured bow) is what
    the published arm angles are posed on."""
    ro, tl, sc = _scene(fx.write(tmp_path / "r", example="sine_joint"))
    try:
        assert sc.solved_pinned.size, "the record carries waist_ik"
        assert sc.meta()["solved_pinned"]
    finally:
        ro.close()

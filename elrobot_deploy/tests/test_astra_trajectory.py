"""The Astra harness's pure parts: frames and units, and the trajectory law."""

from __future__ import annotations

import math

import numpy as np
import pytest

from elrobot_deploy.astra.frames import (
    GripperMap,
    TaskFrame,
    mat_from_rotvec,
    mat_from_rpy,
    rotation_angle,
    rotvec_from_mat,
    rpy_from_mat,
)
from elrobot_deploy.astra.trajectory import Limits, Reference, build, parse_action
from elrobot_deploy.config_loading.config import load
from elrobot_deploy.config_loading.errors import ConfigError

DT = 1 / 60


def wp(xyz, rpy=(0, 0, 0), g=40.0):
    return {"xyz": list(xyz), "rpy": list(rpy), "gripper_mm": g}


def plan_of(waypoints, ref=None, speed="normal", stretch=None, lim=None):
    lim = lim or Limits()
    ref = ref or Reference(np.array([0.25, 0.0, 0.1]), np.eye(3), 55.0)
    wps, sp, errors = parse_action({"waypoints": waypoints, "speed": speed}, lim)
    assert errors == []
    return build(ref, wps, lim, DT, speed=sp, stretch=stretch)


# --- frames ---------------------------------------------------------------------------------


def test_rpy_is_fixed_axis_xyz_and_round_trips():
    # pitch > 0 tips the TCP's x (the approach direction) down: R = Rz Ry Rx.
    R = mat_from_rpy([0, 30, 0])
    assert R[:, 0] == pytest.approx([math.cos(math.radians(30)), 0, -0.5])
    for rpy in ([10, -20, 30], [179, 5, -170], [0, 89.9, 0], [-45, 0, 120]):
        assert rpy_from_mat(mat_from_rpy(rpy)) == pytest.approx(rpy, abs=1e-6)


def test_rotation_vectors_round_trip_up_to_180_degrees():
    rng = np.random.default_rng(1)
    for angle in (0.0, 1e-6, 0.5, 2.0, math.pi - 1e-4, math.pi):
        axis = rng.normal(size=3)
        r = axis / np.linalg.norm(axis) * angle
        R = mat_from_rotvec(r)
        assert rotation_angle(R) == pytest.approx(angle, abs=1e-6)
        assert mat_from_rotvec(rotvec_from_mat(R)) == pytest.approx(R, abs=1e-6)


def test_the_task_frame_is_the_identity_for_our_urdf_and_turns_another():
    # Ours: base_link is REP-103 already (x forward, y left), so nothing is turned.
    assert TaskFrame.from_config(load()).M == pytest.approx(np.eye(3))
    # A description whose base has +y forward (upstream's) needs only base_forward / base_up.
    tf = TaskFrame(base_forward=(0, 1, 0), base_up=(0, 0, 1))
    xyz, _ = tf.to_task([0.0, 0.3, 0.1, 1, 0, 0, 0])
    assert xyz == pytest.approx([0.3, 0.0, 0.1])                     # 30 cm forward, 10 up
    left = tf.to_task([-0.2, 0.0, 0.0, 1, 0, 0, 0])[0]
    assert left == pytest.approx([0.0, 0.2, 0.0])                    # that base's -x is left
    pose = tf.to_base([0.1, 0.2, 0.3], mat_from_rpy([0, 45, 10]))
    back = tf.describe(pose)
    assert back[0] == pytest.approx([0.1, 0.2, 0.3])
    assert back[1] == pytest.approx([0, 45, 10], abs=1e-6)


def test_the_gripper_map_is_linear_open_at_the_lower_limit():
    g = GripperMap(math.radians(3.0), math.radians(139.8), open_mm=55.0, closed_mm=0.0)
    assert g.mm(math.radians(3.0)) == pytest.approx(55.0)
    assert g.mm(math.radians(139.8)) == pytest.approx(0.0)
    assert g.q(27.5) == pytest.approx(math.radians((3.0 + 139.8) / 2))
    assert g.mm(g.q(12.3)) == pytest.approx(12.3)
    assert g.q(80.0) == pytest.approx(math.radians(3.0))             # clipped to the range


def test_fk_at_zero_and_home_reads_as_the_readme_says():
    pytest.importorskip("mujoco")
    from elrobot_deploy.astra.workspace import home_tcp
    from elrobot_deploy.robot.elrobot_model import ElRobotModel, mat_to_pose
    cfg = load()
    tf = TaskFrame.from_config(cfg)
    xyz, rpy = tf.describe(mat_to_pose(ElRobotModel(cfg.urdf).fk(np.zeros(7))))
    assert xyz[0] > 0.3 and abs(xyz[1]) < 0.01                      # straight forward
    assert rpy == pytest.approx([0, 0, 0], abs=2.0)                  # TCP axes = base axes
    xyz, rpy = home_tcp(cfg)
    assert xyz == pytest.approx([0.25, 0.0, 0.10], abs=0.01)         # the profile's words
    assert rpy[1] == pytest.approx(45.0, abs=1.5)                    # pitched 45 deg down


# --- the input ------------------------------------------------------------------------------


def test_parse_action_lists_every_problem_by_waypoint_and_field():
    _, _, errors = parse_action({"waypoints": [
        {"xyz": [0, 0], "rpy": [0, 0, 175], "gripper_mm": 99, "left": 1},
        {"xyz": [0, 0, float("nan")], "rpy": [0, 0, 0]},
        wp([0, 0, 0], g=True)], "speed": "fast", "notes": "x"}, Limits())
    text = "\n".join(errors)
    for needle in ("unknown argument(s) ['notes']", "speed: 'fast'", "waypoint 1: unknown field",
                   "waypoint 1.xyz", "waypoint 1.gripper_mm: 99 is outside 0 ... 55 mm",
                   "waypoint 1.rpy: rotates 175 deg", "waypoint 2: missing gripper_mm",
                   "waypoint 3.gripper_mm: expected a number"):
        assert needle in text, text
    assert parse_action({"waypoints": []}, Limits())[2]
    assert parse_action([1, 2], Limits())[2]
    wps, speed, errors = parse_action({"waypoints": [wp([0.01, 0, 0])]}, Limits())
    assert errors == [] and speed == "normal" and len(wps) == 1


# --- the law ----------------------------------------------------------------------------------


def test_waypoints_are_offsets_in_the_reference_tcp_frame():
    R_ref = mat_from_rpy([0, 45, 90])                    # some tilted reference
    ref = Reference(np.array([0.1, 0.2, 0.3]), R_ref, 30.0)
    plan = plan_of([wp([0.05, 0, 0], g=50), wp([0.05, 0.02, 0], (0, 0, 30), g=10)], ref=ref)
    # Every offset from the SAME reference, not chained; knots hit exactly, on the grid.
    p1, _, _ = plan.targets[0]
    assert p1 == pytest.approx(ref.p + R_ref @ [0.05, 0, 0])
    p2, R2, _ = plan.targets[1]
    assert p2 == pytest.approx(ref.p + R_ref @ [0.05, 0.02, 0])
    assert R2 == pytest.approx(R_ref @ mat_from_rpy([0, 0, 30]))
    assert plan.p[plan.knots[1]] == pytest.approx(p1)
    assert plan.p[plan.knots[2]] == pytest.approx(p2)
    assert plan.R[plan.knots[2]] == pytest.approx(R2, abs=1e-9)
    assert (plan.g[plan.knots[1]], plan.g[plan.knots[2]]) == (50.0, 10.0)
    assert plan.p[0] == pytest.approx(ref.p) and plan.g[0] == 30.0


def test_the_path_starts_and_ends_at_rest_and_never_overshoots():
    plan = plan_of([wp([0.04, 0, 0]), wp([0.04, 0, -0.04]), wp([0.08, 0.03, -0.04], g=0)])
    v = np.linalg.norm(np.diff(plan.p, axis=0), axis=1) / DT
    assert v[0] < 0.005 and v[-1] < 0.005                   # at rest at both ends
    assert v[1:-1].min() > 0                                # never stops at a waypoint
    for a, b in zip(plan.knots[:-1], plan.knots[1:]):
        seg = plan.p[a:b + 1]
        lo = np.minimum(plan.p[a], plan.p[b]) - 1e-9
        hi = np.maximum(plan.p[a], plan.p[b]) + 1e-9
        assert np.all(seg >= lo) and np.all(seg <= hi)      # no dip below the lower waypoint


def test_the_timing_keeps_every_peak_inside_the_limits():
    lim = Limits()
    plan = plan_of([wp([0.15, 0, 0], (0, 0, 60)), wp([0.15, 0.1, -0.05], (0, 0, 60), g=0)],
                   lim=lim)
    v = np.linalg.norm(np.diff(plan.p, axis=0), axis=1) / DT
    a = np.linalg.norm(np.diff(plan.p, n=2, axis=0), axis=1) / DT ** 2
    w = np.array([rotation_angle(plan.R[i].T @ plan.R[i + 1])
                  for i in range(plan.n - 1)]) / DT
    g = np.abs(np.diff(plan.g)) / DT
    assert v.max() <= lim.v * 1.01 and a.max() <= lim.a * 1.05
    assert w.max() <= lim.w * 1.01 and g.max() <= lim.g * 1.01
    assert v.max() >= lim.v * 0.7                           # and actually uses them
    assert all(s >= lim.min_segment_s - 1e-9 for s in plan.seg_s)
    assert np.allclose(plan.seg_s / DT, np.round(plan.seg_s / DT))   # knots on the grid


def test_slow_takes_twice_as_long():
    w = [wp([0.06, 0, 0]), wp([0.06, 0, -0.03], g=0)]
    assert plan_of(w, speed="slow").duration == pytest.approx(2 * plan_of(w).duration, abs=DT)


def test_stretch_lengthens_only_the_segments_it_names():
    w = [wp([0.03, 0, 0]), wp([0.06, 0, 0]), wp([0.09, 0, 0])]
    base = plan_of(w)
    calls = []

    def stretch(plan):
        calls.append(plan.seg_s.copy())
        return np.array([1.0, 2.0, 1.0]) if len(calls) == 1 else np.ones(3)

    slow = plan_of(w, stretch=stretch)
    assert slow.seg_s[1] >= 2 * base.seg_s[1] - DT
    assert slow.seg_s[0] == pytest.approx(base.seg_s[0], abs=2 * DT)
    assert len(calls) == 2


def test_a_still_waypoint_closing_the_gripper_takes_the_gripper_time():
    plan = plan_of([wp([0, 0, 0], g=0.0)])                   # stay, close from 55 to 0 mm
    assert np.allclose(plan.p, plan.p[0])
    assert plan.duration >= 1.5 * 55 / Limits().g - DT       # 60 mm/s peak, a cubic's 1.5x


# --- the profile ----------------------------------------------------------------------------


def test_the_astra_and_record_tables_load_and_refuse_nonsense():
    cfg = load()
    assert cfg.astra.gripper_open_mm == 55.0 and cfg.astra.cameras == ("exo", "wrist")
    assert cfg.astra.anchor_camera == "wrist"
    assert cfg.record.dir.name == "records" and cfg.record.jpeg_quality == 90
    assert load(sets=["astra.max_speed_m_s=0.05"]).astra.max_speed_m_s == 0.05
    with pytest.raises(ConfigError):
        load(sets=["astra.gripper_open_mm=-1"])
    with pytest.raises(ConfigError):
        load(sets=["astra.max_rotation_deg=200"])
    with pytest.raises(ConfigError, match="no key"):
        load(sets=["astra.speed=1"])

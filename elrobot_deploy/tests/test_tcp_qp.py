"""The ElRobot model (`robot/elrobot_model.py`) and its per-tick TCP IK (`runtime/ik/tcp_qp.py`)."""

import sys

import numpy as np
import pytest

pytest.importorskip("mink")

from elrobot_deploy.robot.elrobot_model import (
    ARM_JOINTS,
    DEFAULT_URDF,
    ElRobotModel,
    mat_to_pose,
    pose_to_mat,
)
from elrobot_deploy.runtime.ik.tcp_qp import TcpSolver

REPO = DEFAULT_URDF.parents[2]


@pytest.fixture(scope="module")
def robot():
    return ElRobotModel()


@pytest.fixture
def solver(robot):
    return TcpSolver(robot)


def _tabletop_target(p, pitch_deg):
    """TCP at `p` (base frame, m), jaws pointing forward and pitched down, y left."""
    a = np.radians(pitch_deg)
    T = np.eye(4)
    T[:3, 3] = p
    x = np.array([np.cos(a), 0.0, -np.sin(a)])
    y = np.array([0.0, 1.0, 0.0])
    T[:3, :3] = np.c_[x, y, np.cross(x, y)]
    return T


def _settle(T, q=None):
    free = TcpSolver(qdot_max=100.0)
    q = np.radians([0, -10, 60, 40, 0, 20, 0]) if q is None else q
    for _ in range(300):
        q, err, _ = free.solve(T, q, budget=0.05)
    return q, err


def test_model_matches_the_urdf_fk(robot):
    """MuJoCo's reading of the URDF and the repo's own URDF FK agree on the TCP and camera."""
    sys.path.insert(0, str(REPO))
    from elrobot.kinematics import Robot
    urdf_fk = Robot(DEFAULT_URDF)
    rng = np.random.default_rng(1)
    lim = robot.limits()
    for _ in range(20):
        q = rng.uniform(lim[:, 0], lim[:, 1])
        qd = dict(zip(ARM_JOINTS, q))
        for body in ("tcp", "wrist_camera_optical"):
            np.testing.assert_allclose(robot.fk(q, body), urdf_fk.fk(qd, body), atol=1e-9)


def test_model_has_the_calibrated_limits_and_no_gripper_dof(robot):
    assert robot.model.nq == 7
    lim = np.degrees(robot.limits())
    assert lim[0] == pytest.approx([-85.2, 85.2], abs=0.1)
    assert lim[6] == pytest.approx([-165.9, 150.8], abs=0.1)


def test_pose_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(10):
        q = rng.normal(size=4)
        p = np.r_[rng.normal(size=3), q / np.linalg.norm(q)]
        p = mat_to_pose(pose_to_mat(p))
        np.testing.assert_allclose(pose_to_mat(p), pose_to_mat(mat_to_pose(pose_to_mat(p))), atol=1e-12)
        assert p[3] >= 0


def test_reaches_fk_targets_from_a_nearby_seed(solver, robot):
    """A seed one tick away (inside the speed cap) converges to the 0.1 mm early stop."""
    rng = np.random.default_rng(3)
    lim = robot.limits()
    cap = solver.qdot_max * solver.dt
    for _ in range(40):
        q = rng.uniform(lim[:, 0] * 0.8, lim[:, 1] * 0.8)
        seed = q + rng.uniform(-0.5, 0.5, 7) * cap
        qs, err, ok = solver.solve(robot.fk(q), seed, budget=0.5)
        assert ok
        assert err < 2e-4


def test_accepts_a_pose_vector(solver, robot):
    q = np.radians([0, -20, 40, 50, 10, 30, -10])
    qs, err, ok = solver.solve(mat_to_pose(robot.fk(q)), q, budget=0.5)
    assert ok and err < 1e-4


def test_never_leaves_the_joint_limits(solver, robot):
    """Unreachable targets: the arm goes as far as it can, inside its limits, ok=True."""
    lim = robot.limits()
    rng = np.random.default_rng(4)
    q = np.radians([0, -20, 60, 40, 0, 30, 0])
    for _ in range(60):
        T = np.eye(4)
        T[:3, 3] = rng.uniform([-0.6, -0.6, -0.3], [0.6, 0.6, 0.7])
        q, err, ok = solver.solve(T, q)
        assert ok
        assert np.all(q >= lim[:, 0] - 1e-6) and np.all(q <= lim[:, 1] + 1e-6)


def test_speed_cap_holds_over_the_whole_tick(solver, robot):
    """However far the target and however many iterates, one tick moves each joint at most
    qdot_max * dt from the seed."""
    q = np.radians([0, -20, 60, 40, 0, 30, 0])
    far = robot.fk(np.radians([60, 30, -40, 20, 90, -50, 100]))
    cap = solver.qdot_max * solver.dt
    for _ in range(20):
        qn, err, ok = solver.solve(far, q, budget=0.5)
        assert np.all(np.abs(qn - q) <= cap + 1e-9)
        q = qn


def test_budget_runs_out(solver, robot):
    q = np.radians([0, -20, 60, 40, 0, 30, 0])
    far = robot.fk(np.radians([60, 30, -40, 20, 90, -50, 100]))
    qs, err, ok = solver.solve(far, q, budget=0.0)
    assert not ok
    assert err > 0.01              # the residual is still reported


@pytest.mark.parametrize("translation", [0.0, 0.00005])
def test_rotation_is_solved_even_when_position_is_already_inside_tolerance(robot, translation):
    """The pen-writing failure: position-only early exit silently skipped a rotation."""
    from elrobot_deploy.astra.frames import mat_from_rpy

    solver = TcpSolver(robot)
    q = np.radians([0, -20, 40, 50, 10, 30, -10])
    solver.set_nominal(q)
    target = robot.fk(q)
    target[:3, :3] = target[:3, :3] @ mat_from_rpy([0, -15, 0])
    target[2, 3] += translation
    first, _, ok = solver.solve(target, q, budget=0.5)
    assert ok and np.max(np.abs(first - q)) > 1e-5
    assert np.all(np.abs(first - q) <= solver.qdot_max * solver.dt + 1e-9)
    q = first
    for _ in range(100):
        q, _, ok = solver.solve(target, q, budget=0.5)
        assert ok
    pos, rot = solver.tcp_error(target)
    assert pos < solver.early_stop
    assert rot < solver.orientation_early_stop


def test_disabled_orientation_task_does_not_prevent_early_exit(robot):
    from elrobot_deploy.astra.frames import mat_from_rpy

    solver = TcpSolver(robot, orientation_cost=0)
    q = np.radians([0, -20, 40, 50, 10, 30, -10])
    target = robot.fk(q)
    target[:3, :3] = target[:3, :3] @ mat_from_rpy([0, -65, 0])
    solved, err, ok = solver.solve(target, q, budget=0)
    assert ok and err < solver.early_stop
    np.testing.assert_array_equal(solved, q)


def test_tracks_a_tabletop_circle(solver):
    """The usual case: gripper pitched down, a 5 cm circle at 0.5 Hz, 60 Hz ticks."""
    T0 = _tabletop_target([0.30, 0.0, 0.05], 60)
    q, err = _settle(T0)
    assert err < 2e-4
    errs = []
    for plane in ("side-up", "forward-side"):
        for k in range(240):
            t = k / 60
            c, s = np.cos(np.pi * t) - 1, np.sin(np.pi * t)
            T = T0.copy()
            T[:3, 3] += 0.05 * (np.array([0, -c, s]) if plane == "side-up"
                                else np.array([s, -c, 0]))
            q, err, ok = solver.solve(T, q)
            assert ok
            errs.append(solver.tcp_error(T))
    errs = np.array(errs)
    assert errs[:, 0].max() < 5e-4                 # 0.5 mm
    assert np.degrees(errs[:, 1].max()) < 0.5


def test_set_nominal_moves_the_pull(solver):
    q = np.radians([10, -30, 50, 40, 5, 20, -5])
    solver.set_nominal(q)
    np.testing.assert_allclose(solver.nominal_q, q)
    full = solver.nominal.target_q
    np.testing.assert_allclose(full[solver.robot.qadr], q)

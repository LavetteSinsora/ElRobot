"""Our ElRobot description (description/elrobot_ours): limits, camera mount, TCP and gripper."""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "description" / "elrobot_ours"))

import make_description as md  # noqa: E402
from elrobot.kinematics import ARM_JOINTS, Robot  # noqa: E402


@pytest.fixture(scope="module")
def urdf():
    return ET.parse(md.OUT_URDF).getroot()


@pytest.fixture(scope="module")
def upstream():
    return ET.parse(md.UPSTREAM).getroot()


@pytest.fixture(scope="module")
def robot():
    return Robot(md.OUT_URDF)


def test_limits_match_calibration(urdf):
    """The URDF is regenerated after every recalibration; this catches a stale one."""
    limits, _ = md.calibrated_limits()
    for k, (lo, hi) in limits.items():
        el = urdf.find(f"joint[@name='rev_motor_0{k}']/limit")
        assert float(el.get("lower")) == pytest.approx(lo, abs=1e-4), k
        assert float(el.get("upper")) == pytest.approx(hi, abs=1e-4), k


def test_gripper_uses_the_hand_swept_closed_stop(urdf):
    _, closed = md.calibrated_limits()
    assert math.degrees(closed) == pytest.approx(142.8, abs=0.1)
    upper = float(urdf.find("joint[@name='rev_motor_08']/limit").get("upper"))
    assert math.degrees(closed - upper) == pytest.approx(md.GRIPPER_MARGIN_DEG, abs=0.01)


def test_mount_fit_reproduces_the_cad_frame():
    """Fitted from the printed STL alone, the S frame must give the script's own numbers."""
    _, _, checks = md.body_print_to_S(md.read_stl(md.PRINTED / "camera_mount_yoke_facet_body.stl"))
    a = math.radians(md.VIEW_TILT_DEG)
    np.testing.assert_allclose(checks["board_front_S"], md.BOARD_FRONT_S, atol=0.01)
    np.testing.assert_allclose(checks["view_S"], [math.cos(a), 0, -math.sin(a)], atol=1e-3)
    np.testing.assert_allclose(checks["holes_S"], md.FOOT_HOLES_S, atol=0.01)


def test_mount_feet_sit_on_the_base_holes(upstream):
    T = md.mount_pose(upstream)
    holes, normal = md.base_holes(upstream)
    feet = (T[:3, :3] @ (md.FOOT_HOLES_S * 0.001).T).T + T[:3, 3]
    np.testing.assert_allclose(feet, holes, atol=1e-6)
    assert T[:3, 2] @ normal == pytest.approx(1.0)


def test_camera_looks_forward_and_down_at_zero_pose(robot):
    T = robot.fk({}, "wrist_camera_optical")
    view, right = T[:3, 2], T[:3, 0]
    assert view[1] > 0.85                                       # along the arm (+y)
    assert math.degrees(math.asin(-view[2])) == pytest.approx(md.VIEW_TILT_DEG, abs=2)
    assert right[0] > 0.99                                      # image right = the robot's right (+x)
    tcp = robot.fk({}, "tcp")[:3, 3]
    assert T[2, 3] > tcp[2] + 0.04                              # it sits above the gripper


def _jaw_tip_points(robot, q, link):
    """The last 5 mm of a jaw's mesh, in the base frame."""
    root = ET.parse(md.OUT_URDF).getroot()
    vis = root.find(f"link[@name='{link}']/visual")
    off = np.array([float(v) for v in vis.find("origin").get("xyz").split()])
    P = md.read_stl((md.OUT_URDF.parent / vis.find("geometry/mesh").get("filename")).resolve())
    P = P.reshape(-1, 3) * 0.001 + off
    T = robot.fk(q, link)
    P = P @ T[:3, :3].T + T[:3, 3]
    x_tcp = robot.fk(q, "tcp")[:3, 0]
    reach = P @ x_tcp
    return P[reach > reach.max() - 0.005]


def test_tcp_is_between_the_jaw_tips_and_does_not_move(robot):
    for q8 in (0.0, 1.0, 2.0):
        q = {"rev_motor_08": q8}
        T = robot.fk(q, "tcp")
        a = _jaw_tip_points(robot, q, "Gripper_Jaw_02_v1_1") @ T[:3, 1]
        b = _jaw_tip_points(robot, q, "Gripper_Jaw_01_v1_1") @ T[:3, 1]
        tcp_y = T[:3, 3] @ T[:3, 1]
        # the TCP sits halfway between the two jaws' inner faces along its y (left) axis
        assert (a.min() + b.max()) / 2 == pytest.approx(tcp_y, abs=0.002)
    np.testing.assert_allclose(robot.fk({"rev_motor_08": 0.0}, "tcp"),
                               robot.fk({"rev_motor_08": 2.0}, "tcp"))


def test_jaws_meet_at_the_closed_stop(robot):
    _, closed = md.calibrated_limits()
    q = {"rev_motor_08": closed}
    T = robot.fk(q, "tcp")
    a = _jaw_tip_points(robot, q, "Gripper_Jaw_02_v1_1") @ T[:3, 1]
    b = _jaw_tip_points(robot, q, "Gripper_Jaw_01_v1_1") @ T[:3, 1]
    assert a.min() - b.max() == pytest.approx(0.0, abs=0.001)    # inner faces touch


def test_tcp_axes_out_left_up_at_zero_pose(robot):
    R = robot.fk({}, "tcp")[:3, :3]
    assert R[:, 0] @ [0, 1, 0] > 0.99                           # x out of the jaws (+y)
    assert R[:, 1] @ [-1, 0, 0] > 0.99                          # y left (-x)
    assert R[:, 2] @ [0, 0, 1] > 0.99                           # z up


def test_arm_chain_unchanged_from_upstream(urdf, upstream):
    """Only limits differ on the arm: origins and axes are upstream's."""
    for name in ARM_JOINTS[:7]:
        ours, theirs = (r.find(f"joint[@name='{name}']") for r in (urdf, upstream))
        assert ours.find("origin").attrib == theirs.find("origin").attrib
        assert ours.find("axis").attrib == theirs.find("axis").attrib

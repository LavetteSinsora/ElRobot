import numpy as np
import pytest

from elrobot.kinematics import Robot


@pytest.fixture(scope="module")
def robot():
    return Robot()


def pos_mm(robot, q_deg, joint):
    q = {f"rev_motor_0{i}": np.radians(v) for i, v in q_deg.items()}
    return robot.joint_frame(q, joint)[:3, 3] * 1000


def test_zero_pose_is_upright_upper_arm_and_horizontal_forearm(robot):
    shoulder = pos_mm(robot, {}, "rev_motor_02")
    elbow = pos_mm(robot, {}, "rev_motor_03")
    gripper = pos_mm(robot, {}, "rev_motor_08")
    np.testing.assert_allclose(elbow - shoulder, [0, -0.8, 91.3], atol=0.5)
    assert gripper[1] > 300 and abs(gripper[2] - 218.5) < 1


def test_positive_shoulder_tips_arm_forward(robot):
    elbow = pos_mm(robot, {2: 90}, "rev_motor_03")
    shoulder = pos_mm(robot, {}, "rev_motor_02")
    d = elbow - shoulder
    assert d[1] > 90 and abs(d[2]) < 3  # upper arm now points along +y


def test_rest_pose_folds_the_forearm_back_over_the_upper_arm(robot):
    """Rest pose (poses.py): shoulder -90 (upper arm back), elbow 0, wrist pitch +90."""
    q = {2: -90, 4: 90}
    shoulder = pos_mm(robot, q, "rev_motor_02")
    elbow = pos_mm(robot, q, "rev_motor_03")
    wrist = pos_mm(robot, q, "rev_motor_05")
    gripper = pos_mm(robot, q, "rev_motor_08")
    assert elbow[1] < shoulder[1] - 85      # elbow behind the shoulder
    assert gripper[1] > shoulder[1] + 90    # gripper out in front (~98 mm)
    assert abs(gripper[2] - wrist[2]) < 3   # wrist-to-gripper still level


def test_mimic_jaws_follow_gripper(robot):
    q = {"rev_motor_08": 1.0}
    T0 = robot.fk({}, "Gripper_Jaw_01_v1_1")
    T1 = robot.fk(q, "Gripper_Jaw_01_v1_1")
    assert np.linalg.norm(T1[:3, 3] - T0[:3, 3]) == pytest.approx(0.0115, abs=1e-6)

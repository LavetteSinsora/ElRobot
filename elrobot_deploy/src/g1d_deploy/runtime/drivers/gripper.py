"""`rt/dex1/{left,right}/cmd` — the end-effector command binding.

P8 hands this module the `(2, 6)` hand block (`schema.HAND_DIM`) and nothing else: no
torque, no gains per call, no interpolation. Everything above the wire is the
loop's, everything at it is here.

**Which six numbers reach a motor is a property of the end effector, and these
topics are the Dex-1 gripper's.** `dex1_1_gripper_server -l -r` runs one server
per gripper, so there is one topic per hand and each carries a `MotorCmds_` with
a single motor — the jaw — at index 0. Only dim 0 of each hand's block is
therefore on the wire, and it is a gear rotation in radians, the quantity the
policy contract's `hand` speaks. A robot file with `[end_effector] type =
"brainco"` swaps in `rt/brainco/{left,right}/cmd` (`resolved.for_hand`): six
motors per topic, and still ONE grip value on the way in. `brainco_targets`
spreads it over them the way Unitree's own controller-trigger mode does, rather
than inventing a per-finger mapping — what a policy that emits six values would
command is undecided (`robot/end_effectors.py`).

The full `(2, 6)` still crosses the interface, and `FakeHandCmd` records all of
it. P8's stall arithmetic operates on the whole block, and the day the Brainco
mapping is settled it is this file that changes, not the loop.

The contract's `hand` for a policy author:
`docs/deploying_a_policy/policy_contract.md`, "The actions"; the hand types:
`docs/setup/robot_toml.md`, "`[end_effector]`".
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from g1d_deploy.runtime.shared_memory import schema

# Each per-gripper topic carries exactly one motor, the jaw (the vendor stack's
# `Gripper_Single_JointIndex.kGripper`). Which gripper is the topic's business,
# not the index's — `hw.topic_hand_cmd[h]` is hand `h`'s.
JAW_MOTOR_ID = 0

# The vendor's gripper gains (`Dex1_Gripper_Controller.connect`). They are wire
# facts about one actuator, not a tuning knob: the robot file's `gains.kp/kd`
# are the 29 body joints and the gripper is not one of them.
GRIPPER_KP = 10.0
GRIPPER_KD = 0.05

#: BrainCo: motors per topic (thumb, thumb-aux, index, middle, ring,
#: pinky), and the one field beside `q` Unitree's own controller sets.
BRAINCO_MOTORS = 6
BRAINCO_DQ = 1.0


def brainco_targets(grip: float) -> np.ndarray:
    """One grip value in [0, 1] (0 = open) -> the six BrainCo motor targets.

    Unitree's own controller-trigger mapping (xr_teleoperate
    `Brainco_Controller_ctrl`) with the trigger replaced by our grip value:
    motor 0 waits for the second half of the squeeze, motor 1 closes over the
    first half and stops at 0.7, and the four fingers follow 1:1 up to 0.98.
    (The vendor drives the index from the grip button instead; there is one
    value here, so it follows the others.)
    """
    g = float(grip)
    finger = float(np.clip(g, 0.0, 0.98))
    return np.array([float(np.clip((g - 0.5) / 0.5, 0.0, 0.98)),
                     float(np.clip(g / 0.5, 0.0, 0.7)),
                     finger, finger, finger, finger])


class HandCmd(Protocol):
    """What P8 depends on."""

    def connect(self) -> None: ...
    def publish(self, cmd_2x6: np.ndarray) -> None: ...
    def close(self) -> None: ...


def _accept(cmd_2x6) -> np.ndarray:
    cmd = np.asarray(cmd_2x6, dtype=np.float64)
    if cmd.shape != (schema.N_HANDS, schema.HAND_DIM):
        raise ValueError(
            f"hand command must be ({schema.N_HANDS}, {schema.HAND_DIM}), "
            f"got {cmd.shape}")
    return cmd


class FakeHandCmd:
    """Records what was published; commands nothing."""

    def __init__(self, hw):
        self.hw = hw
        self.connected = False
        self.closed = False
        self.published: list[np.ndarray] = []

    def connect(self) -> None:
        self.connected = True

    def publish(self, cmd_2x6) -> None:
        if not self.connected:
            raise RuntimeError("publish before connect()")
        self.published.append(_accept(cmd_2x6).copy())

    def close(self) -> None:
        self.closed = True

    @property
    def cmd(self) -> np.ndarray:
        """(n, 2, 6) — every command published, in order."""
        if not self.published:
            return np.zeros((0, schema.N_HANDS, schema.HAND_DIM))
        return np.array(self.published)


class DdsHandCmd:
    """The real publisher: one `MotorCmds_` per hand on `hw.topic_hand_cmd[h]`.

    Each message is built once and mutated in place per publish, for the same
    reason as `arm_command.DdsLowCmd`: at 200 Hz the allocation and the gain writes
    are pure overhead, and the firmware re-reads every field anyway.

    Both hands are published on every call, in `[left, right]` order. The idle
    hand is not skipped: the Dex-1 server holds the last command it was given,
    so a gripper that stops being addressed holds whatever it last heard — so
    both are written every tick, precisely so it is never in doubt.
    """

    def __init__(self, hw):
        self.hw = hw
        self.connected = False
        self._brainco = getattr(hw, "hand", "dex1") == "brainco"
        self._pubs: list = []
        self._msgs: list = []

    def connect(self) -> None:
        from g1d_deploy.runtime.drivers.dds.channel import ChannelPublisher
        from g1d_deploy.runtime.drivers.dds.idl.unitree_go import MotorCmds_
        from g1d_deploy.runtime.drivers.dds.idl.unitree_go import (default_MotorCmd_ as unitree_go_msg_dds__MotorCmd_)

        from g1d_deploy.runtime.drivers.dds import domain as dds
        dds.init(self.hw.dds_domain, self.hw.dds_iface)
        for topic in self.hw.topic_hand_cmd:
            pub = ChannelPublisher(topic, MotorCmds_)
            pub.Init()

            msg = MotorCmds_()
            if self._brainco:
                # As Unitree's own BrainCo controller sets them: `q` and `dq`,
                # no gains — the hand's own driver owns its position loop.
                msg.cmds = [unitree_go_msg_dds__MotorCmd_()
                            for _ in range(BRAINCO_MOTORS)]
                for cmd in msg.cmds:
                    cmd.kp = 0.0
                    cmd.kd = 0.0
                    cmd.dq = BRAINCO_DQ
                    cmd.tau = 0.0
                    cmd.q = 0.0
            else:
                msg.cmds = [unitree_go_msg_dds__MotorCmd_()
                            for _ in range(JAW_MOTOR_ID + 1)]
                cmd = msg.cmds[JAW_MOTOR_ID]
                cmd.kp = GRIPPER_KP
                cmd.kd = GRIPPER_KD
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.q = 0.0

            self._pubs.append(pub)
            self._msgs.append(msg)
        self.connected = True

    def publish(self, cmd_2x6) -> None:
        if not self.connected:
            raise RuntimeError("publish before connect()")
        cmd = _accept(cmd_2x6)
        for hand, (pub, msg) in enumerate(zip(self._pubs, self._msgs)):
            if self._brainco:
                for motor, q in zip(msg.cmds, brainco_targets(cmd[hand, 0])):
                    motor.q = float(q)
            else:
                msg.cmds[JAW_MOTOR_ID].q = float(cmd[hand, 0])
            pub.Write(msg)

    def close(self) -> None:
        self._pubs = []
        self._msgs = []
        self.connected = False


__all__ = ["BRAINCO_DQ", "BRAINCO_MOTORS", "GRIPPER_KD", "GRIPPER_KP",
           "JAW_MOTOR_ID", "DdsHandCmd", "FakeHandCmd", "HandCmd",
           "brainco_targets"]

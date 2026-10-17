"""`unitree_hg.msg.dds_` — the G1's low-level state and command.

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics").
BSD 3-Clause License: see `elrobot_deploy/runtime/drivers/dds/idl/__init__.py`.

Vendored from unitree_sdk2py 1.0.1 (`idl/unitree_hg/msg/dds_/_*_.py`, generated
by Eclipse Cyclone DDS idlc Python Backend, IDL v0.11.0) and its
`idl/default.py` constructors. Definitions unchanged; nested types referenced
by class (module docstring of `idl/__init__.py`).
"""

# No `from __future__ import annotations`: cyclonedds reads the field
# annotations as live types, and would try to import a string as a module.

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class IMUState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.IMUState_"):
    quaternion: types.array[types.float32, 4]
    gyroscope: types.array[types.float32, 3]
    accelerometer: types.array[types.float32, 3]
    rpy: types.array[types.float32, 3]
    temperature: types.int16


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorState_"):
    mode: types.uint8
    q: types.float32
    dq: types.float32
    ddq: types.float32
    tau_est: types.float32
    temperature: types.array[types.int16, 2]
    vol: types.float32
    sensor: types.array[types.uint32, 2]
    motorstate: types.uint32
    reserve: types.array[types.uint32, 4]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorCmd_"):
    mode: types.uint8
    q: types.float32
    dq: types.float32
    tau: types.float32
    kp: types.float32
    kd: types.float32
    reserve: types.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowState_"):
    version: types.array[types.uint32, 2]
    mode_pr: types.uint8
    mode_machine: types.uint8
    tick: types.uint32
    imu_state: IMUState_
    motor_state: types.array[MotorState_, 35]
    wireless_remote: types.array[types.uint8, 40]
    reserve: types.array[types.uint32, 4]
    crc: types.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowCmd_"):
    mode_pr: types.uint8
    mode_machine: types.uint8
    motor_cmd: types.array[MotorCmd_, 35]
    reserve: types.array[types.uint32, 4]
    crc: types.uint32


# --- zero-valued constructors (unitree_sdk2py `idl/default.py`) --------------


def default_MotorCmd_() -> MotorCmd_:
    return MotorCmd_(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def default_MotorState_() -> MotorState_:
    return MotorState_(0, 0.0, 0.0, 0.0, 0.0, [0, 0], 0.0, [0, 0], 0, [0, 0, 0, 0])


def default_IMUState_() -> IMUState_:
    return IMUState_([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0], 0)


def default_LowCmd_() -> LowCmd_:
    return LowCmd_(0, 0, [default_MotorCmd_() for _ in range(35)], [0, 0, 0, 0], 0)


def default_LowState_() -> LowState_:
    return LowState_([0, 0], 0, 0, 0, default_IMUState_(),
                     [default_MotorState_() for _ in range(35)],
                     [0] * 40, [0, 0, 0, 0], 0)


__all__ = ["IMUState_", "LowCmd_", "LowState_", "MotorCmd_", "MotorState_",
           "default_IMUState_", "default_LowCmd_", "default_LowState_",
           "default_MotorCmd_", "default_MotorState_"]

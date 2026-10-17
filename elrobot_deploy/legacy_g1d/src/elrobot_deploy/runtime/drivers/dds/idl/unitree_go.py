"""`unitree_go.msg.dds_` — the motor command/state lists the grippers speak
(the Dex-1 and the BrainCo alike: `rt/dex1/*/cmd|state`, `rt/brainco/*`).

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics").
BSD 3-Clause License: see `elrobot_deploy/runtime/drivers/dds/idl/__init__.py`.

Vendored from unitree_sdk2py 1.0.1 (`idl/unitree_go/msg/dds_/_*_.py`) and its
`idl/default.py` constructors. Definitions unchanged; nested types referenced
by class (module docstring of `idl/__init__.py`).
"""

# No `from __future__ import annotations`: cyclonedds reads the field
# annotations as live types, and would try to import a string as a module.

from dataclasses import dataclass, field

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmd_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorCmd_"):
    mode: types.uint8
    q: types.float32
    dq: types.float32
    tau: types.float32
    kp: types.float32
    kd: types.float32
    reserve: types.array[types.uint32, 3]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmds_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorCmds_"):
    cmds: types.sequence[MotorCmd_] = field(default_factory=lambda: [])


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorState_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorState_"):
    mode: types.uint8
    q: types.float32
    dq: types.float32
    ddq: types.float32
    tau_est: types.float32
    q_raw: types.float32
    dq_raw: types.float32
    ddq_raw: types.float32
    temperature: types.uint8
    lost: types.uint32
    reserve: types.array[types.uint32, 2]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorStates_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorStates_"):
    states: types.sequence[MotorState_] = field(default_factory=lambda: [])


# --- zero-valued constructors (unitree_sdk2py `idl/default.py`) --------------


def default_MotorCmd_() -> MotorCmd_:
    return MotorCmd_(0, 0.0, 0.0, 0.0, 0.0, 0.0, [0, 0, 0])


def default_MotorState_() -> MotorState_:
    return MotorState_(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, [0, 0])


__all__ = ["MotorCmd_", "MotorCmds_", "MotorState_", "MotorStates_",
           "default_MotorCmd_", "default_MotorState_"]

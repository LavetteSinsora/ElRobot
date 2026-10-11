"""`geometry_msgs.msg.dds_.Point32_` — the lift column's `rt/hispeed_state`
and `rt/cmd_hispeed`.

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics").
BSD 3-Clause License: see `elrobot_deploy/runtime/drivers/dds/idl/__init__.py`.

Vendored from unitree_sdk2py 1.0.1 (`idl/geometry_msgs/msg/dds_/_Point32_.py`)
and its `idl/default.py` constructor. Definition unchanged.
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
class Point32_(idl.IdlStruct, typename="geometry_msgs.msg.dds_.Point32_"):
    x: types.float32
    y: types.float32
    z: types.float32


def default_Point32_() -> Point32_:
    return Point32_(0.0, 0.0, 0.0)


__all__ = ["Point32_", "default_Point32_"]

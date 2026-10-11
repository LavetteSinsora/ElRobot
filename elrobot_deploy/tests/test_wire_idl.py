"""The vendored DDS types and channel layer against unitree_sdk2py itself.

`runtime/drivers/dds/idl/` replaces the SDK's generated message classes and `dds/channel.py` its
channel factory. The robot is on the other end of both, so "equivalent" has to
mean identical bytes, not similar code:

* every vendored type has the SDK's DDS type name, and the SAME XTypes type
  information and type map (what discovery exchanges and matches on);
* for the same field values, the serialized CDR bytes are identical — checked
  on randomized samples, in both directions (each side deserializes the
  other's bytes to equal fields);
* the LowCmd CRC equals the SDK's `CRC().Crc(msg)` (its native library on
  Linux) on randomized messages;
* the CycloneDDS config XML is the SDK's, character for character.

Needs `cyclonedds`; the comparisons need `unitree_sdk2py` too and are
skipped where it is not installed. Nothing here opens a DDS domain.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("cyclonedds")

from g1d_deploy.runtime.drivers.dds import channel, crc  # noqa: E402
from g1d_deploy.runtime.drivers.dds.idl import geometry_msgs, unitree_go, unitree_hg  # noqa: E402

sdk = pytest.importorskip("unitree_sdk2py")


def _sdk_types():
    from unitree_sdk2py.idl.geometry_msgs.msg import dds_ as sgm
    from unitree_sdk2py.idl.unitree_go.msg import dds_ as sgo
    from unitree_sdk2py.idl.unitree_hg.msg import dds_ as shg
    return {
        (unitree_hg, "IMUState_"): shg.IMUState_,
        (unitree_hg, "MotorState_"): shg.MotorState_,
        (unitree_hg, "MotorCmd_"): shg.MotorCmd_,
        (unitree_hg, "LowState_"): shg.LowState_,
        (unitree_hg, "LowCmd_"): shg.LowCmd_,
        (unitree_go, "MotorCmd_"): sgo.MotorCmd_,
        (unitree_go, "MotorCmds_"): sgo.MotorCmds_,
        (unitree_go, "MotorState_"): sgo.MotorState_,
        (unitree_go, "MotorStates_"): sgo.MotorStates_,
        (geometry_msgs, "Point32_"): sgm.Point32_,
    }


PAIRS = [pytest.param(getattr(mod, name), theirs,
                      id=f"{mod.__name__.rsplit('.', 1)[1]}.{name}")
         for (mod, name), theirs in _sdk_types().items()]


@pytest.mark.parametrize("ours, theirs", PAIRS)
def test_type_names_and_xtypes_information_are_the_sdks(ours, theirs):
    assert ours.__idl_typename__ == theirs.__idl_typename__
    for t in (ours, theirs):
        t.__idl__.populate()
        t.__idl__.fill_type_data()
    # (TypeInformation, TypeMapping) as serialized for discovery.
    assert ours.__idl__._xt_bytedata == theirs.__idl__._xt_bytedata


# --- randomized samples, built identically on both sides ----------------------


def _f32(rng):
    return float(np.float32(rng.uniform(-100, 100)))


def _u(rng, bits):
    return rng.randrange(0, 1 << bits)


def _i16(rng):
    return rng.randrange(-(1 << 15), 1 << 15)


def _hg_motor_cmd(rng):
    return dict(mode=_u(rng, 8), q=_f32(rng), dq=_f32(rng), tau=_f32(rng),
                kp=_f32(rng), kd=_f32(rng), reserve=_u(rng, 32))


def _hg_motor_state(rng):
    return dict(mode=_u(rng, 8), q=_f32(rng), dq=_f32(rng), ddq=_f32(rng),
                tau_est=_f32(rng), temperature=[_i16(rng), _i16(rng)],
                vol=_f32(rng), sensor=[_u(rng, 32), _u(rng, 32)],
                motorstate=_u(rng, 32), reserve=[_u(rng, 32) for _ in range(4)])


def _imu(rng):
    return dict(quaternion=[_f32(rng) for _ in range(4)],
                gyroscope=[_f32(rng) for _ in range(3)],
                accelerometer=[_f32(rng) for _ in range(3)],
                rpy=[_f32(rng) for _ in range(3)], temperature=_i16(rng))


def _go_motor_cmd(rng):
    return dict(mode=_u(rng, 8), q=_f32(rng), dq=_f32(rng), tau=_f32(rng),
                kp=_f32(rng), kd=_f32(rng), reserve=[_u(rng, 32) for _ in range(3)])


def _go_motor_state(rng):
    return dict(mode=_u(rng, 8), q=_f32(rng), dq=_f32(rng), ddq=_f32(rng),
                tau_est=_f32(rng), q_raw=_f32(rng), dq_raw=_f32(rng),
                ddq_raw=_f32(rng), temperature=_u(rng, 8), lost=_u(rng, 32),
                reserve=[_u(rng, 32), _u(rng, 32)])


def _build(which: str, types: dict, rng):
    """One sample of `which`, as `types` (ours or the SDK's) builds it."""
    if which == "LowCmd_":
        return types["LowCmd_"](
            mode_pr=_u(rng, 8), mode_machine=_u(rng, 8),
            motor_cmd=[types["HGMotorCmd_"](**_hg_motor_cmd(rng)) for _ in range(35)],
            reserve=[_u(rng, 32) for _ in range(4)], crc=_u(rng, 32))
    if which == "LowState_":
        return types["LowState_"](
            version=[_u(rng, 32), _u(rng, 32)], mode_pr=_u(rng, 8),
            mode_machine=_u(rng, 8), tick=_u(rng, 32),
            imu_state=types["IMUState_"](**_imu(rng)),
            motor_state=[types["HGMotorState_"](**_hg_motor_state(rng))
                         for _ in range(35)],
            wireless_remote=[_u(rng, 8) for _ in range(40)],
            reserve=[_u(rng, 32) for _ in range(4)], crc=_u(rng, 32))
    if which == "MotorCmds_":
        n = rng.randrange(0, 8)
        return types["MotorCmds_"](cmds=[types["GoMotorCmd_"](**_go_motor_cmd(rng))
                                         for _ in range(n)])
    if which == "MotorStates_":
        n = rng.randrange(0, 8)
        return types["MotorStates_"](
            states=[types["GoMotorState_"](**_go_motor_state(rng)) for _ in range(n)])
    if which == "Point32_":
        return types["Point32_"](x=_f32(rng), y=_f32(rng), z=_f32(rng))
    raise KeyError(which)


def _ours():
    return {"LowCmd_": unitree_hg.LowCmd_, "LowState_": unitree_hg.LowState_,
            "HGMotorCmd_": unitree_hg.MotorCmd_, "HGMotorState_": unitree_hg.MotorState_,
            "IMUState_": unitree_hg.IMUState_, "MotorCmds_": unitree_go.MotorCmds_,
            "MotorStates_": unitree_go.MotorStates_, "GoMotorCmd_": unitree_go.MotorCmd_,
            "GoMotorState_": unitree_go.MotorState_, "Point32_": geometry_msgs.Point32_}


def _theirs():
    from unitree_sdk2py.idl.geometry_msgs.msg import dds_ as sgm
    from unitree_sdk2py.idl.unitree_go.msg import dds_ as sgo
    from unitree_sdk2py.idl.unitree_hg.msg import dds_ as shg
    return {"LowCmd_": shg.LowCmd_, "LowState_": shg.LowState_,
            "HGMotorCmd_": shg.MotorCmd_, "HGMotorState_": shg.MotorState_,
            "IMUState_": shg.IMUState_, "MotorCmds_": sgo.MotorCmds_,
            "MotorStates_": sgo.MotorStates_, "GoMotorCmd_": sgo.MotorCmd_,
            "GoMotorState_": sgo.MotorState_, "Point32_": sgm.Point32_}


@pytest.mark.parametrize("which", ["LowCmd_", "LowState_", "MotorCmds_",
                                   "MotorStates_", "Point32_"])
def test_serialized_bytes_equal_the_sdks(which):
    ours, theirs = _ours(), _theirs()
    for seed in range(25):
        a = _build(which, ours, random.Random(seed))
        b = _build(which, theirs, random.Random(seed))
        wire = a.serialize()
        assert wire == b.serialize(), (which, seed)
        # And each side reads the other's bytes back to the same message
        # (compared re-serialized: a uint8 array reads back as `bytes`, not
        # the list it was built from — a Python representation, not a wire one).
        assert type(a).deserialize(b.serialize()).serialize() == wire
        assert type(b).deserialize(wire).serialize() == wire


def test_the_default_constructors_are_the_sdks():
    from unitree_sdk2py.idl import default as d
    pairs = [(unitree_hg.default_LowCmd_, d.unitree_hg_msg_dds__LowCmd_),
             (unitree_hg.default_LowState_, d.unitree_hg_msg_dds__LowState_),
             (unitree_go.default_MotorCmd_, d.unitree_go_msg_dds__MotorCmd_),
             (unitree_go.default_MotorState_, d.unitree_go_msg_dds__MotorState_),
             (geometry_msgs.default_Point32_, d.geometry_msgs_msg_dds__Point32_)]
    for ours, theirs in pairs:
        assert ours().serialize() == theirs().serialize(), ours.__name__


def test_the_lowcmd_crc_is_the_sdks(monkeypatch):
    """Against the SDK's own CRC: its native library where the install has
    one, else its pure-Python reference loop (`_crc_py`) — the .so is missing
    from some installs, which its constructor does not survive on Linux."""
    import os

    from unitree_sdk2py.utils import crc as sdk_crc
    from unitree_sdk2py.utils.crc import CRC
    lib = os.path.join(os.path.dirname(sdk_crc.__file__), "lib")
    if not (os.path.isdir(lib)
            and any(f.startswith("crc_") for f in os.listdir(lib))):
        monkeypatch.setattr(sdk_crc.platform, "system", lambda: "Darwin")
    ref = CRC()
    ours, theirs = _ours(), _theirs()
    for seed in range(50):
        a = _build("LowCmd_", ours, random.Random(seed))
        b = _build("LowCmd_", theirs, random.Random(seed))
        assert crc.hg_lowcmd_crc(a) == ref.Crc(b), seed
    zero = unitree_hg.default_LowCmd_()
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    assert crc.hg_lowcmd_crc(zero) == ref.Crc(unitree_hg_msg_dds__LowCmd_())


def test_the_domain_config_is_the_sdks():
    from unitree_sdk2py.core import channel_config as cc
    assert channel.ChannelConfigHasInterface == cc.ChannelConfigHasInterface
    assert channel.ChannelConfigAutoDetermine == cc.ChannelConfigAutoDetermine


def test_dds_init_binds_an_ambiguous_nic_by_address(monkeypatch):
    """`dds/domain.py::init` on a NIC holding several addresses switches THIS
    module's template to `address=` and binds the robot-subnet address."""
    from g1d_deploy.runtime.drivers.dds import domain as dds

    calls = []
    monkeypatch.setattr(channel, "ChannelFactoryInitialize",
                        lambda *a: calls.append(a))
    monkeypatch.setattr(channel, "ChannelConfigHasInterface",
                        channel.ChannelConfigHasInterface)
    monkeypatch.setattr(dds, "ipv4_addresses",
                        lambda iface: ["192.168.1.111", "192.168.123.118"])
    assert dds.init(0, "enp4s0") == "192.168.123.118"
    assert calls == [(0, "192.168.123.118")]
    assert 'address="$__IF_NAME__$"' in channel.ChannelConfigHasInterface

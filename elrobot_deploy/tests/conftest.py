"""Fixtures every section's tests share.

The arena is small on purpose: 8x8 "cameras" and a horizon of 8 keep a full
create/attach cycle under a millisecond, so a test can afford one per test
rather than sharing mutable state between them.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pytest

from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.config_loading.derive import derive
from g1d_deploy.runtime.shared_memory import Arena, build, open_all, schema


#: The machine file every test runs against. It states a CPU table: a laptop
#: cannot derive one (no /sys topology), and a test must not depend on the
#: checkout's own (uncommitted) machine.toml.
MACHINE_TOML = Path(__file__).resolve().parent / "machine.toml"


@pytest.fixture(autouse=True)
def _machine(monkeypatch):
    """The machine file resolves the DDS interface by auto-detecting the NIC on
    the robot's subnet, and refuses when there is none — which a test box has
    no reason to have. Tests never open a socket, so they declare a
    placeholder rather than the refusal being weakened to accommodate them."""
    monkeypatch.setenv("G1D_MACHINE", str(MACHINE_TOML))
    monkeypatch.setenv("G1D_IFACE", "lo0")
    for var in ("G1D_ROBOT_IP", "G1D_BOARD_SSH_USER", "G1D_BOARD_SSH_PASSWORD"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def hw():
    return cfgmod.load_hardware()


@pytest.fixture
def cfg():
    return cfgmod.DeployConfig(
        prompt="test",
        checkpoint="",              # the EchoPolicy dry run: no checkpoint to find
        sync=True,
        replan_period_ticks=3,
        action_horizon=8,
        control_hz=30.0,
        cameras={"rwrist": cfgmod.CameraSpec(8, 8),
                 "head": cfgmod.CameraSpec(8, 16)},
        primary_camera="rwrist",
        home_q=np.zeros(schema.ARM_DOF),
        home_waist=np.zeros(schema.WAIST_DOF),
        home_hand=np.zeros((schema.N_HANDS, schema.HAND_DIM)),
        exit_q=np.zeros(schema.ARM_DOF),
        exit_waist=np.zeros(schema.WAIST_DOF),
        exit_hand=np.zeros((schema.N_HANDS, schema.HAND_DIM)),
        state_history_s=0.1,
    )


@pytest.fixture
def derived(cfg, hw):
    return derive(cfg, hw)


@pytest.fixture
def layout(cfg, derived):
    return build(cfg, derived)


@pytest.fixture
def arena(layout):
    name = f"g1d-t{os.getpid()}-{uuid.uuid4().hex[:8]}"
    a = Arena.create(name, layout)
    try:
        yield a
    finally:
        a.close()
        a.unlink()


@pytest.fixture
def rings(arena, cfg):
    return open_all(arena, cfg)

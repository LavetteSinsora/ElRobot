"""P12, the Astra source, without P6: when it refuses, when it gives up, how it lets go."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from elrobot_deploy.runtime.processes import p12_astra
from elrobot_deploy.runtime.processes.p12_astra import AstraSource
from elrobot_deploy.runtime.shared_memory import schema

WP = {"waypoints": [{"xyz": [0.02, 0, 0], "rpy": [0, 0, 0], "gripper_mm": 30}]}


def fk(q7):
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.25, 0.1]
    return T


@pytest.fixture
def source(rings, cfg):
    rings["control"].session = schema.ACTIVE
    rings["control"].bus = schema.BUS_RUNNING
    q = np.r_[np.zeros(7), cfg.limits.lower[7]]
    rings["state"].write(t=time.monotonic(), q=q, q_cmd=q)
    return AstraSource(rings, cfg, fk=fk, log=lambda s: None)


def test_it_refuses_outside_a_rollout_and_on_a_stopped_bus(source, rings):
    assert source.why_not() is None
    rings["control"].session = schema.READY
    assert "not in a rollout" in source.why_not()
    rings["control"].session = schema.ACTIVE
    rings["control"].bus = schema.BUS_HOLDING
    assert "HOLDING" in source.why_not()


def test_the_first_reference_is_the_held_command_and_the_plan_starts_there(source):
    plan, _ = source.plan(WP)
    assert plan.p[0] == pytest.approx([0.0, 0.25, 0.1])
    assert plan.ref.gripper_mm == pytest.approx(55.0)            # the open end
    assert plan.p[-1] == pytest.approx([0.02, 0.25, 0.1])        # TCP x = base x here (fk)
    with pytest.raises(ValueError, match="gripper_mm"):
        source.plan({"waypoints": [{"xyz": [0, 0, 0], "rpy": [0, 0, 0], "gripper_mm": 70}]})


def test_without_p6_the_probe_gives_up_and_hands_back(source, rings, monkeypatch):
    monkeypatch.setattr(p12_astra, "P6_WAIT_S", 0.2)
    plan, held = source.plan(WP)
    out = source.execute(plan, held)
    assert out["t0"] is None and "P6 took no target" in out["interrupted"]
    assert rings["control"].intervene % 2 == 0                   # released: the arm holds
    assert rings["intervene"].meta()["newest"] > 0               # it did offer rest rows
    assert source.ref is None                                    # re-anchors next time


def test_leaving_active_mid_action_releases_at_once(source, rings, monkeypatch):
    plan, held = source.plan(WP)
    out = {}
    th = threading.Thread(target=lambda: out.update(source.execute(plan, held)))
    th.start()
    time.sleep(0.15)
    assert rings["control"].intervene % 2 == 1                   # driving (probing)
    rings["control"].session = schema.HOME                       # Reset pressed
    th.join(2.0)
    assert not th.is_alive()
    assert "not in a rollout" in out["interrupted"]
    assert rings["control"].intervene % 2 == 0

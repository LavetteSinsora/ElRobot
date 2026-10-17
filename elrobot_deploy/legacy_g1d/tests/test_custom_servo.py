"""`[control.servo] smoother` / `feedforward = "pkg.module:factory"`: a deploy
file replaces the 500 Hz shaping's two pieces with its own, under the same
switches (`filter_hz = 0` / `torque_ff = 0` still turn them off), imported at
launch in elrobot_deploy's Python because P7 runs there."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from elrobot_deploy.config_loading import deploy_file
from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.config_loading.validate import validate
from elrobot_deploy.runtime.execution import servo_filter
from elrobot_deploy.runtime.shared_memory import schema

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
HERE = __name__                                     # this module, importable


class Hold:
    """A smoother that ignores the line: the plainest thing with the protocol."""

    def __init__(self, servo, hw, q0):
        self.q = np.asarray(q0, float).copy()
        self.lead_s = 0.05

    def step(self, reference, h):
        z = np.zeros_like(self.q)
        return self.q, z, z


class Constant:
    def __init__(self, servo, hw):
        self.value = servo.torque_ff

    def tau(self, v, a):
        return np.full(schema.BODY_DOF, self.value)


def test_named_modules_replace_the_shipped_pieces(hw):
    q0 = np.ones(schema.BODY_DOF)
    got = servo_filter.build(cfgmod.ServoShaping(), hw, q0,
                             smoother=f"{HERE}:Hold", feedforward=f"{HERE}:Constant")
    assert isinstance(got.smoother, Hold) and got.lead_s == 0.05
    assert isinstance(got.feedforward, Constant)
    assert np.all(got.feedforward.tau(q0, q0) == 0.5)     # torque_ff is its input


def test_the_switches_still_turn_them_off(hw):
    got = servo_filter.build(cfgmod.ServoShaping.linear(), hw,
                             np.zeros(schema.BODY_DOF),
                             smoother=f"{HERE}:Hold", feedforward=f"{HERE}:Constant")
    assert got.smoother is None and got.feedforward is None


def test_the_lead_is_used_only_with_filter_lead(hw):
    got = servo_filter.build(cfgmod.ServoShaping(filter_lead=False), hw,
                             np.zeros(schema.BODY_DOF), smoother=f"{HERE}:Hold")
    assert got.lead_s == 0.0


def _deploy(tmp_path, servo_lines: str) -> Path:
    from . import example_files
    p = example_files.task(tmp_path / "t")
    text = p.read_text()
    assert text.count("[control.servo]\n") == 1
    text = text.replace("filter_hz = 4.0", "")        # set below instead
    p.write_text(text.replace("[control.servo]\n", "[control.servo]\n" + servo_lines))
    return p


def test_the_deploy_file_names_them_and_they_survive_to_the_config(tmp_path):
    r = deploy_file.resolve(_deploy(tmp_path, f'smoother = "{HERE}:Hold"\n'
                                    f'feedforward = "{HERE}:Constant"\n'
                                    "filter_hz = 6.0\n"), record=False)
    assert r.cfg.servo_smoother == f"{HERE}:Hold"
    assert r.cfg.servo_feedforward == f"{HERE}:Constant"
    assert r.cfg.servo.filter_hz == 6.0


def test_one_that_does_not_import_is_refused_at_launch(cfg, hw):
    with pytest.raises(ConfigError, match=r"\[control.servo\] smoother"):
        validate(cfgmod.replace(cfg, servo_smoother="no_such_g1d_mod:Thing"), hw)


def test_a_custom_feedforward_needs_no_servo_model(cfg, hw):
    import dataclasses
    bare = dataclasses.replace(hw, servo_model=None)
    with pytest.raises(ConfigError, match="servo_model"):
        validate(cfg, bare)
    validate(cfgmod.replace(cfg, servo_feedforward=f"{HERE}:Constant"), bare)

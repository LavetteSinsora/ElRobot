"""Regressions from the 2026-09-30 skill runs (four agents writing adapters
for real checkpoints):

* `describe` reported the child's `sys.executable`, so an interpreter that is
  a WRAPPER (sets PYTHONPATH, execs a venv python) was replaced by the bare
  python for the class's `resolve_checkpoint`, which then could not import
  the adapter;
* `check-policy --no-run` never ran the class's `__init__`, so a failing
  constructor (a stamp check, a missing environment variable) surfaced only at
  level 2, after taking the GPU.
"""

from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from g1d_deploy.cli import main as cli
from g1d_deploy.policy_contract.describe import describe

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_describe_keeps_a_wrapper_interpreter(tmp_path):
    wrapper = tmp_path / "py.sh"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)
    info = describe(f"{EXAMPLES / 'policies' / 'hold.py'}:Policy", python=str(wrapper))
    assert info.python == str(wrapper)


def test_level_one_constructs_the_class_and_catches_its_init(tmp_path, capfd):
    from . import example_files
    example_files.task(tmp_path, policy=None)
    (tmp_path / "policy.py").write_text(textwrap.dedent("""
        from g1d_deploy import policy_contract as g1d

        class Policy(g1d.Policy):
            action_mode, end_effector, anchor_camera = "eef", "dex1", "head_left"

            def __init__(self, checkpoint, prompt, robot):
                super().__init__(checkpoint, prompt, robot)
                raise RuntimeError("stamp mismatch")

            def infer(self, obs, reader):
                raise AssertionError
    """))
    with pytest.raises(SystemExit) as got:
        cli.main(["check-policy", str(tmp_path / "deploy.toml"), "--no-run"])
    out = capfd.readouterr()
    assert got.value.code == 1
    assert "construct the policy (GPU hidden)" in out.out
    assert "stamp mismatch" in out.out + out.err
    assert "--- 2." not in out.out


def test_an_adapter_builds_offline_as_p5_builds_it():
    from g1d_deploy.policy_contract.offline import build
    policy, cfg, hw = build(EXAMPLES / "policies" / "history.toml")
    assert policy.cameras["rwrist"].crop_source == "fallback_lens_circle"
    assert policy.settings.replan_period == cfg.P
    images, q = policy.model_inputs(policy.dummy_observation(), policy.reader())
    assert images.shape[0] == 3 and q.shape == (50, 14)
    policy, _, _ = build(EXAMPLES / "policies" / "sine_eef.toml", warmup=True)
    assert policy.infer(policy.dummy_observation(), policy.reader()).n == 50

"""`g1d_deploy check-policy`: resolves like `run`, then loads the checkpoint
and runs one inference on the dummy observation under the policy's own
interpreter. Only what does not depend on the input fails level 2 (the action
type and count); the values, meaningless on a dummy, are notes. No robot."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from g1d_deploy.cli import main as cli

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _check(argv, capfd):
    with pytest.raises(SystemExit) as got:
        cli.main(["check-policy", *argv])
    out = capfd.readouterr()
    return got.value.code, out.out + out.err


@pytest.mark.parametrize("example", ["sine_eef", "sine_joint"])
def test_the_shipped_examples_pass_both_levels(example, capfd):
    code, out = _check([str(EXAMPLES / "policies" / f"{example}.toml")], capfd)
    assert code == 0, out
    assert "--- 2. the policy loads" in out
    assert "[  ok  ] the action type and count" in out and "[ FAIL ]" not in out


def _adapter(tmp_path, body: str) -> Path:
    from . import example_files
    example_files.task(tmp_path, policy=None)
    (tmp_path / "policy.py").write_text(textwrap.dedent(body))
    return tmp_path / "deploy.toml"


_BAD_HAND = """
    import numpy as np
    from g1d_deploy import policy_contract as g1d
    from g1d_deploy.policy_contract import EefActions

    class Policy(g1d.Policy):
        action_mode = "eef"
        end_effector = "dex1"
        anchor_camera = "head_left"

        def infer(self, obs, reader):
            n = 50
            pose = np.zeros((n, 2, 7)); pose[..., 3] = 1.0
            # a gripper WIDTH in metres, not Dex-1 gear radians: -0.01 is
            # below the travel, which P5 would refuse live
            return EefActions(pose=pose, hand=np.full((n, 2), -0.01))
"""


def test_values_on_the_dummy_are_a_note_not_a_failure(tmp_path, capfd):
    """Black frames make any model's values meaningless, so P5's value checks
    are reported, and the check still passes."""
    code, out = _check([str(_adapter(tmp_path, _BAD_HAND))], capfd)
    assert code == 0, out
    assert "[ note ] on the dummy input, P5 would refuse these values" in out
    assert "travel" in out


_WRONG_TYPE = """
    import numpy as np
    from g1d_deploy import policy_contract as g1d
    from g1d_deploy.policy_contract import JointActions

    class Policy(g1d.Policy):
        action_mode = "eef"
        end_effector = "dex1"
        anchor_camera = "head_left"

        def infer(self, obs, reader):
            n = 50
            return JointActions(arm_q=np.zeros((n, 2, 7)), waist_q=np.zeros((n, 3)),
                                hand=np.zeros((n, 2)))
"""


def test_the_wrong_action_type_fails_level_two(tmp_path, capfd):
    code, out = _check([str(_adapter(tmp_path, _WRONG_TYPE))], capfd)
    assert code == 1
    assert "[ FAIL ] the action type and count" in out


def test_level_two_is_refused_while_a_deployment_holds_the_gpu(capfd, monkeypatch):
    from g1d_deploy.dashboards.web import gpu
    monkeypatch.setattr(gpu, "refusal", lambda: "a live deployment is running")
    code, out = _check([str(EXAMPLES / "policies" / "sine_eef.toml")], capfd)
    assert code == 1
    assert "level 2 not run: a live deployment is running" in out


_NOT_A_POLICY = """
    class Policy:
        pass
"""


def test_a_class_that_is_not_a_policy_fails_level_one(tmp_path, capfd):
    code, out = _check([str(_adapter(tmp_path, _NOT_A_POLICY))], capfd)
    assert code == 1
    assert "[ FAIL ]" in out and "--- 2." not in out


def test_no_run_stops_after_resolving(capfd):
    code, out = _check([str(EXAMPLES / "policies" / "sine_eef.toml"), "--no-run"], capfd)
    assert code == 0
    assert "--- 2." not in out

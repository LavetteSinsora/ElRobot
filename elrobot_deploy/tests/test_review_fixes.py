"""Regressions for the 2026-09-29 review of the docs against the code.

* a chunk longer than `action_horizon` is cut to H, not refused: an adapter
  "returns the whole chunk" at its trained horizon and cannot know H;
* the deploy file's `[policy] anchor_camera` / `uses_cameras` override the
  class's in P5 (they used to reach only the dashboard and the record);
* `uses_cameras` is exactly the set `obs.frames` carries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from g1d_deploy.cli import main as cli
from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.policy_contract import EefActions, JointActions, Policy
from g1d_deploy.policy_contract.actions import first_actions
from g1d_deploy.policy_contract.base import anchor_camera, observation_cameras

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_first_actions_cuts_either_type_and_leaves_short_chunks_alone():
    e = EefActions(pose=np.tile([0, 0, 0, 1, 0, 0, 0.0], (10, 2, 1)),
                   hand=np.zeros((10, 2)))
    assert first_actions(e, 4).n == 4 and first_actions(e, 40) is e
    j = JointActions(arm_q=np.zeros((10, 2, 7)), waist_q=np.zeros((10, 3)),
                     hand=np.zeros((10, 2)))
    assert first_actions(j, 3).waist_q.shape == (3, 3)


def test_a_horizon_below_the_trained_one_still_passes_check_policy(capfd):
    with pytest.raises(SystemExit) as got:
        cli.main(["check-policy", str(EXAMPLES / "policies" / "sine_eef.toml"),
                  "--set", "control.action_horizon=20",
                  "--set", "control.replan_period=10"])
    out = capfd.readouterr().out
    assert got.value.code == 0, out
    assert "n=20" in out


class _P(Policy):
    action_mode, end_effector = "eef", "dex1"
    anchor_camera = "head_left"
    uses_cameras = ("head_left", "rwrist")

    def infer(self, obs, reader):
        raise AssertionError


def test_the_deploy_files_camera_overrides_reach_p5(cfg):
    cams = {"head_left": cfgmod.CameraSpec(8, 8), "rwrist": cfgmod.CameraSpec(8, 8),
            "lwrist": cfgmod.CameraSpec(8, 8)}
    by_hand = cfgmod.replace(cfg, cameras=cams, primary_camera="rwrist")
    assert observation_cameras(_P, by_hand) == ("head_left", "rwrist")
    over = cfgmod.replace(by_hand, observed_cameras=("lwrist", "rwrist"))
    assert observation_cameras(_P, over) == ("lwrist", "rwrist")
    assert anchor_camera(_P, over) == "rwrist"
    stray = cfgmod.replace(by_hand, primary_camera="nope")
    assert anchor_camera(_P, stray) == "head_left"

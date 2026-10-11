"""TEMPLATE adapter: copy this folder (policy.py, conversions.py, __init__.py,
deploy.toml) next to the model code and fill in every TODO.

Keep `__init__.py`: `from .conversions import ...` works because g1d_deploy
imports a file that sits in a package as a package member. Give the folder a
name that is a valid identifier and does not shadow an installed module.

As shipped it runs (and passes `g1d_deploy check-policy`): with no model it
holds the arm where it is. The contract: docs/deploying_a_policy/policy_contract.md.
"""

from __future__ import annotations

import numpy as np

from g1d_deploy import policy_contract as g1d
from g1d_deploy.policy_contract import EefActions

from .conversions import normalize_quat  # noqa: F401  (tcp_to_flange, xyzw_to_wxyz, ...)


class Policy(g1d.Policy):
    # --- declarations (docs/deploying_a_policy/policy_contract.md, "Declarations") ----------------
    action_mode = "eef"            # TODO "eef" (flange poses) | "joint" (joint angles)
    end_effector = "dex1"          # the hand the policy's gripper values are for
    anchor_camera = "head_left"    # TODO the camera the model relies on most for fine motion
    # uses_cameras = ("head_left", "rwrist")         # exactly what obs.frames carries
    # history = {"state": 1.0, "head_left": 2.0}     # only if the model takes past inputs
    trained_control_hz = 30.0      # TODO the spacing of actions in the TRAINING DATA, Hz
    trained_action_horizon = 50    # TODO actions per chunk the model outputs
    # max_prompt_tokens = 48       # TODO the length prompts are padded/compiled at (not a training limit)

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)
        # TODO cheap things only (no GPU): model config, tokenizer, norm stats,
        # e.g. self.stats = load_norm_stats(checkpoint / "stats.json").
        self.model = None

    # def count_prompt_tokens(self, text):
    #     return len(self.tokenizer(TEMPLATE.format(text))["input_ids"])

    def warmup(self) -> None:
        # TODO restore the weights here (the expensive part), e.g.
        # self.model = load_model(self.checkpoint)
        super().warmup()               # one dummy inference: compiles
        self._hold = None              # forget what the dummy latched

    # def on_settings(self, settings):
    #     self.sampler = compile_sampler(settings.sample_steps)   # per-rollout knob

    def infer(self, obs, reader) -> EefActions:
        H = self.trained_action_horizon        # return the whole chunk; the runtime cuts it
        if self.model is None:
            # No model yet: hold the measured pose (replace this whole branch).
            if getattr(self, "_hold", None) is None:
                self._hold = (obs.state.flange.copy(), obs.state.hand.copy())
            flange, hand = self._hold
            return EefActions(pose=np.repeat(flange[None], H, axis=0),
                              hand=np.clip(np.repeat(hand[None], H, axis=0),
                                           self.robot.hand_min, self.robot.hand_max))

        # 1. Inputs exactly as training saw them. Frames are RGB at native size;
        #    a fisheye is already cropped square around its lens circle. Resize,
        #    normalise as the training pipeline did.
        image = obs.frames[self.anchor_camera].pixels            # uint8 (h, w, 3)
        state = obs.state                                          # q, flange, hand
        # 2. The model.
        out = self.model(image, state, obs.prompt)                 # TODO
        # 3. Into the contract: un-normalise; delta -> absolute (apply_delta);
        #    TCP -> flange (tcp_to_flange); rotation -> wxyz (conversions.py);
        #    [left, right]; gripper -> Dex-1 radians, 0 = closed, clipped.
        pose = out["pose"]                                          # TODO (n, 2, 7)
        pose[..., 3:] = normalize_quat(pose[..., 3:])
        hand = np.clip(out["gripper"], self.robot.hand_min, self.robot.hand_max)
        return EefActions(pose=pose, hand=hand)

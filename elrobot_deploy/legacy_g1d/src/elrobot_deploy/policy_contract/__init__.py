"""The policy surface: what an adapter subclasses, receives and returns.

Imported by the adapter in ITS OWN interpreter (P5) and by the infra's; it
therefore needs numpy only. A policy is anything that returns actions in the
contract — a network, a scripted controller, a bridge to an external planner.

    from elrobot_deploy.policy_contract import Policy, EefActions, JointActions

    class MyPolicy(Policy):
        action_mode = "eef"
        end_effector = "dex1"
        anchor_camera = "rwrist"

        def __init__(self, checkpoint, prompt, robot):
            super().__init__(checkpoint, prompt, robot)
            ...                                     # whatever is cheap to load

        def warmup(self):
            ...                                     # the slow load, if any
            super().warmup()                        # one infer on a dummy obs

        def infer(self, obs, reader):
            ...                                     # obs.frames[cam].pixels is RGB
            return EefActions(pose=..., hand=...)

The whole contract, with the joint-order table, is `docs/deploying_a_policy/policy_contract.md`.

Re-exported here so `from elrobot_deploy.policy_contract import ...` is the one
import a policy needs:

* `base`           — the `Policy` class, `CameraInfo`, loading and checking one
* `actions`        — `EefActions`, `JointActions`, the ring's `Chunk`, validation
* `observation`    — `Observation`, `Reader` (history), `nearest_to_times` /
                     `subsample_every`, the RGB conversion
* `deadline_shift` — the re-timing the infra applies to every policy
* `anchor`         — the anchor rule

Not re-exported: `loading` (finding the class a deploy file names), `describe`
(asking the adapter's interpreter about its class), `checkpoints` (the default
checkpoint layout), `check` and `offline` (`elrobot_deploy check-policy`).
"""

from elrobot_deploy.policy_contract.anchor import Anchor, acquire_anchor
from elrobot_deploy.policy_contract.base import (CameraInfo, EchoPolicy, Policy,
                                             PolicyError, PromptTooLong, RobotSpec,
                                             bind_runtime, camera_info,
                                             deliver_settings,
                                             check_against, check_end_effector,
                                             make_policy, observation_cameras,
                                             validate_policy_class)
from elrobot_deploy.policy_contract.actions import (ACTION_MODES, EEF, JOINT, Actions,
                                                Chunk, ChunkError, EefActions,
                                                JointActions, Limits, validate_actions,
                                                validate_chunk)
from elrobot_deploy.policy_contract.observation import (DummyReader, Frame, FrameTimes,
                                                    Observation, Reader, State,
                                                    StateWindow, dummy_observation,
                                                    subsample_every, nearest_to_times,
                                                    observe)
from elrobot_deploy.policy_contract.deadline_shift import (deadline_shift_slots,
                                                       shift_to_deadline)

__all__ = ["ACTION_MODES", "Actions", "Anchor", "CameraInfo", "Chunk",
           "ChunkError", "EEF", "camera_info", "deliver_settings",
           "DummyReader", "EchoPolicy", "EefActions", "Frame", "FrameTimes",
           "JOINT", "JointActions", "Limits", "Observation", "Policy",
           "PolicyError", "PromptTooLong", "Reader", "RobotSpec", "State",
           "StateWindow", "acquire_anchor", "bind_runtime", "check_against", "check_end_effector",
           "deadline_shift_slots", "dummy_observation", "subsample_every",
           "make_policy", "nearest_to_times",
           "observation_cameras", "observe", "shift_to_deadline",
           "validate_actions", "validate_chunk", "validate_policy_class"]

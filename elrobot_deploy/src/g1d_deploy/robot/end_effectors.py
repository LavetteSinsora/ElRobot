"""The end effectors this runtime can drive — built in, not in the robot file.

A robot file says only WHICH end effector is bolted on, and optionally where
its tool centre point is:

    [end_effector]
    type = "dex1"                                   # dex1 | brainco
    tcp = [0.184759, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # optional, flange frame

Everything else about a type — its DDS topics, its travel, which end is
closed, and whether the policy contract speaks it — is a fact about the
product, the same on every robot that carries it, so it lives here.
`config_loading/resolved.py::load_hardware` copies the chosen type's entry into `Hardware`
(`topic_hand_cmd`, `topic_hand_state`, `hand_min`, `hand_max`, `hand`), once,
so P1, P8, P11 and CHECK cannot disagree about which hand they talk to.

**The policy contract's `hand` is the end effector's own quantity**
(`docs/deploying_a_policy/policy_contract.md`, "The actions"): for a Dex-1,
one jaw value per hand in gear radians, 0 = closed. Only
types with `contract = True` can run a policy; the BrainCo is reserved: its
wire is implemented (one grip value per hand in [0, 1], 0 = OPEN, spread over
six motors by `runtime/drivers/gripper.py::brainco_targets`) but what a policy would command for
six fingers has not been decided, so a policy is refused on it at load rather
than handed units nobody chose.

Adding an end effector is a code change here plus its wire in
`runtime/drivers/gripper.py` and P8 (`docs/setup/robot_toml.md`,
"`[end_effector]`"); numpy-free, imported by P5.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndEffector:
    """One type's built-in facts."""

    type: str
    #: One topic per hand, `[left, right]`.
    topic_hand_cmd: tuple[str, str]
    topic_hand_state: tuple[str, str]
    #: Travel of the one value per hand the wire carries, in the type's units.
    hand_min: float
    hand_max: float
    #: True when a LARGER value is more closed (the BrainCo: 0 = open).
    close_is_max: bool
    #: Whether the policy action contract supports this type (`Policy.
    #: end_effector` may name it). False = the robot runs, a policy is refused.
    contract: bool
    #: What `hand` means to a policy, for docs and refusals.
    units: str


END_EFFECTORS: dict[str, EndEffector] = {
    # The Dex-1 gripper. ROBOT: `dex1_1_gripper_server -l -r` gives every
    # gripper its own topic carrying its own single motor (both state topics
    # measured at ~492 Hz, one motor each, on the reference robot, 2026-08-20;
    # the vendor class default `rt/unitree_actuator/{cmd,state}` is silent
    # there).
    #
    # Travel MEASURED on the reference robot: the server reports ~0 at the closed hard
    # stop and ~5.4 at the open one, and rollout records agree (measured
    # `grip_meas` 1.19 … 5.4248). These bound the COMMAND at the last point
    # before the wire; the driver itself clamps only the step, not the target.
    "dex1": EndEffector(
        type="dex1",
        topic_hand_cmd=("rt/dex1/left/cmd", "rt/dex1/right/cmd"),
        topic_hand_state=("rt/dex1/left/state", "rt/dex1/right/state"),
        hand_min=0.0, hand_max=5.4, close_is_max=False, contract=True,
        units="Dex-1 jaw, gear radians, 0 = fully closed … 5.4 = fully open"),
    # The BrainCo Revo2 hand. VENDOR, Unitree xr_teleoperate
    # `teleop/robot_control/robot_hand_brainco.py`: the same `unitree_go`
    # MotorCmds_/MotorStates_ as the Dex-1 and one topic per hand, but SIX
    # motors each, commanded as a normalized position in [0, 1] — 0 fully
    # OPEN. The runtime carries one grip value per hand on the wire; its
    # board-side driver is out of scope. Not in the policy contract (above).
    "brainco": EndEffector(
        type="brainco",
        topic_hand_cmd=("rt/brainco/left/cmd", "rt/brainco/right/cmd"),
        topic_hand_state=("rt/brainco/left/state", "rt/brainco/right/state"),
        hand_min=0.0, hand_max=1.0, close_is_max=True, contract=False,
        units="reserved: not supported by the action contract"),
}

#: The types a robot file's `[end_effector] type` may name.
TYPES = tuple(END_EFFECTORS)

#: The Dex-1's travel, rad. The intervention's grip knobs are written in these
#: units, and another hand gets the same fraction of ITS travel.
DEX1_TRAVEL = END_EFFECTORS["dex1"].hand_max - END_EFFECTORS["dex1"].hand_min


def get(ee_type: str) -> EndEffector:
    """The built-in entry for `ee_type`; KeyError names the known ones."""
    try:
        return END_EFFECTORS[ee_type]
    except KeyError:
        raise KeyError(f"end effector {ee_type!r} is not one of {list(TYPES)}") from None


__all__ = ["DEX1_TRAVEL", "END_EFFECTORS", "EndEffector", "TYPES", "get"]

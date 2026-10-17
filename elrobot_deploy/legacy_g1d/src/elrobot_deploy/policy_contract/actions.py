"""The action contract, and the chunk it becomes on the ring.

Used by P5 (validation, publishing), P6 and P9 (the `Chunk` they read back).
Design: `docs/deploying_a_policy/policy_contract.md`, "The actions" and "What
happens to the actions".

Two layers, and the line between them is the whole point of this module:

* **What a policy returns**: `EefActions` or `JointActions`. Absolute, in the
  robot's own units, timed as if action `j` were due `(j+1)·dt` after
  `obs.t_anchor` — for a learned policy, how training labelled it. Nothing about the schedule,
  the ring, the waist or the IK appears here, because none of it is the
  policy's.

* **What the infra publishes**: `Chunk`. The same actions after P5 has checked
  them, re-timed them onto the deadlines they will actually meet
  (`deadline_shift.shift_to_deadline`), truncated them to `P` in sync, and packed the
  hand into the `(2, 6)` hand block P8 reads. `Chunk` carries its `mode` so
  that P6 — a different process, which never sees the policy class — can tell
  which half of its loop a tick takes, and so that the record can say it too.

**Why two validations.** `validate_actions` checks the POLICY's output against
the contract (shapes, finiteness, hand travel, joint limits, horizon) and
every error it raises names something the policy did. `validate_chunk` checks
what is about to reach the ring (finiteness again, horizon, `n == P` in sync),
and the only thing it can catch that the first could not is the infra's own
arithmetic going wrong. Joint limits and hand travel are enforced on the
policy's output and then CLIPPED after the shift, never re-checked: the shift
interpolates against the measured state, which may itself sit a hair outside a
URDF bound (the bow rests against a stop at -0.0436 rad), and a run must not be
refused for a number the policy never produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Union

import numpy as np

from elrobot_deploy.robot import end_effectors
from elrobot_deploy.runtime.shared_memory import schema

#: The two values `Policy.action_mode` may take. Everything that branches on the
#: mode compares against these, never against a literal.
EEF, JOINT = "eef", "joint"
ACTION_MODES = (EEF, JOINT)

#: The ring's encoding of the same thing (`schema.chunk_dtype`'s `mode` word).
MODE_CODE = {EEF: 0, JOINT: 1}
MODE_NAME = {v: k for k, v in MODE_CODE.items()}

#: The Dex-1's travel in gear radians, 0 = fully CLOSED (measured on the
#: reference robot; `end_effectors.END_EFFECTORS["dex1"]`). What `hand` is bounded by
#: when no robot is supplied (tests driving P5 without one).
DEX1_HAND_RANGE = (end_effectors.END_EFFECTORS["dex1"].hand_min,
                   end_effectors.END_EFFECTORS["dex1"].hand_max)

#: Absolute tolerance on the hand range and on the joint limits, radians.
#: Float slop, not a margin: a policy that clips to `[0, 5.4]` in float32 and
#: hands back float64 must not be refused for the rounding.
LIMIT_EPS = 1e-6

N = schema.N_HANDS


class ChunkError(ValueError):
    """Actions that do not satisfy the contract, or a sync chunk that is not
    exactly `P` actions long."""


# --- what a policy returns ---------------------------------------------------


def _block(name: str, value, shape: tuple) -> np.ndarray:
    """`value` as a float64 array of exactly `shape`, or a ChunkError that says
    which field and which shape — a `(n, 7, 2)` pose that reached the ring would
    be a silently transposed arm, not an exception."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ChunkError(f"{name} has shape {arr.shape}, the contract wants {shape}")
    return arr


def _n(name: str, value) -> int:
    shape = np.shape(value)
    if not shape:
        raise ChunkError(f"{name} is a scalar; the contract wants one row per action")
    return int(shape[0])


@dataclass(frozen=True)
class EefActions:
    """`n` end-effector actions: arm + hand.

    `pose`    (n, 2, 7)  `[left, right]`; xyz metres + **wxyz** quaternion of the
                         **flange** (`wrist_yaw_link`); absolute, in the robot
                         base frame at `t_anchor`.
    `hand`    (n, 2)     the end effector's own quantity, per the policy's
                         `end_effector` (which must match the robot file's).
                         Dex-1: jaw gear radians, 0 … 5.4, **0 = fully closed**.
                         BrainCo: not supported by the contract yet.

    There is deliberately no waist. The IK owns it: pure arm IK
    commands zero and solves against the MEASURED trunk; a trunk planner
    decides for itself.
    """

    pose: np.ndarray
    hand: np.ndarray

    def __post_init__(self) -> None:
        n = _n("EefActions.pose", self.pose)
        object.__setattr__(self, "pose", _block(
            "EefActions.pose", self.pose, (n, N, schema.POSE_DIM)))
        object.__setattr__(self, "hand", _block(
            "EefActions.hand", self.hand, (n, N)))

    @property
    def n(self) -> int:
        return int(self.pose.shape[0])


@dataclass(frozen=True)
class JointActions:
    """`n` joint-space actions.

    `arm_q`   (n, 2, 7)  radians; per arm shoulder pitch/roll/yaw, elbow, wrist
                         roll/pitch/yaw — the order of `docs/deploying_a_policy/policy_contract.md`'s table.
    `waist_q` (n, 3)     radians in `kinematics.WAIST_JOINT_NAMES` slot order (yaw, bow,
                         absent). REQUIRED: a joint policy commands the whole
                         upper body, and the infra assumes nothing about the
                         waist — a policy that wants it upright returns zeros.
    `hand`    (n, 2)     as `EefActions`.

    Build it by keyword: `JointActions(arm_q=..., waist_q=..., hand=...)`.
    """

    arm_q: np.ndarray
    waist_q: np.ndarray
    hand: np.ndarray

    def __post_init__(self) -> None:
        n = _n("JointActions.arm_q", self.arm_q)
        object.__setattr__(self, "arm_q", _block(
            "JointActions.arm_q", self.arm_q, (n, N, schema.ARM_DOF // N)))
        if self.waist_q is None:
            raise ChunkError(
                "JointActions.waist_q is None; it is REQUIRED in joint mode "
                "(radians, (n, 3), slot order yaw/bow/absent). Return "
                "np.zeros((n, 3)) for an upright waist.")
        object.__setattr__(self, "waist_q", _block(
            "JointActions.waist_q", self.waist_q, (n, schema.WAIST_DOF)))
        object.__setattr__(self, "hand", _block(
            "JointActions.hand", self.hand, (n, N)))

    @property
    def n(self) -> int:
        return int(self.arm_q.shape[0])


Actions = Union[EefActions, JointActions]


class Limits(NamedTuple):
    """Everything `validate_actions` bounds against, read once from the robot.

    `arm` `(14, 2)` and `waist` `(3, 2)` are `[lower, upper]` from the URDF
    (`kinematics.arm_limits`, `kinematics.waist_limits`); `hand` is the mounted end
    effector's travel in its own units (`Hardware.hand_min/max`, built in per
    type). None for `arm`/`waist` means "not checked", and for `hand` the
    Dex-1's range — tests driving P5 without a robot.
    """

    arm: np.ndarray | None = None
    waist: np.ndarray | None = None
    hand: tuple[float, float] | None = None

    @classmethod
    def from_robot(cls, robot) -> "Limits":
        from elrobot_deploy.robot.kinematics import arm_limits, waist_limits
        return cls(arm=arm_limits(robot.urdf), waist=waist_limits(robot.urdf),
                   hand=(float(robot.hand_min), float(robot.hand_max)))


def _finite(name: str, block) -> None:
    """Non-finite first, everywhere, because it is the one that reaches the
    motors. A NaN survives BOTH velocity clamps — `np.clip(nan, lo, hi)` is
    `nan` — so P6's clamp against the seed and P7's last-defence clamp pass it
    through untouched, and it is published on the very first command. It is
    the likeliest way a policy fails, and this is the only place between it and `rt/lowcmd`. A diverged
    sampler, a bad normalization constant or a malformed waypoint from an
    external planner all end here."""
    if block is None:
        return
    if not np.isfinite(block).all():
        bad = int((~np.isfinite(block)).sum())
        raise ChunkError(
            f"{name} has {bad} non-finite value(s). A NaN here is published to "
            f"the arm: neither velocity clamp stops it.")


def _within(name: str, block, lo, hi, what: str) -> None:
    out = (block < np.asarray(lo) - LIMIT_EPS) | (block > np.asarray(hi) + LIMIT_EPS)
    if out.any():
        idx = tuple(int(v) for v in np.argwhere(out)[0])
        lo_i = np.broadcast_to(lo, block.shape)[idx]
        hi_i = np.broadcast_to(hi, block.shape)[idx]
        raise ChunkError(
            f"{name}[{', '.join(map(str, idx))}] = {block[idx]:.4f} is outside "
            f"{what} [{lo_i:.4f}, {hi_i:.4f}]")


def first_actions(actions, n: int):
    """The first `n` actions of an `EefActions` / `JointActions` (the same type).

    How P5 honours "return the whole chunk": a policy that outputs its own
    full horizon, run with a shorter `action_horizon`, is cut to it here rather
    than refused — the adapter never has to know `H`."""
    import dataclasses
    if actions.n <= n:
        return actions
    return type(actions)(**{f.name: (None if getattr(actions, f.name) is None
                                     else getattr(actions, f.name)[:n])
                            for f in dataclasses.fields(actions)})


def validate_actions(actions, mode: str, cfg, limits: Limits | None = None) -> None:
    """The POLICY's output against the contract. Every error names the policy.

    Horizon: `1 <= n <= H`, and in sync **`n >= P`** — not `== P`, because
    truncating to `P` is now the infra's (it has to happen AFTER the deadline
    shift, see `deadline_shift.py`), so a policy returns its whole chunk and the
    rows past `P` become the lookahead a trunk planner may be shown. Fewer than
    `P` would leave the tick line short, and P5's extrapolation of the next
    inference window (`validate_chunk`) would span two epochs.
    """
    want = EefActions if mode == EEF else JointActions
    if not isinstance(actions, want):
        raise ChunkError(
            f"action_mode = {mode!r} but infer() returned "
            f"{type(actions).__name__}; it must return {want.__name__}")

    blocks = ([("pose", actions.pose)] if mode == EEF
              else [("arm_q", actions.arm_q), ("waist_q", actions.waist_q)])
    blocks += [("hand", actions.hand)]
    for name, block in blocks:
        _finite(f"{want.__name__}.{name}", block)

    if mode == EEF:
        norms = np.linalg.norm(actions.pose[..., 3:], axis=-1)
        if (np.abs(norms - 1.0) > 1e-3).any():
            i, h = (int(v) for v in np.argwhere(np.abs(norms - 1.0) > 1e-3)[0])
            raise ChunkError(
                f"EefActions.pose[{i}, {h}] has a quaternion of norm "
                f"{norms[i, h]:.4f}; the contract wants a unit wxyz quaternion")

    _within("hand", actions.hand, *hand_range(limits),
            "the end effector's travel (Dex-1: gear radians, 0 = closed) — "
            "clip it in the policy")

    if mode == JOINT and limits is not None:
        if limits.arm is not None:
            arm = actions.arm_q.reshape(actions.n, schema.ARM_DOF)
            _within("arm_q (as 14 joints)", arm, limits.arm[:, 0],
                    limits.arm[:, 1], "the URDF joint range")
        if limits.waist is not None:
            # The waist never goes through IK and is written straight to the
            # wire at kp=300, so this is the only place it can be bounded. The
            # absent third slot's range is exactly [0, 0] (`kinematics.waist_limits`).
            _within("waist_q", actions.waist_q, limits.waist[:, 0],
                    limits.waist[:, 1], "the URDF joint range")

    n = actions.n
    if n < 1:
        raise ChunkError("infer() returned no actions")
    if n > cfg.H:
        raise ChunkError(f"infer() returned {n} actions; action_horizon is {cfg.H}")
    if cfg.sync and n < cfg.P:
        raise ChunkError(
            f"sync needs at least replan_period_ticks = {cfg.P} actions, infer() "
            f"returned {n}. The infra truncates to P itself.")


# --- what the ring carries --------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """`n` actions as P5 publishes them: checked, re-timed, cut to `P` in sync.

    Exactly one of `pose` (eef) and `arm_q` (joint) is present. `waist` is
    present exactly when `arm_q` is — the policy's `waist_q`, required in joint
    mode; in eef mode the waist is P6's (the IK owns it), and a chunk cannot say
    otherwise — the constructor refuses an eef chunk with a waist and a joint
    chunk without one, which makes "who commands the trunk" a property of the
    type rather than a convention P6 has to trust.

    `hand` is the `(2, 6)` block P8 reads: the policy's one value per hand in
    dim 0, the rest zero (the ring's width is the wire's).
    """

    hand: np.ndarray                    # (n, 2, 6)
    pose: np.ndarray | None = None      # (n, 2, 7)  eef
    arm_q: np.ndarray | None = None     # (n, 2, 7)  joint
    waist: np.ndarray | None = None     # (n, 3)     joint (always)
    #: The rows `truncate` cut, kept rather than dropped. `None` on a chunk that
    #: was never truncated, which is every chunk in async. They are NOT actions
    #: and no count anywhere includes them — `n` is still the number that will
    #: run. They exist so that `cfg.plan_lookahead_ticks` can publish a few of
    #: them for P6's trunk planner to look at.
    plan: "Chunk | None" = None

    def __post_init__(self) -> None:
        if (self.pose is None) == (self.arm_q is None):
            raise ChunkError("a chunk carries exactly one of pose (eef) and "
                             "arm_q (joint)")
        if self.pose is not None and self.waist is not None:
            raise ChunkError("an eef chunk carries no waist: the IK owns it")
        if self.arm_q is not None and self.waist is None:
            raise ChunkError("a joint chunk always carries the waist "
                             "(`waist_q` is required in joint mode)")
        n = _n("chunk.hand", self.hand)
        want = [("hand", (n, N, schema.HAND_DIM))]
        if self.pose is not None:
            want.append(("pose", (n, N, schema.POSE_DIM)))
        if self.arm_q is not None:
            want.append(("arm_q", (n, N, schema.ARM_DOF // N)))
        if self.waist is not None:
            want.append(("waist", (n, schema.WAIST_DOF)))
        for name, shape in want:
            got = np.shape(getattr(self, name))
            if got != shape:
                raise ChunkError(
                    f"chunk.{name} has shape {got}, the contract wants {shape}")

    @property
    def mode(self) -> str:
        return EEF if self.pose is not None else JOINT

    @property
    def n(self) -> int:
        return int(np.shape(self.hand)[0])

    def _rows(self, s: slice) -> "Chunk":
        cut = (lambda a: None if a is None else a[s])
        return Chunk(hand=self.hand[s], pose=cut(self.pose),
                     arm_q=cut(self.arm_q), waist=cut(self.waist))

    def truncate(self, n: int) -> "Chunk":
        """The first `n` actions — sync mode's truncation to `P`. The remainder
        is kept on `plan`; the returned chunk's `n` and contract are unchanged."""
        tail = self._rows(slice(n, None)) if self.n > n else None
        head = self._rows(slice(None, n))
        return Chunk(hand=head.hand, pose=head.pose, arm_q=head.arm_q,
                     waist=head.waist, plan=tail)


def validate_chunk(chunk: Chunk, cfg) -> None:
    """Length and finiteness, checked before the chunk can reach the ring.

    THE SYNC INVARIANT: a sync chunk is exactly `P` actions. P5 extrapolates
    the next inference window from the tick line (`p5_policy.PolicyLoop`), and
    that is valid only because a sync chunk is exactly
    `P` actions long: the previous epoch's last tick is `nP` and the window
    opens at `nP + 2`. A chunk one action longer lets P5 read a pre-inference
    deadline and schedule `t_next_inference` in the *past*, by exactly the
    inference latency — and nothing downstream would report it.

    The lookahead rows are checked too. They are never published as targets,
    but P6 hands them to the trunk planner, whose OUTPUT is the commanded waist
    for the whole chunk — so a NaN among them reaches `rt/lowcmd` by a longer
    road, and one with no clamp on it at all.
    """
    for name in ("pose", "arm_q", "waist", "hand"):
        _finite(f"chunk.{name}", getattr(chunk, name))
    if chunk.plan is not None and chunk.plan.n:
        for name in ("pose", "arm_q", "waist", "hand"):
            _finite(f"chunk.plan.{name} (lookahead rows; the trunk plan solved "
                    f"over them is commanded)", getattr(chunk.plan, name))

    if chunk.n < 1:
        raise ChunkError("chunk is empty")
    if chunk.n > cfg.H:
        raise ChunkError(
            f"chunk of {chunk.n} actions exceeds action_horizon {cfg.H}")
    if cfg.sync and chunk.n != cfg.P:
        raise ChunkError(
            f"sync chunk has {chunk.n} actions, must be exactly "
            f"replan_period_ticks = {cfg.P}. P5's inference-window "
            f"extrapolation spans two epochs otherwise.")


def hand_range(limits: Limits | None) -> tuple[float, float]:
    """The travel `hand` is bounded by: the robot's, else the Dex-1's."""
    if limits is not None and limits.hand is not None:
        return limits.hand
    return DEX1_HAND_RANGE


def hand_block(actions) -> np.ndarray:
    """`(n, 2, 6)` for P8: the policy's `hand` (n, 2) in dim 0."""
    out = np.zeros((actions.n, N, schema.HAND_DIM))
    out[:, :, 0] = actions.hand
    return out


__all__ = ["ACTION_MODES", "Actions", "Chunk", "ChunkError", "EEF", "EefActions",
           "DEX1_HAND_RANGE", "JOINT", "JointActions", "Limits", "MODE_CODE",
           "MODE_NAME", "hand_block", "hand_range", "validate_actions",
           "validate_chunk"]

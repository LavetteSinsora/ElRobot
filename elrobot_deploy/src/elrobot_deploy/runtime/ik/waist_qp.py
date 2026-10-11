"""Whole-body IK — the arm plus the trunk, solved a whole chunk at a time.

`arm_qp.DualArmSolver` answers "where must the 14 arm joints be", one tick at a
time, with the trunk pinned at the measured waist. This module answers "where
must the arm AND the trunk be", 25 ticks at a time. It is opt-in
(`ik = "waist_qp"`, and `waist_planner.py` uses its passes A+B); with an
arm-mode solver nothing here is imported.

Where it sits among the solvers: `docs/infra_logic/ik_solvers.md`, "`waist_qp`
— the whole-body QP". The design itself is this docstring. Numbers quoted
below were measured on the reference robot (2026-09) unless stated otherwise.

---

## Why the trunk is worth anything at all

Because it is the only part of the machine that can move the flange *within the
frame the policy commands in*. Measured on the compiled model, displacement of
the flange in the base frame per unit of each non-arm DOF:

    LZ_mt_Joint (column stage 1, full 210 mm)   0.000000 mm
    LZ_it_Joint (column stage 2, full 210 mm)   0.000000 mm
    Left/Right_Wheel_Joint (0.5 rad)            0.000000 mm
    torso_Joint (waist YAW, 0.2 rad)           33.899555 mm
    Yaw_Joint   (waist BOW, 0.2 rad)           54.648988 mm

Exactly zero for the column and the base, and that is structural rather than
unlucky: `g1d_model.BASE_BODY` is `LZ_it_Link`, which sits ABOVE the lift column and
rigidly on the wheeled base, so the pose frame rides both of them. Raise the
column and the target rides up with the flange. Drive the base and the whole
frame drives with it.

So in the base frame the waist is the whole of whole-body IK, and a solver that
adds the column and the base to that frame is adding two columns of zeros to the
Jacobian. This is worth stating plainly because "add the base and the column" is
the obvious reading of "whole-body" and it does not, by itself, do anything.

## What makes the column and the base real: the snapshot frame

They become live the moment the target stops riding the robot. At the top of a
chunk this module captures

    T_snap = backend.base_pose()          # the base frame, right now, in world

and expresses all 25 of the chunk's poses against it, ONCE:

    target_world[k] = T_snap @ pose_base[k]

Those 25 world poses are then held fixed for the chunk while the trunk moves
underneath them. Now extending the column raises the flange in the world and the
target does not follow, so the column is 420 mm of extra vertical reach; driving
the base is however many metres of extra horizontal reach.

**This is also the more correct semantics, independently of the extra DOF.** The
policy observed an anchor and emitted deltas from it; the poses those compose to
are where the hand should be *in the world at the instant of that observation*.
If the trunk then moves, the cup on the table has not moved, and a target
expressed against the moving trunk would chase it. While the trunk does not
move the two readings coincide exactly. It is what stops the extra DOF from
being self-defeating.

The frame is re-captured at every replan, which is what bounds odometry drift to
a single chunk (~800 ms).

## The shape of a chunk solve

Three passes rather than a monolithic chunk-MPC, for one reason: when
something goes wrong, the pass it went wrong in says what kind of wrong it is.

  **A — where would the trunk LIKE to be.** Solve each of the 25 ticks
  independently with the trunk free and no rate limit at all. This is not a
  trajectory and is never executed; it is 25 answers to "if the trunk could
  teleport, where would it stand for this tick".

  **B — retime that into something the hardware can do.** One small banded QP
  per trunk DOF over the 25 knots: stay near pass A, stay smooth, start from
  where the trunk actually is, and respect rate and acceleration bounds.
  **This pass is where anticipation comes from** and it is the only place it
  could come from. Being near the pass-A answer at tick 20 while bounded to
  0.35 rad/s and smooth forces the motion to begin around tick 8 — nobody wrote
  "start early", it is what the bounds and the smoothness cost imply. A reactive
  per-tick solver cannot produce it at any weight setting.

  **C — solve the arm along that trunk trajectory.** Per-tick differential IK,
  the same machinery `arm_qp.DualArmSolver` runs, with the trunk DOF present but
  pulled hard toward pass B's trajectory.

  The trunk is pulled, not pinned, and that is the accuracy-first requirement
  made concrete: if following pass B exactly would cost position error, pass C
  is allowed to deviate from it, because the residual is the thing we promised
  not to trade. Pass B decides what the trunk does when accuracy is indifferent;
  pass C overrules it when accuracy is not.

## Rate limiting, and a trap

**`mink.VelocityLimit` bounds one QP ITERATE, not one tick.** The solve loop
runs up to `ik_max_iters` iterates of `v * dt` per tick, so a limit of 1.0 rad/s
at dt = 1/30 permits 0.033 rad per iterate and therefore up to 0.67 rad in a
single 33 ms tick if the loop runs its full 20. That is 20x the rate anybody
asked for, and it is invisible: every iterate is individually inside the limit.

A waist-enabled run reached 0.436 rad of trunk yaw — 25 degrees — on a
"1.0 rad/s" setting for exactly this reason.

So the real limit is enforced here as an explicit box on the configuration,
re-applied after every iterate, against the tick's own starting posture:

    q_slow  in  q_slow_start  +-  rate * dt

Clamped inside the loop rather than after it, so that the remaining iterates
re-solve from the clamped configuration and the arm takes up the slack —
clamping only at the end would move the trunk out from under an arm solution
that assumed it, and put the error back at the flange where we were trying to
remove it.

## What is deliberately NOT here

**No async.** Sync only. The chunk-horizon passes assume the 25
targets are all known and all belong to one epoch, which only sync guarantees
(chunks partition the tick line, one contributor each). In async two chunks
overlap by design and pass A would be solving a blend that changes underneath it.

**No wheel odometry.** The base's own motion is dead-reckoned from what this
module commanded, in the snapshot frame, and thrown away at the next replan. The
G1-D publishes no odometry this package consumes, and over 800 ms the drift
is ignorable. It is the reason the frame is re-captured every chunk rather
than maintained across a run.

**No head-camera gaze task.** A live risk, recorded rather than solved: a
policy trained with the waist frozen and the base static sees, under trunk
excursion, a head camera in poses no demonstration contained. That is why
excursion is worth reporting beside the residual, and why the default weights
are set where they are — see `TIERS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from elrobot_deploy.robot import kinematics as kin
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.timing import now
from elrobot_deploy.robot.kinematics import SIDES

#: The trunk DOF groups, coarsest knob this module has. A group is on or off;
#: within a group the per-DOF costs come from `WbikWeights`.
#:
#: Ordered by the hierarchy the request states, most tolerated first: the waist
#: should be the thing that moves, the column should barely move, the base
#: should be a last resort and its rotation the last resort of the last resort.
GROUPS = ("waist", "column", "base")

#: Floor on a pass-B knot's tracking weight — see `_knot_weights`. Not zero,
#: because a chunk in which the trunk is wanted nowhere would otherwise have an
#: objective with no tracking term at all and be decided entirely by smoothness.
_KNOT_FLOOR = 0.02


@dataclass(frozen=True)
class WbikWeights:
    """The cost vector, and the rate bounds it is solved against.

    Two families of number and they do different jobs, which is worth keeping
    straight because tuning the wrong one produces a plausible-looking robot
    that is wrong in a way no metric catches.

    **`cost_*` decides WHO MOVES.** It is the diagonal of `M` in the velocity
    cost `‖u‖²_M`, carried by `mink.DampingTask`. It is a preference, expressed
    against the arm's 1.0: a column entry of 40 means "a metre of column is
    worth 40 radians of arm", and the QP buys whichever is cheaper for the
    target in front of it. Raising it never makes the robot more accurate; it
    makes the robot reach the same place a different way.

    **`rate_*` decides WHAT IS POSSIBLE.** These are hardware, in rad/s and m/s,
    and they are enforced as a hard per-tick box (see the module docstring on
    why `mink.VelocityLimit` alone is not that box). A run that violates one is
    a run the machine cannot execute, so the (since removed) `wbik_eval` checks them rather than
    trusting them.

    `posture_*` is the pull back toward nominal — a
    posture POTENTIAL rather than a nullspace velocity, so that a cyclic EE path
    returns the trunk to where it started instead of accumulating drift.
    """

    #: Absolute size of the whole `M` diagonal. The `cost_*` entries below are
    #: RATIOS against `cost_arm`; this is what turns them into numbers the QP
    #: weighs against a position residual.
    #:
    #: **It is not a free parameter and it is not a taste.** `mink.DampingTask`
    #: contributes `lambda_i^2 * v_i^2` against a frame task whose residual is
    #: `position_cost^2 * ||J v - x_dot||^2` with `position_cost = 1.0`. At a
    #: typical 20 mm error over a 33 ms tick, `x_dot` is ~0.6 m/s and a useful
    #: `v` is ~1 rad/s, so the tracking term is ~0.36 — and a `lambda` of 6
    #: prices one radian of waist at 36, a hundred times the entire benefit of
    #: solving the tick at all. The QP then correctly refuses to move anything.
    #:
    #: Measured, on the worst tick of one recorded rollout: at `damping_scale = 1.0`
    #: every joint moved 1.5e-5 rad per iterate and 20 iterates took the residual
    #: from 20.933 mm to 20.882 mm. The arm was as frozen as the trunk, which is
    #: also why the `off` tier — meant to reproduce the deployed solver exactly —
    #: came out at 1.541 mm against a recorded 0.808 mm. `DualArmIK` has no
    #: damping task at all; it passes `damping=1e-3` to `solve_ik` as a uniform
    #: regulariser, and that is the scale this has to match to be the same solver.
    damping_scale: float = 1e-3

    # --- who moves (ratios, arm = 1.0) -------------------------------------
    cost_arm: float = 1.0
    #: The waist is the DOF we WANT recruited, so it is cheap — but not as cheap
    #: as the arm, or it would be used for motions the arm could have absorbed
    #: on its own, and every one of those is head-camera excursion bought for
    #: nothing.
    cost_waist_yaw: float = 6.0
    #: Dearer than the yaw. The bow carries the whole upper body against gravity
    #: and rests on a mechanical stop 0.039 rad forward of its commanded zero,
    #: so it is the axis whose commanded angle is
    #: least trustworthy.
    cost_waist_bow: float = 12.0
    #: Per metre, against radians — so the raw number looks large and is not.
    #: 0.42 m of travel at cost 40 is worth ~17 "arm radians" end to end.
    cost_column: float = 40.0
    cost_base_lin: float = 400.0
    #: The most expensive thing on the robot to move, by request and by physics:
    #: base yaw swings the hand on a lever arm equal to its reach, so a small
    #: rotation throws the flange a long way and any error in it is amplified
    #: the same amount.
    cost_base_yaw: float = 1200.0

    # --- pull back toward nominal ------------------------------------------
    posture_arm: float = 0.02          # matches cfg.ik_posture_cost
    posture_waist: float = 0.10
    posture_column: float = 0.40
    posture_base: float = 2.00

    # --- what the hardware can do (per tick, hard) -------------------------
    #: rad/s. The robot file's `[limits] qdot_max` is the vendor's uniform 2*pi for
    #: every joint on the wire, and P6 already clamps the arm against it.
    rate_arm: float = 2.0 * np.pi
    #: rad/s, PER AXIS. The two waist joints are not alike and one number for
    #: both was costing the good one. Measured on the reference robot,
    #: 2026-09-09 (position steps and ramps on each axis), steady-state tracking
    #: error against commanded rate, with the velocity feedforward P7 sends:
    #:
    #:     yaw   3.8 +  4.5*v mrad      2.06 rad/s achieved, 36 ms rise
    #:     bow  14.4 + 51.0*v mrad      overshoots 41-55%, 70 ms rise
    #:
    #: **The yaw tracks eleven times better.** At 1.0 rad/s it is 8 mrad behind
    #: its command; the bow at 0.35 is already 32. That is a real difference in
    #: the joint, not in the tuning: the yaw is a vertical axis carrying no
    #: gravity load, the bow is horizontal and carries the whole upper body.
    #:
    #: **But the residual does not care, and that is the surprise.** Swept over
    #: 6 recordings, holding the other axis at 0.35:
    #:
    #:     yaw   bow    TCP mean   p95
    #:     0.20  0.35     0.882   4.735
    #:     0.35  0.35     0.847   4.937      <- shipped
    #:     1.00  0.35     0.858   4.760
    #:     0.60  0.20     0.946   5.619
    #:     1.00  0.20     0.952   5.687
    #:     0.60  0.15     1.047   6.321
    #:
    #: The YAW rate barely matters — anything from 0.20 to 1.00 moves the mean
    #: by under 4%, because the yaw never asks for more than 0.120 rad and even
    #: 0.20 rad/s covers that in half a chunk. **The BOW rate is the whole
    #: story**: 0.35 -> 0.20 costs 12%, 0.35 -> 0.15 costs 24%, because the bow
    #: asks for 0.201 rad and 0.20 rad/s cannot deliver it in one chunk.
    #:
    #: So a fast yaw CANNOT buy back a slow bow. That was the hope and the
    #: sweep refuses it: 1.00/0.20 is no better than 0.60/0.20.
    #:
    #: Both ship at 0.35, which is the knee — 0.60/0.60 measured 0.838, a 1%
    #: gain not worth the extra travel. The split is kept even though the two
    #: values agree, because they are now known to be independent and the bow
    #: is the one an operator may need to cap for stability.
    #:
    #: **On tipping: 0.35 rad/s is not a violent motion.** It is 20 deg/s, and
    #: the bow's whole excursion is 0.201 rad over an 800 ms chunk. The alarming
    #: speeds a position STEP produces — 2.4 to 4.6 rad/s measured — come from
    #: the servo executing a step as fast as it can. The
    #: solver never commands a step; pass B rate-limits and smooths every
    #: trajectory it emits.
    rate_waist_yaw: float = 0.35
    rate_waist_bow: float = 0.35
    #: m/s. Unitree G1-D high-level API 1002: vz in [-1,1] scaled by 76.5 mm/s
    #: (`runtime/drivers/column.py::FULL_SCALE_MPS`).
    rate_column: float = 0.0765
    #: m/s. The hardware allows 1.5 (Unitree G1-D high-level API 1001) and that is a
    #: driving speed, not a manipulating speed. Capped here at a fifth of it.
    rate_base_lin: float = 0.30
    #: rad/s. Hardware allows 0.6; this is a third of it, for the lever-arm
    #: reason in `cost_base_yaw`.
    rate_base_yaw: float = 0.20

    # --- acceleration, for pass B only -------------------------------------
    #: Fraction of the rate bound that may be gained or lost in one tick. The
    #: acceleration bound is most of the motion's visual smoothness;
    #: this is that bound, expressed relatively so one number serves every DOF.
    accel_frac: float = 0.25

    # --- how hard pass C is held to pass B's trajectory ---------------------
    #: Cost on the trunk's deviation from the retimed trajectory. High enough
    #: that the trunk follows it, finite so that accuracy can still overrule it.
    track_slow: float = 5.0

    # --- pass B shape -------------------------------------------------------
    smooth_vel: float = 4.0            # lambda on first difference
    smooth_acc: float = 40.0           # mu on second difference

    #: Refuse to command the bow behind where it is actually resting.
    #:
    #: **This is a hardware fact, not a preference.** The trunk rests against a
    #: physical obstruction 0.039 rad forward of its commanded zero and no torque
    #: moves it (measured: the stop is ~49x stiffer than kp=300).
    #: The URDF's lower limit is -0.0436 rad, which the model believes and the
    #: machine does not, so the QP will happily solve the arm into a trunk pose
    #: 0.08 rad behind anything the robot can reach.
    #:
    #: It does, too. Measured over three rollouts with this off: **39.0% of
    #: ticks (1194 of 3063) commanded the bow behind its rest**, the worst by
    #: 0.082 rad, with the commanded angle bottoming out exactly on the URDF's
    #: -0.0436. On hardware every one of those is a no-op — the joint stays on
    #: the stop — and the arm is then solved for a frame the robot is not in,
    #: which is the same class of silent metre-scale bias `kin.place`'s docstring
    #: warns about and the reason `p6_target` solves against the MEASURED trunk
    #: in the first place.
    #:
    #: So the bow's floor is the trunk's own measured rest at the top of the
    #: chunk, and the solver may only ever bow FORWARD of it. Forward is
    #: unaffected: the joint has 2.36 rad of travel that way and the stop is
    #: behind it.
    #:
    #: **TURN THIS OFF IF THE OBSTRUCTION IS EVER REMOVED, and do not leave it
    #: on "to be safe".** The floor is derived from the MEASURED rest, which is
    #: the right bound only while that rest is a wall the joint cannot leave.
    #: With the obstruction gone the trunk still sits somewhere — wherever kp
    #: and gravity settle it — and flooring at that position would forbid
    #: perfectly reachable travel for no reason, silently, in a way no metric
    #: distinguishes from the arm simply not needing it.
    #:
    #: What it is worth: the removed `wbik_eval --no-bow-stop` prices exactly this, and the
    #: report carries the number. On the four-recording hard subset, removing
    #: the obstruction takes the improvement from 69.5% to 75.9% — the backward
    #: travel is genuinely useful, it is just not available today.
    bow_respects_stop: bool = True

    #: The bow's floor as a FIXED angle (the physical stop, +0.040 rad on the
    #: reference robot) rather than the chunk's measured rest. The
    #: measured-rest floor ratchets: with the
    #: trunk at rest it IS the stop and the floor is right; once a chunk has
    #: bowed the trunk to 0.3 rad the next chunk's floor is 0.3 rad, and the
    #: bow can only ever go further. Over 11 recorded waist-mode runs the
    #: commanded bow climbed 0.04 -> 0.6 and never came back more than the
    #: servo's own slop per seam. None = the measured rest (`waist_qp`'s
    #: behaviour; `waist_planner` sets it). Only read when `bow_respects_stop`.
    bow_floor: float | None = None

    #: Pull the trunk toward NOMINAL (yaw 0, bow at `bow_floor`) instead of
    #: toward the chunk's own origin. Pass A's posture target is set from the
    #: seed configuration, so the "pull back toward nominal" the weights
    #: describe was a pull toward wherever the trunk already was — a per-chunk
    #: damping, not a potential. With a real nominal, a trunk no longer needed
    #: forward drifts back at `posture_waist`. Needs `bow_floor`.
    nominal_trunk: bool = False

    #: Solve pass A at every `pass_a_stride`-th tick and interpolate between,
    #: rather than at all K of them. 1 is "solve every tick" and is exact.
    #:
    #: Pass A is 41.9% of the chunk solve on the deploy box (183.7 ms of 438.7),
    #: and it is the pass whose answer is least entitled to that share: it does
    #: not produce the published trajectory. It produces a DEMAND CURVE that
    #: pass B then retimes under a rate limit, an acceleration limit, an
    #: excursion box and a per-knot weight. A demand curve sampled every fourth
    #: tick and linearly filled is a different input to that retiming only where
    #: the demand has structure finer than four ticks — and the tier's own rate
    #: limit forbids the trunk from having any. At `rate_waist = 0.35 rad/s` the
    #: waist can move 0.047 rad in four ticks, against an excursion cap of 0.12.
    #:
    #: The failure this can produce is a SPIKE: one tick that wanted the trunk
    #: badly, sitting between knots that did not, averaged away before pass B
    #: ever sees it. That is why this is to be measured against residual over
    #: recordings rather than argued.
    #:
    #: The first and last tick are always knots, whatever the stride, so a chunk
    #: never extrapolates past its own ends.
    #:
    #: MEASURED, deploy box, pinned to p6_target's core, 8 recordings / 11,124
    #: ticks, `waist` tier. Residual is at the virtual TCP:
    #:
    #:     stride    TCP mean    ms/chunk    vs arms-only
    #:     off        2.016 mm      119.3           --
    #:     1          1.325 mm      564.5       -34.3%
    #:     3          1.326 mm      342.1       -34.2%
    #:     8          1.337 mm      300.7       -33.7%
    #:
    #: 8 is the shipped setting: it halves the chunk solve for 0.9% of residual,
    #: and 0.9% of 1.33 mm is 12 micrometres. The curve is flat rather than
    #: monotone -- stride 3 measured fractionally BETTER than stride 1 -- which
    #: is the signature of a quantity that was never resolution-limited.
    pass_a_stride: int = 8

    #: Hard cap on how far a slow DOF may travel from where the chunk started,
    #: in that DOF's own units. `None` means no cap of ours — the joint's URDF
    #: range and the rate limit are the whole bound.
    #:
    #: **The waist has no cap.** A cap (0.12 rad was once used) is never
    #: kinematic: it exists to keep a head camera near a policy's training
    #: distribution, because trunk excursion moves the head view into poses no
    #: demonstration contained. A policy that reads the head camera may want
    #: one (`excursion_waist`); the infra does not presume it.
    #:
    #: Removing it HELPS, and the trunk does not run away without it. Measured
    #: over 351 chunks: 1.427 mm -> 1.394 mm mean, p95 9.978 -> 9.648, and the
    #: worst per-chunk excursion the solver actually chose was 0.201 rad — well
    #: inside what `rate_waist` permits over a chunk (0.29 rad) and nowhere near
    #: the URDF's range (yaw +-2.7053, bow -0.0436..+2.3562). The rate limit was
    #: always the binding constraint; a cap cuts into useful travel.
    #:
    #: The column and the base keep theirs, for a different and non-negotiable
    #: reason: the column has 431 mm of measured physical travel, and the base
    #: has no limit of any kind, so an uncapped base will happily propose
    #: driving a metre to save 3 mm at the flange.
    excursion_waist: float | None = None
    excursion_column: float = 0.06
    excursion_base_lin: float = 0.15
    excursion_base_yaw: float = 0.08


#: The configurations this module can run in, coarse to fine. The shipped
#: `waist_qp` solver is the `waist` tier; the wider two are not deployable
#: (nothing commands the column or the base from P6).
#:
#: These are not four ideas — they are one idea with the hierarchy tightened one
#: notch at a time. `off` is the arm-only QP reached through this code path,
#: and it exists so that "this module with everything disabled" can be shown
#: to reproduce `arm_qp`'s numbers exactly rather than approximately.
TIERS: dict[str, tuple[str, ...]] = {
    "off": (),
    "waist": ("waist",),
    "waist_column": ("waist", "column"),
    "full": ("waist", "column", "base"),
}


@dataclass
class ChunkSolution:
    """What one `solve_chunk` produced, one row per tick.

    `err` is the same quantity `BODY_TARGET.err` has always carried — worst
    flange POSITION error in metres, at the configuration actually published —
    so a wbik record and a baseline record are directly comparable without
    anybody having to remember which solver wrote them.
    """

    #: (K, 14) arm joint angles.
    arm: np.ndarray
    #: (K, 3) commanded waist, in `kin.WAIST_JOINT_NAMES` slot order. Slot 2 is
    #: always zero — there is no motor behind it.
    waist: np.ndarray
    #: (K,) worst-flange position residual, metres.
    err: np.ndarray
    #: (K,) worst **virtual TCP** position residual, metres — the residual to
    #: judge the solver by. `err` above measures the flange, which is 184.759 mm
    #: short of where the robot actually touches anything, so it cannot see
    #: orientation error turning into position error over that lever arm. See
    #: `WholeBodySolver._tcp_error`.
    tcp_err: np.ndarray
    #: (K,) worst-flange orientation residual, radians. New beside `err`: with
    #: the trunk free there is more than one way to reach a position and the
    #: orientation is what distinguishes them, so a position-only metric would
    #: hide a solver that reaches the right point facing the wrong way.
    rot_err: np.ndarray
    #: (K,) column height, metres above the chunk's starting height. Commanded
    #: as a velocity on `rt/cmd_hispeed`; this is its integral.
    column: np.ndarray
    #: (K, 3) base pose (x, y, yaw) in the snapshot frame, relative to its
    #: origin. Commanded as (v, omega); this is the dead-reckoned integral.
    base: np.ndarray
    #: True if every tick solved inside its budget.
    ok: bool
    #: Wall-clock seconds the whole chunk took.
    elapsed: float
    #: Per-pass timing, for the budget table in the report.
    timing: dict = field(default_factory=dict)
    #: Always False, kept for readers that expect the field.
    #: There is no arms-only comparison to fall back TO: the trunk is
    #: planned once and pinned, so every published tick is the whole-body
    #: answer. See `WholeBodySolver.plan_chunk`.
    fell_back: bool = False

    def __len__(self) -> int:
        return len(self.arm)


def _banded_retime(ideal, start, rate, accel, dt, lo, hi,
                   w_track, w_vel, w_acc):
    """Pass B for one DOF: the closest smooth, rate-feasible trajectory.

    Minimises `sum w_track_k (s_k - ideal_k)^2 + w_vel*sum (ds)^2
    + w_acc*sum (d2s)^2` subject to `|ds| <= rate*dt`,
    `|d2s| <= accel*dt^2` and `lo <= s_k - start <= hi`, with `s_{-1} = start`.

    Solved by projected gradient rather than by handing it to a QP solver. The
    objective is a tridiagonal quadratic and the feasible set is a box plus two
    difference bounds, so a few hundred cheap iterations land well inside the
    tolerance that matters — the answer feeds a differential IK pass that will
    move off it anyway. It also keeps this module's only hard dependency the one
    `arm_qp.py` already has, instead of putting `qpsolvers` on P6's import path for
    a 25-variable problem.

    **`w_track` is per-knot, and that is what makes the pass anticipatory.**
    It started as one number for the whole chunk and that version does not
    anticipate at all: on a demand that appears only at tick 20, the twenty
    earlier knots whose ideal is "stay where you are" outvote it twenty to five,
    and the trunk leaves at tick 14 and never arrives. Measured — it reached
    0.0696 of a 0.10 rad demand, and 0.00033 of it by tick 10.

    The fix is to notice that those twenty knots are not asking for anything.
    Pass A puts the ideal at the origin there because the trunk made no
    difference to that tick, not because the trunk is needed at the origin. So
    the caller weights each knot by how much the trunk actually bought it (see
    `_pass_a`'s `gain`), the indifferent knots stop pulling, and the tick-20
    demand propagates backward through `w_acc` until the trajectory leaves early
    enough to arrive on time. Nobody writes "start early"; it is what an
    unopposed smoothness term does with a rate bound.
    """
    K = len(ideal)
    if K == 0:
        return np.zeros(0)
    ideal = np.asarray(ideal, np.float64)
    w_track = np.broadcast_to(np.asarray(w_track, np.float64), (K,)).copy()
    s = np.clip(ideal.copy(), start + lo, start + hi)
    dmax = rate * dt
    amax = accel * dt * dt

    # --- backward reachability: the LATEST this DOF may leave --------------
    #
    # The soft objective alone does not arrive on time, and no weighting fixes
    # it. Measured, on a demand of 0.10 rad that appears at tick 20 of 25 with a
    # 0.35 rad/s bound: over the whole (w_vel, w_acc) grid the best trajectory
    # reached 0.064 by tick 20 and 0.094 by tick 24. The objective is indifferent
    # between arriving late and arriving short — spreading the miss across the
    # five demanding knots is cheaper than rushing to meet the first of them —
    # so it always chooses to trail.
    #
    # Arrival is a reachability question, not a cost question, so it is answered
    # as one. For every knot that carries a real demand, the trajectory must be
    # within `dmax * (j - k)` of it at tick k or it cannot possibly get there by
    # tick j. Propagated backward:
    #
    #     lb[k] = max( demand[k], lb[k+1] - dmax )
    #     ub[k] = min( demand[k], ub[k+1] + dmax )
    #
    # which is the standard latest-departure tube. Intersecting the trajectory
    # with it each iteration is what actually makes the trunk leave early; the
    # smoothness weights then decide only how gracefully.
    #
    # Only knots the caller weighted as real demands take part — an indifferent
    # knot must not constrain anything, or the tube collapses onto pass A's
    # arbitrary answer at every tick. If a chunk's demands conflict (one knot
    # wants +0.1, another -0.1, closer together than the rate allows) the tube
    # inverts; the midpoint is then the best available compromise and the
    # forward sweeps still guarantee feasibility.
    demand = w_track >= 0.5 * float(w_track.max())
    lb = np.full(K, -np.inf)
    ub = np.full(K, np.inf)
    for k in range(K - 1, -1, -1):
        nlb = -np.inf if k == K - 1 else lb[k + 1] - dmax
        nub = np.inf if k == K - 1 else ub[k + 1] + dmax
        if demand[k]:
            nlb, nub = max(nlb, ideal[k]), min(nub, ideal[k])
        lb[k], ub[k] = nlb, nub
    lb = np.maximum(lb, start + lo)
    ub = np.minimum(ub, start + hi)
    bad = lb > ub
    if bad.any():
        mid = 0.5 * (lb[bad] + ub[bad])
        lb[bad] = ub[bad] = mid

    def _project(s):
        """Box, then rate, then acceleration — and the rate bound RE-APPLIED.

        The acceleration sweep sets `s_k` from `p + v +- amax`, which is a
        statement about the second difference and says nothing about the first;
        left as the last word it can hand back a step larger than `dmax`. It
        did: a 0.35 rad/s bound at dt = 1/30 permits 0.011667 per tick and the
        unrepaired sweep returned 0.012948.
        """
        s = np.clip(s, lb, ub)
        for _ in range(2):
            p = start
            for k in range(K):
                s[k] = min(max(s[k], p - dmax), p + dmax)
                p = s[k]
            if amax <= 0.0 or K < 2:
                break
            p, v = start, 0.0
            for k in range(K):
                s[k] = min(max(s[k], p + v - amax), p + v + amax)
                s[k] = min(max(s[k], p - dmax), p + dmax)
                v, p = s[k] - p, s[k]
        # The box LAST as well as first. The two sweeps above bound differences
        # and know nothing about the excursion cap, so either can walk the
        # trajectory out of it — a 0.12 rad cap came back as 0.1244.
        #
        # `lb`/`ub` rather than the raw box: they already include it, and
        # re-clipping to the box alone would discard the reachability tube that
        # the whole anticipation rests on.
        return np.clip(s, start + lo, start + hi)

    # Lipschitz-safe step for the tridiagonal quadratic: the second-difference
    # operator has spectral norm <= 16, the first <= 4.
    step = 1.0 / (2.0 * (float(w_track.max()) + 4.0 * w_vel + 16.0 * w_acc))
    s = _project(s)
    for _ in range(400):
        g = 2.0 * w_track * (s - ideal)
        prev = np.concatenate(([start], s[:-1]))
        d1 = s - prev
        g += 2.0 * w_vel * d1
        g[:-1] -= 2.0 * w_vel * d1[1:]
        if K >= 2:
            d2 = np.diff(d1)
            g[1:] += 2.0 * w_acc * d2
            g[:-1] -= 2.0 * w_acc * d2
        s = _project(s - step * g)
    return s


def _nominal_trunk_posture_class():
    """Built lazily: mink is imported only inside the solvers."""
    import mink

    class _NominalTrunkPosture(mink.PostureTask):
        """A `PostureTask` whose trunk entries always target `nominal`, however
        the solver re-seeds it. `set_target_from_configuration` is the hook
        every pass uses; the arm entries still follow the configuration."""

        def __init__(self, model, *, cost, slow_q, nominal):
            super().__init__(model, cost=cost)
            self._slow_q, self._nominal = list(slow_q), np.asarray(nominal, np.float64)

        def set_target_from_configuration(self, configuration):
            q = np.asarray(configuration.q, np.float64).copy()
            q[self._slow_q] = self._nominal
            self.set_target(q)

    return _NominalTrunkPosture


class WholeBodySolver:
    """Chunk in, 25 whole-body configurations out.

    Mirrors `arm_qp.DualArmSolver` deliberately: same construction arguments, same
    `solve()` for a single tick, plus `solve_chunk()`. A caller that only ever
    calls `solve()` with every group disabled gets bit-identical answers to the
    deployed solver, which is what makes the flag safe to leave in.
    """

    def __init__(self, cfg, *, backend=None, urdf: str | None = None,
                 groups: Sequence[str] = ("waist",),
                 weights: WbikWeights | None = None,
                 rel: float | None = None, abs_tol: float = 2e-6, min_iters: int = 2,
                 tcp=None):
        """`rel` turns on the convergence stop of `arm_qp.DualArmSolver` in every
        pass and per-tick solve here (off by default).
        `tcp` is the robot file's (7,) TCP or None. With one, the QP drives the
        TCP (targets composed with it, the model's TCP site as the task frame)
        and `err` / `tcp_err` / the early stop are at the TCP; without, all of
        them are at the flange (`arm_qp.DualArmSolver`, the same rule)."""
        import mujoco

        from elrobot_deploy.robot import mujoco_model as _g1
        from elrobot_deploy.robot import g1d_model as _g1d

        #: The robot file's TCP, (7,) or None, and flange -> TCP as a 4x4.
        self.tcp = None if tcp is None else np.asarray(tcp, np.float64).reshape(7)
        self._Xtcp = None if self.tcp is None else kin.pose_to_mat(self.tcp)
        bad = set(groups) - set(GROUPS)
        if bad:
            raise ValueError(f"unknown wbik group(s) {sorted(bad)}; "
                             f"known groups are {GROUPS}")
        self._mujoco = mujoco
        self.cfg = cfg
        self.dt = float(cfg.dt)
        self.early_stop = float(cfg.ik_early_stop)
        self.max_iters = int(cfg.ik_max_iters)
        self.rel, self.abs_tol, self.min_iters = rel, float(abs_tol), int(min_iters)
        self.groups = tuple(g for g in GROUPS if g in groups)
        self.w = weights or WbikWeights()

        # The base needs three joints the shipped URDF does not have, so it
        # needs the other model. Built only when the base is actually enabled:
        # three extra DOF cost nothing to solve but they do change `base_pose()`
        # from a constant into a function, and a run that is not using them
        # should not pay that.
        mobile = "base" in self.groups
        if backend is None:
            backend = _g1d.G1DBackend(urdf, mobile_base=mobile, tcp=self.tcp)
        elif (self.tcp is None) != (getattr(backend, "tcp", None) is None):
            raise ValueError("the backend's TCP site does not match this "
                             "solver's tcp: build it with G1DBackend(tcp=...)")
        elif mobile and not getattr(backend, "mobile_base", False):
            raise ValueError(
                "wbik group 'base' needs a backend built with mobile_base=True; "
                "the shipped URDF roots at AGV_link and compiles it into the "
                "worldbody, where it has no DOF to solve for")
        self.backend = backend
        m = backend.model
        self._jmap = kin.joint_addresses(backend)

        # The collision set follows the BACKEND's model, not this solver's
        # groups. Those usually agree and once did not: a caller may build a
        # solver with empty groups on a backend that is already MOBILE, so a
        # `full` tier hands a mobile backend to a groups-free solver. Keyed
        # off `mobile` that picked the fixed model's set, which names `"world"`
        # — a body that stops existing the moment `AGV_link` is given a parent —
        # and construction raised.
        trunk = (_g1d.MOBILE_TRUNK_COLLISION_BODIES
                 if getattr(backend, "mobile_base", False)
                 else _g1d.TRUNK_COLLISION_BODIES)
        self.ik = _g1.DualArmIK(backend,
                                posture_cost=cfg.ik_posture_cost,
                                collision_min_dist=cfg.ik_collision_min_dist,
                                trunk_bodies=trunk,
                                arm_bodies=_g1d.ARM_COLLISION_BODIES)

        # --- which qpos/dof addresses each group owns -----------------------
        def _adr(name):
            j = m.joint(name)
            return int(m.jnt_qposadr[j.id]), int(m.jnt_dofadr[j.id])

        self.slow_q, self.slow_v, self.slow_rate = [], [], []
        self.slow_cost, self.slow_posture, self.slow_exc = [], [], []
        self.slow_name = []

        if "waist" in self.groups:
            # WAIST_JOINT_NAMES is (yaw, bow, None) — slot order, not axis
            # order, and the two G1-D names are backwards from their own axes.
            # Map through the tuple, never by name (`robot/g1d_model.py`).
            for slot, jn in enumerate(kin.WAIST_JOINT_NAMES):
                if jn is None:
                    continue
                cost = (self.w.cost_waist_yaw if slot == 0
                        else self.w.cost_waist_bow)
                rate = (self.w.rate_waist_yaw if slot == 0
                        else self.w.rate_waist_bow)
                self._add(_adr(jn), jn, cost, self.w.posture_waist,
                          rate, self.w.excursion_waist)
        if "column" in self.groups:
            # Two telescoping stages, one command. `rt/cmd_hispeed` carries a
            # single vertical velocity, so the stages are driven together and
            # each carries half the commanded height. Treated as two DOF with
            # half the rate each, which is the same trajectory and keeps the
            # per-DOF machinery uniform.
            for jn in ("LZ_mt_Joint", "LZ_it_Joint"):
                self._add(_adr(jn), jn, self.w.cost_column,
                          self.w.posture_column, self.w.rate_column / 2.0,
                          self.w.excursion_column / 2.0)
        if "base" in self.groups:
            for jn, cost, rate, exc in (
                    ("wbik_base_x", self.w.cost_base_lin,
                     self.w.rate_base_lin, self.w.excursion_base_lin),
                    # y is present in the model and all but banned here: the
                    # G1-D is differential-drive and `Move` ignores vy outright
                    # (Unitree high-level API 1001). Rather than write the
                    # nonholonomic constraint — which is not linear in the
                    # configuration and would take the problem out of QP shape —
                    # it is priced at 50x the forward direction and bounded to a
                    # centimetre, so the solver routes lateral demand through
                    # yaw-then-drive or through the arm, as the real base must.
                    ("wbik_base_y", self.w.cost_base_lin * 50.0,
                     self.w.rate_base_lin, 0.01),
                    ("wbik_base_yaw", self.w.cost_base_yaw,
                     self.w.rate_base_yaw, self.w.excursion_base_yaw)):
                self._add(_adr(jn), jn, cost, self.w.posture_base, rate, exc)

        self.slow_q = np.array(self.slow_q, dtype=np.intp)
        self.slow_v = np.array(self.slow_v, dtype=np.intp)
        self.slow_rate = np.array(self.slow_rate, dtype=np.float64)
        self.slow_cost = np.array(self.slow_cost, dtype=np.float64)
        self.slow_posture = np.array(self.slow_posture, dtype=np.float64)
        self.slow_exc = np.array(self.slow_exc, dtype=np.float64)

        # --- the QP's cost vector and the DOF it may use --------------------
        # **This must be `self.ik._mask`, not a mask of our own.** It is the
        # vector `DualArmIK.solve_tick` multiplies the solved velocity by before
        # integrating, so a mask stored anywhere else is a mask that does
        # nothing — the QP solves for a moving trunk, the integrator throws that
        # component away, and the whole module silently reduces to the arm-only
        # solver it was supposed to extend. It did exactly that for the first
        # run of the (since removed) `wbik_eval`, which reported the waist tier and the off tier
        # agreeing to three decimal places.
        self._mask = np.zeros(m.nv)
        self._mask[backend.all_arm_dofs] = 1.0
        self._mask[self.slow_v] = 1.0
        self.ik._mask = self._mask

        import mink

        # `M` — the "who moves" knob, as a per-DOF L2 penalty on joint
        # velocity. Everything not in an enabled group gets a prohibitive entry
        # as well as its 1e-9 velocity limit; the limit is what actually freezes
        # it, this just means the QP never wants to try.
        n_slow_v = len(self.slow_v)
        damping = np.full(m.nv, 1e4)
        damping[backend.all_arm_dofs] = self.w.cost_arm * self.w.damping_scale
        if n_slow_v:
            damping[self.slow_v] = self.slow_cost * self.w.damping_scale
        self._damping = mink.DampingTask(m, damping)

        # The posture task `DualArmIK` built is a single scalar over every DOF.
        # Replaced with per-DOF ones — and with TWO of them, because pass A and
        # pass C want opposite things from the trunk and a single task cannot be
        # both.
        #
        # Pass A asks "where would the trunk LIKE to be", so the trunk must be
        # nearly free there: its posture entry is the same weak `posture_*` pull
        # that stops a cyclic path from accumulating trunk drift, and nothing more.
        #
        # Pass C asks "track pass B's trajectory unless accuracy says otherwise",
        # so the trunk's posture entry is `track_slow` — 250x the arm's — with
        # the target set to the retimed trajectory each tick.
        #
        # Running pass A under the pass-C task is not a small error: it pins the
        # trunk to the seed, pass A returns the seed, pass B retimes a constant,
        # and every tier reports the arm-only answer (all four tiers inside
        # 0.01 mm of each other).
        def _posture(slow_cost):
            c = np.full(m.nv, self.w.posture_arm)
            c[backend.all_arm_dofs] = self.w.posture_arm
            if n_slow_v:
                c[self.slow_v] = slow_cost
            return mink.PostureTask(m, cost=c)

        # `slow_posture` is per-DOF (the column is pulled home harder than the
        # waist, the base harder still), so pass A keeps the hierarchy even
        # while the trunk is otherwise free to answer.
        self._posture_free = _posture(
            self.slow_posture if n_slow_v else self.w.posture_arm)
        if self.w.nominal_trunk:
            if self.w.bow_floor is None:
                raise ValueError("WbikWeights.nominal_trunk needs bow_floor")
            nominal = np.zeros(len(self.slow_q))
            bow_i = (self.slow_name.index("Yaw_Joint")
                     if "Yaw_Joint" in self.slow_name else None)
            if bow_i is not None:
                nominal[bow_i] = float(self.w.bow_floor)
            self._posture_free = _nominal_trunk_posture_class()(
                m, cost=np.asarray(self._posture_free.cost, np.float64),
                slow_q=self.slow_q, nominal=nominal)
        self._posture_track = _posture(self.w.track_slow)
        self._use_posture(self._posture_free)

        # mink's own per-iterate limit stays, loose, as a shape on the step; the
        # per-TICK bound is `_clamp_slow`, which is the one that is real.
        self._install_velocity_limit(m, backend)

        self._arm_rate = np.full(schema.ARM_DOF, self.w.rate_arm)
        self.base = None                      # set per chunk; see solve_chunk
        #: Absolute lower bound per trunk DOF, in model qpos terms. Only the bow
        #: ever has one; everything else is -inf and bounded by the QP's own
        #: `ConfigurationLimit`. Set per solve, from the MEASURED trunk — see
        #: `WbikWeights.bow_respects_stop`.
        self._floor = np.full(len(self.slow_q), -np.inf)
        self._bow_i = (self.slow_name.index("Yaw_Joint")
                       if "Yaw_Joint" in self.slow_name else None)

    def _set_floor(self, waist_meas):
        """Pin the bow's floor at the trunk's measured rest for this solve."""
        self._floor[:] = -np.inf
        if self._bow_i is None or not self.w.bow_respects_stop:
            return
        if self.w.bow_floor is not None:
            self._floor[self._bow_i] = float(self.w.bow_floor)
            return
        # slot 1 of the (3,) waist is the bow — `kin.WAIST_AXIS_NAMES`. Read
        # through the slot, never the joint name: on the G1-D `Yaw_Joint` IS
        # the bow and `torso_Joint` IS the yaw, which is backwards from both
        # names (`robot/g1d_model.py`).
        rest = float(np.asarray(waist_meas, np.float64)[1])
        if np.isfinite(rest):
            self._floor[self._bow_i] = rest

    # --- construction helpers ----------------------------------------------

    def _add(self, adr, name, cost, posture, rate, exc):
        q, v = adr
        self.slow_q.append(q)
        self.slow_v.append(v)
        self.slow_name.append(name)
        self.slow_cost.append(cost)
        self.slow_posture.append(posture)
        self.slow_rate.append(rate)
        # `None` means "no cap of our own": the joint's own URDF range and the
        # rate limit are the whole bound. Represented as an infinity so every
        # comparison downstream stays arithmetic rather than branching.
        self.slow_exc.append(np.inf if exc is None else exc)

    def add_limit(self, limit) -> None:
        """One more mink limit in the QP (`arm_planner.guard`)."""
        self.ik._limits.append(limit)

    def _use_posture(self, task):
        """Make `task` the posture term the QP sees.

        `DualArmIK` keeps the posture task in two places — the attribute
        `posture`, which callers set targets through, and `_task_list`, which the
        solve actually reads — so swapping one and not the other gives a solver
        whose posture target is written to a task nobody is solving.
        """
        self.ik.posture = task
        self.ik._task_list = [self.ik.tasks["left"], self.ik.tasks["right"],
                              task, self._damping]

    def _install_velocity_limit(self, m, backend):
        """Replace the model-wide freeze with one that lets our groups move.

        `mujoco_model.DualArmIK` freezes every non-arm joint at 1e-9 rad/s, which is what
        pins the trunk for the deployed solver. Each enabled group gets a real
        limit instead; everything else keeps the freeze, so the wheels and the
        gripper jaws stay exactly as immobile as they were.
        """
        import mink

        from elrobot_deploy.robot.mujoco_model import ARM_JOINTS

        arm_names = {n for names in ARM_JOINTS.values() for n in names}
        free = dict(zip(self.slow_name, self.slow_rate))
        vel = {}
        for i in range(m.njnt):
            n = m.joint(i).name
            if n in arm_names:
                vel[n] = 100.0
            elif n in free:
                # Generous per-iterate: the tick bound is applied separately and
                # a tight limit here would only starve the QP of step size.
                vel[n] = max(float(free[n]) * 10.0, 1.0)
            else:
                vel[n] = 1e-9
        # Two limits, kept side by side: one that lets the enabled groups move
        # and one that freezes every non-arm joint the way `DualArmIK` does.
        #
        # **The frozen one is not a convenience.** Execution pins the trunk at
        # the plan, and clamping the trunk AFTER the QP has solved does not
        # produce that: the QP still routes task velocity through the trunk,
        # that component is then thrown away, and the arm is left holding a
        # solution computed on the assumption the trunk would help. Measured —
        # with the post-hoc clamp the arms-only solve came out 15.0% worse than
        # a real one, which is the whole reason the trunk leaves the QP rather
        # than being clipped on the way out.
        # Freezing inside the QP, by giving those joints a 1e-9 rad/s limit, is
        # what makes the two solvers actually comparable.
        frozen = {n: (100.0 if n in arm_names else 1e-9)
                  for n in (m.joint(i).name for i in range(m.njnt))}
        self._vel_free = mink.VelocityLimit(m, vel)
        self._vel_frozen = mink.VelocityLimit(m, frozen)
        for i, lim in enumerate(self.ik._limits):
            if isinstance(lim, mink.VelocityLimit):
                self._vel_slot = i
                self.ik._limits[i] = self._vel_free
                return
        self._vel_slot = len(self.ik._limits)
        self.ik._limits.append(self._vel_free)

    def _use_velocity(self, frozen: bool) -> None:
        """Which velocity limit the QP solves under. See `_install_velocity_limit`."""
        self.ik._limits[self._vel_slot] = (self._vel_frozen if frozen
                                           else self._vel_free)

    # --- geometry -----------------------------------------------------------

    def _targets_world(self, pose_2x7, snap) -> list[np.ndarray]:
        """Base-frame FLANGE poses -> world targets of the DRIVEN frame: the
        TCP when the solver has one (composed with it), else the flange."""
        out = [snap @ kin.pose_to_mat(np.asarray(pose_2x7[h], np.float64))
               for h in range(schema.N_HANDS)]
        return out if self._Xtcp is None else [T @ self._Xtcp for T in out]

    def _position_error(self, targets) -> float:
        """Worst position error of the driven frame (TCP, else flange), m."""
        return max(float(np.linalg.norm(self.backend.ik_xpos(s) - T[:3, 3]))
                   for s, T in zip(SIDES, targets))

    def _orientation_error(self, targets) -> float:
        """Worst orientation error of the driven frame, radians.

        `arccos((tr(R_err) - 1) / 2)` is the rotation angle of
        `R_des @ R_cur.T` — the magnitude of the rotation log map, which is the
        quantity an orientation residual has to be and not a quaternion or Euler
        subtraction. Only the magnitude is needed here; mink's
        `FrameTask` does the vector form inside the QP.
        """
        worst = 0.0
        for s, T in zip(SIDES, targets):
            R = self.backend.ik_pose(s)[:3, :3]
            c = (np.trace(T[:3, :3] @ R.T) - 1.0) * 0.5
            worst = max(worst, float(np.arccos(np.clip(c, -1.0, 1.0))))
        return worst

    def _tcp_error(self, targets) -> float:
        """Worst position error at the TCP, metres — since the QP drives the
        TCP whenever the robot file gives one, this is `_position_error` (and
        at the flange when it gives none). Kept as its own name because
        `ChunkSolution.tcp_err` and the offline tools read it."""
        return self._position_error(targets)

    def _place(self, arm_q14, waist_q3, slow_vals=None):
        """`kin.place`, plus this module's own trunk DOF.

        `kin.place` zeroes the whole qpos and writes the waist and the arms; it
        is the same function P1's FK calls, and using it rather than an
        equivalent local copy is what stops the pose the policy observes and the
        pose the IK solves for drifting into different geometries. The column
        and the base are written after it, because `place` would zero them.
        """
        kin.place(self.backend, arm_q14, waist_q3, self._jmap)
        if slow_vals is not None:
            self.backend.data.qpos[self.slow_q] = slow_vals
        self._mujoco.mj_forward(self.backend.model, self.backend.data)

    def _clamp_slow(self, start_vals, chunk_origin=None):
        """Hold the trunk inside one tick's travel, and the chunk's excursion.

        Applied after every QP iterate — see the module docstring. Two bounds,
        and they answer different questions: `slow_rate * dt` is what the
        hardware can do in 33 ms, `slow_exc` is how far this chunk is allowed to
        wander from where it began, which is a statement about the head camera's
        training distribution rather than about the motors.
        """
        q = self.ik.config.q
        lo = start_vals - self.slow_rate * self.dt
        hi = start_vals + self.slow_rate * self.dt
        if chunk_origin is not None:
            lo = np.maximum(lo, chunk_origin - self.slow_exc)
            hi = np.minimum(hi, chunk_origin + self.slow_exc)
        lo = np.maximum(lo, self._floor)
        hi = np.maximum(hi, lo)          # a floor above `hi` collapses, not inverts
        v = np.clip(q[self.slow_q], lo, hi)
        if not np.array_equal(v, q[self.slow_q]):
            q[self.slow_q] = v
            self.ik.config.update(q)

    # --- one tick -----------------------------------------------------------

    def _solve_tick(self, targets, budget, start_slow, chunk_origin=None,
                    slow_target=None):
        """Iterate the QP toward one tick's targets. Returns `(err, ok, iters)`.

        The loop is `arm_qp.DualArmSolver.solve`'s, with two additions: the trunk
        clamp after each iterate, and an optional posture pull toward pass B's
        trajectory. `ok=False` still means one thing only — the wall clock ran
        out — and is deliberately not a quality gate, for the reason `arm_qp.py`
        gives at length: this QP has a hard error floor set by the posture task
        and gating on it would refuse every realistic tick.
        """
        if slow_target is not None:
            q = self.ik.config.q.copy()
            q[self.slow_q] = slow_target
            self.ik.posture.set_target(q)

        expires = now() + budget
        err = self._position_error(targets)
        iters = 0
        while err > self.early_stop:
            if now() >= expires:
                return err, False, iters
            if iters >= self.max_iters:
                break
            prev = self.ik.config.q.copy()
            self.ik.solve_tick(targets[0], targets[1], iters=1, dt=self.dt)
            iters += 1
            self._clamp_slow(start_slow, chunk_origin)
            if np.array_equal(self.ik.config.q, prev):
                break
            new = self._position_error(targets)
            if (self.rel is not None and iters >= self.min_iters
                    and err - new < max(self.rel * err, self.abs_tol)):
                err = new
                break                       # the convergence stop (`arm_qp.py`)
            err = new
        return err, True, iters

    def solve(self, pose_2x7, waist_3, seed_q14, budget: float | None = None):
        """One tick at a GIVEN trunk. `arm_qp.DualArmSolver.solve`'s exact
        signature, return, and behaviour.

        **The trunk is pinned at `waist_3` and does not move.** This is the
        execution half of the plan-and-execute split: `plan_chunk` decides where
        the trunk goes, and every tick after that is an ordinary arms-only solve
        in the frame the trunk will be in. So `waist_3` is an input here, never
        a seed.

        Leaving the trunk free here — the QP may move it, only the arm is
        returned — is a coherent design (a reactive per-tick whole-body
        solve) and it is NOT the one P6 runs: P6 commands the PLANNED trunk, so
        an arm solved for some other trunk is an arm solved in a frame the robot
        will not be in. Measured on one recorded rollout:
        5.24 mm mean at the TCP against
        0.85 mm for the same chunks with the trunk pinned. A free trunk is
        buying a better QP answer to a question nobody was going to execute.

        There is no snapshot frame on this path and there cannot be: a single
        tick has no chunk to snapshot at the top of. So it uses the live base
        frame, exactly as the deployed solver does, and the column and the base
        are inert here — which is the honest behaviour, since a solver that
        moved the base to chase a target that moves with the base would drive
        forever without closing a millimetre.
        """
        budget = self.dt if budget is None else float(budget)
        self._use_posture(self._posture_free)
        self._set_floor(waist_3)
        self._place(seed_q14, waist_3)
        base = self.backend.base_pose()
        targets = self._targets_world(pose_2x7, base)
        self.ik.config.update(self.backend.data.qpos.copy())
        self.ik.posture.set_target_from_configuration(self.ik.config)
        start = self.ik.config.q[self.slow_q].copy()
        # Out of the QP and out of the integrator's mask, not clipped after the
        # fact: a post-hoc clamp still lets the QP route task velocity through
        # the trunk and then throws that component away, leaving the arm holding
        # a solution computed on the assumption the trunk would help. See
        # `_install_velocity_limit`.
        self._use_velocity(True)
        saved = self.ik._mask
        if len(self.slow_v):
            m = saved.copy()
            m[self.slow_v] = 0.0
            self.ik._mask = m
        try:
            err, ok, _ = self._solve_tick(targets, budget, start)
        finally:
            self.ik._mask = saved
            self._use_velocity(False)
        return self._arm(), err, ok

    # --- a whole chunk ------------------------------------------------------

    def plan_chunk(self, poses, waist_meas, seed_q14, *,
                   budget: float | None = None, slow_start=None):
        """`(K, 2, 7)` flange poses -> the trunk's `(K, 3)` waist commands.

        **Passes A and B only.** This is the whole of what the whole-body solver
        contributes to a deployed run: a trunk trajectory for the chunk. Once it
        exists, every tick is an ordinary arms-only IK solve with the trunk
        placed at the planned angle — which is exactly what `arm_qp.DualArmSolver`
        does with the MEASURED trunk, so P6 streams the chunk through
        the per-tick solve and this module is never called again until
        the next chunk lands.

        That split is what lets the chunk stream. A and B need the whole chunk
        (B's reachability tube is a backward sweep, so tick 0's bound depends on
        tick 24's demand) and together cost ~80 ms; everything after them is
        per-tick and forward-only, so the first target is published ~85 ms after
        the chunk arrives instead of waiting ~300 ms for a whole-chunk solve.

        Returned in `kin.WAIST_JOINT_NAMES` slot order, ready to hand to
        `solve()` and to write into the target ring. Slot 2 is always
        `waist_meas`'s — there is no motor behind it.
        """
        poses = np.asarray(poses, dtype=np.float64)
        K = len(poses)
        budget = (self.dt * K if budget is None else float(budget))
        traj, _ = self._plan(poses, waist_meas, seed_q14, budget, slow_start)
        out = np.tile(np.asarray(waist_meas, np.float64), (K, 1))
        if len(self.slow_q):
            for k in range(K):
                out[k] = self._waist_from_slow(traj[k], waist_meas)
        return out

    def _plan(self, poses, waist_meas, seed_q14, budget, slow_start):
        """Passes A and B. -> `(traj, origin)` in slow-joint coordinates."""
        K = len(poses)
        # The snapshot frame, captured ONCE, at the trunk the chunk starts from.
        # See the module docstring: this is what makes the column and the base
        # worth anything.
        self._set_floor(waist_meas)
        self._place(seed_q14, waist_meas, slow_start)
        snap = self.backend.base_pose().copy()
        self.base = snap
        origin = self.backend.data.qpos[self.slow_q].copy()
        targets = [self._targets_world(poses[k], snap) for k in range(K)]
        if not len(self.slow_q):
            return np.zeros((K, 0)), origin
        ideal = self._pass_a(targets, seed_q14, waist_meas, origin, budget)
        return self._pass_b(ideal, origin, K), origin

    def _waist_from_slow(self, slow_row, fallback) -> np.ndarray:
        """One row of pass B's trajectory -> a `(3,)` commanded waist.

        Mapped through `kin.WAIST_JOINT_NAMES`, whose index IS the slot — never
        by axis name, because the G1-D's two waist joints are named backwards
        from their own axes (`torso_Joint` yaws, `Yaw_Joint` bows). Slot 2 keeps
        the fallback: there is no motor behind it.
        """
        out = np.asarray(fallback, np.float64).copy()
        if "waist" in self.groups:
            for i, name in enumerate(self.slow_name):
                if name in kin.WAIST_JOINT_NAMES:
                    out[kin.WAIST_JOINT_NAMES.index(name)] = slow_row[i]
        return out

    def solve_chunk(self, poses, waist_meas, seed_q14, *,
                    budget: float | None = None,
                    slow_start=None,
                    first_span: float | None = None) -> ChunkSolution:
        """`(K, 2, 7)` base-frame flange poses -> `K` whole-body configurations.

        Plan, then execute: `plan_chunk`'s trunk trajectory, then one arms-only
        solve per tick with the trunk pinned at it. Kept as one call for the
        replay tools, the overlay renderer and the dashboard, all of which want
        a whole chunk at once; P6 does the same two things separately, so that
        the per-tick half can stream.

        `waist_meas` is the MEASURED trunk at the top of the chunk, for the same
        reason P6 reads it fresh every tick: the trunk rests 0.043 rad forward of
        its commanded zero and no torque moves it, so a solve against the
        commanded trunk places the arm in a frame the robot is not in. Here it
        seeds the plan — the trunk may move from there, but it starts at the
        truth.

        `first_span` is how much time the arm has to get from `seed_q14` to the
        chunk's FIRST tick, in seconds. It is not always one tick: P6 seeds from
        the written target with the largest k below the chunk's anchor, and a
        new chunk snaps the anchor to `k_exec + 2`, so the predecessor can be
        two or more ticks back. The per-tick path computes exactly this as
        `(k - seed.k) * dt` and clamps against it; passing it in keeps the two
        paths clamping the same way, instead of the chunk path spuriously
        refusing a step the arm has two ticks to make.
        """
        poses = np.asarray(poses, dtype=np.float64)
        K = len(poses)
        budget = (self.dt * K if budget is None else float(budget))
        t0 = now()
        timing = {}

        traj, origin = self._plan(poses, waist_meas, seed_q14, budget,
                                  slow_start)
        timing["plan"] = now() - t0

        # `_plan` left the backend placed and `self.base` snapshotted, so the
        # targets are rebuilt against that same frame rather than a fresh one.
        targets = [self._targets_world(poses[k], self.base) for k in range(K)]

        t2 = now()
        out = self._pass_c(targets, seed_q14, waist_meas, origin, traj,
                           budget - (now() - t0), first_span=first_span)
        timing["execute"] = now() - t2

        out.timing = timing
        out.elapsed = now() - t0
        return out

    def _pass_a(self, targets, seed_q14, waist_meas, origin, budget):
        """Where the trunk would like to be at each tick, rates ignored.

        Each tick solved independently from the SAME seed — the chunk's starting
        posture — rather than chained. Chaining would make tick 20's answer a
        function of the path taken to it, which is precisely the rate-limited
        question pass B exists to ask; asking it here too would bake one
        arbitrary answer in before the smoothing that is supposed to choose it.

        The excursion cap still applies. Without it this pass will happily
        propose driving the base a metre to save 3 mm at the flange, and pass B
        would then spend its whole budget refusing to.

        `WbikWeights.pass_a_stride` solves a subset of the ticks and fills the
        rest by linear interpolation. Each knot that IS solved is solved exactly
        as it would be at stride 1 — the same seed, the same per-tick budget —
        so the stride changes which ticks are asked, never how one is answered.
        """
        K = len(targets)
        ideal = np.zeros((K, len(self.slow_q)))
        # Both ends are always knots: interpolation between them is filling in,
        # extrapolation past them would be inventing.
        stride = max(1, int(self.w.pass_a_stride))
        knots = sorted({*range(0, K, stride), K - 1})
        big = self.slow_rate.copy()
        self.slow_rate = self.slow_rate * 1e3          # rates off for this pass
        self._use_posture(self._posture_free)
        try:
            for k in knots:
                self._place(seed_q14, waist_meas, origin)
                self.ik.config.update(self.backend.data.qpos.copy())
                self.ik.posture.set_target_from_configuration(self.ik.config)
                start = self.ik.config.q[self.slow_q].copy()
                # Per-KNOT budget stays `budget / (3K)`, not `budget / (3*len(
                # knots))`. A knot solved at stride 4 must be the same answer it
                # would have been at stride 1, or the sweep measures two changes
                # at once; the saving is meant to come from asking fewer ticks,
                # not from answering each one worse.
                self._solve_tick(targets[k], budget / (3.0 * K), start, origin)
                ideal[k] = self.ik.config.q[self.slow_q]
        finally:
            self.slow_rate = big
        if len(knots) < K:
            grid = np.arange(K, dtype=np.float64)
            kk = np.asarray(knots, dtype=np.float64)
            for j in range(ideal.shape[1]):
                ideal[:, j] = np.interp(grid, kk, ideal[knots, j])
        return ideal

    def _knot_weights(self, ideal, origin):
        """How much each knot's pass-A answer should count, per DOF.

        Pass A puts the ideal at the origin for every tick the trunk made no
        difference to. Those knots are not asking for the trunk to be at the
        origin — they are not asking for anything — but a uniform tracking
        weight lets them vote as though they were, and on a 25-tick chunk the
        indifferent majority outvotes the few ticks that need the trunk. The
        trajectory then leaves late and never arrives, which is precisely the
        reactive behaviour the horizon exists to avoid.

        So a knot's weight is how far it wanted the trunk to move, normalised
        per DOF. `_KNOT_FLOOR` keeps a chunk where nothing is demanded from
        having a degenerate all-zero objective; it is small enough that one real
        demand dominates fifty indifferent knots.
        """
        d = np.abs(ideal - origin[None, :])
        peak = d.max(axis=0)
        peak[peak <= 0] = 1.0
        return _KNOT_FLOOR + (1.0 - _KNOT_FLOOR) * (d / peak[None, :])

    def _pass_b(self, ideal, origin, K):
        """Retime pass A into something the hardware can actually execute."""
        traj = np.zeros((K, len(self.slow_q)))
        w_knot = self._knot_weights(ideal, origin)
        for j in range(len(self.slow_q)):
            # The excursion box is symmetric about the origin; the bow's floor
            # cuts the bottom off it. Expressed as an offset because
            # `_banded_retime` works in deltas from `origin`.
            lo = -float(self.slow_exc[j])
            if np.isfinite(self._floor[j]):
                lo = max(lo, float(self._floor[j] - origin[j]))
            traj[:, j] = _banded_retime(
                ideal[:, j], float(origin[j]),
                float(self.slow_rate[j]),
                float(self.slow_rate[j]) * self.w.accel_frac / self.dt,
                self.dt, min(lo, 0.0), float(self.slow_exc[j]),
                w_knot[:, j], self.w.smooth_vel, self.w.smooth_acc)
        return traj

    def _pass_c(self, targets, seed_q14, waist_meas, origin, traj, budget,
                first_span=None):
        """Solve the arm along the planned trunk trajectory, chained.

        **The trunk is PINNED at the plan, not steered toward it.** Each tick
        places the trunk at `traj[k]`, takes it out of the QP and out of the
        integrator's mask, and solves the arm — which is exactly what
        `arm_qp.DualArmSolver.solve` does when handed a waist, and is why P6 can
        stream this loop through the per-tick solve instead of calling
        back into this module every tick.

        Keeping the trunk in the QP with a soft pull toward `traj` would let
        the solve nudge it off the plan where that helped the arm. That
        freedom is measured and it is not worth anything: over 351 chunks,
        pinning cost 1.5% of the mean residual (1.448 mm against 1.427) and
        IMPROVED the worst tick (38.002 mm against 38.952). What it buys is that
        the published trunk is the planned trunk, exactly — so a tick can be
        solved and published the moment its turn comes, with no later pass
        entitled to move the trunk underneath it.
        """
        K = len(targets)
        arm = np.zeros((K, schema.ARM_DOF))
        waist = np.zeros((K, schema.WAIST_DOF))
        err = np.zeros(K)
        tcp = np.zeros(K)
        rot = np.zeros(K)
        col = np.zeros(K)
        base = np.zeros((K, 3))
        ok = True

        seed = np.asarray(seed_q14, np.float64).copy()
        slow = origin.copy()
        per_tick = max(budget / max(K, 1), 1e-4)
        self._use_posture(self._posture_free)
        # An ARMS-ONLY solve, not a whole-body solve with the answer clipped:
        # the trunk comes out of the QP itself and out of the integrator's mask,
        # so what comes back is what `arm_qp.DualArmSolver` would have produced with
        # the same waist handed to it. See `_install_velocity_limit`.
        self._use_velocity(True)
        saved_mask = self.ik._mask
        if len(self.slow_v):
            m = saved_mask.copy()
            m[self.slow_v] = 0.0
            self.ik._mask = m

        col_idx = [i for i, n in enumerate(self.slow_name)
                   if n in ("LZ_mt_Joint", "LZ_it_Joint")]
        base_idx = {n: i for i, n in enumerate(self.slow_name)
                    if n.startswith("wbik_base_")}

        for k in range(K):
            # The plan, not wherever the previous tick's QP left the trunk. With
            # the trunk masked out these are the same thing; taking it from the
            # plan says so, and keeps this loop's output equal to what P6
            # produces tick by tick from the same trajectory.
            if len(self.slow_q):
                slow = np.asarray(traj[k], np.float64).copy()
            self._place(seed, waist_meas, slow)
            self.ik.config.update(self.backend.data.qpos.copy())
            self.ik.posture.set_target_from_configuration(self.ik.config)
            start = self.ik.config.q[self.slow_q].copy()
            e, tick_ok, _ = self._solve_tick(
                targets[k], per_tick, start, origin)
            ok = ok and tick_ok

            # The arm's own per-tick clamp, the one P6 has always applied. Here
            # rather than in P6 because with the trunk moving, a clamp applied
            # after the fact would leave the arm somewhere the trunk solution
            # did not assume.
            q_arm = self._arm()
            span = self.dt if (k or first_span is None) else float(first_span)
            lim = self._arm_rate * span
            arm[k] = np.clip(q_arm, seed - lim, seed + lim)
            seed = arm[k]

            slow = self.ik.config.q[self.slow_q].copy()
            waist[k] = self._waist_of(self.ik.config.q, waist_meas)
            err[k] = e
            tcp[k] = self._tcp_error(targets[k])
            rot[k] = self._orientation_error(targets[k])
            if col_idx:
                col[k] = float(np.sum(slow[col_idx] - origin[col_idx]))
            for j, n in enumerate(("wbik_base_x", "wbik_base_y",
                                   "wbik_base_yaw")):
                if n in base_idx:
                    base[k, j] = float(slow[base_idx[n]] - origin[base_idx[n]])

        self.ik._mask = saved_mask
        self._use_velocity(False)
        return ChunkSolution(arm=arm, waist=waist, err=err, tcp_err=tcp,
                             rot_err=rot, column=col, base=base, ok=ok,
                             elapsed=0.0)

    def _waist_of(self, q, fallback):
        """The (3,) commanded waist this configuration implies.

        Slot 2 stays whatever the caller passed — there is no motor behind it
        and `kin.waist_limits` bounds it to exactly zero, so writing a solved
        value there would be publishing an axis the robot does not have.
        """
        out = np.asarray(fallback, np.float64).copy()
        if "waist" in self.groups:
            out[self._jmap.waist_cols] = q[self._jmap.waist]
        return out

    def _arm(self) -> np.ndarray:
        return self.ik.config.q[self._jmap.arm].copy()

    # --- reporting ----------------------------------------------------------

    def feasible(self, sol: ChunkSolution) -> dict:
        """Does this solution stay inside the rates the hardware has?

        Reported rather than asserted. A violation here is a bug in the clamp,
        not a condition a caller should handle at runtime — but it is exactly
        the thing that is invisible in a residual number, and the whole claim of
        this module is that the accuracy was not bought with motion the machine
        cannot perform.
        """
        out = {}
        w = self.w
        for name, series, rate in (
                ("arm", sol.arm, w.rate_arm),
                ("waist_yaw", sol.waist[:, 0:1], w.rate_waist_yaw),
                ("waist_bow", sol.waist[:, 1:2], w.rate_waist_bow),
                ("column", sol.column[:, None], w.rate_column),
                ("base_lin", sol.base[:, :2], w.rate_base_lin),
                ("base_yaw", sol.base[:, 2:3], w.rate_base_yaw)):
            if series.size == 0:
                continue
            d = np.abs(np.diff(series, axis=0)) / self.dt
            peak = float(d.max()) if d.size else 0.0
            out[name] = {"peak_rate": peak, "limit": float(rate),
                         "ok": peak <= rate * 1.02}
        return out


def waist_qp(cfg, *, urdf=None, qdot_max=None, tcp=None):
    """The `waist_qp` factory (`runtime/ik/__init__.py::SOLVERS`): the waist tier (yaw + bow). P6
    plans each chunk's trunk with passes A and B, then solves the arm per tick
    on the planned trunk."""
    return WholeBodySolver(cfg, urdf=urdf, groups=TIERS["waist"], tcp=tcp)


waist_qp.mode = "waist"


__all__ = ["GROUPS", "TIERS", "ChunkSolution", "WbikWeights",
           "WholeBodySolver", "waist_qp"]

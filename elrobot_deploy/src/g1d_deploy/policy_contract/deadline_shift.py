"""The deadline shift — re-timing a chunk onto the deadlines it will actually meet.

Applied by P5 to every chunk after validation and before sync truncation
(`runtime/processes/p5_policy.py`); pure numpy, no I/O. Design:
`docs/deploying_a_policy/policy_contract.md`, "What happens to the actions".

It lives in the infra because nothing in it is about a model: it is a
correction for the anchor rule (`anchor.py`), which is the infra's, and every
policy whose action `j` means "`(j+1)·dt` after the image" needs it.

WHY IT EXISTS

The contract times action `j` at exactly `(j+1)·dt` after the anchor image's
capture time (for a learned policy, that is how training labelled it).
Deployment does not deliver that. `acquire_anchor` takes the
anchor camera's frame NEAREST `t_trigger`, either side, so with
`e = t_anchor - t_trigger` (positive when the anchor waited for a fresher frame,
bounded by half a camera period):

    action j is due at  t_trigger + (j+1)·dt = t_anchor - e + (j+1)·dt

i.e. every action is due `e` EARLIER than the chunk claims — a uniform shift of
`-e/dt` slots. Worked example, dt = 33.33 ms: `t_trigger` 10 ms, `t_anchor`
25 ms, so `e = +15`. The policy's first action reads as 25 + 33.33 = 58.33 ms, but
its deadline is 10 + 33.33 = 43.33 ms — 55% of the way from the anchor state to
the first action, which is `shift_slots = -0.45`.

Measured over a set of recorded rollouts without it: mean first-action
offset 28.52 ms against a 33.33 ms label, sd 10.42 — a 4.81 ms systematic bias
on top of the spread.

WHAT IS INTERPOLATED

Row 0 of the interpolation grid is the MEASURED state at the anchor — flange
pose, joint angles, hand — so a shift toward the anchor interpolates against
ground truth rather than against a second prediction. Rows 1..n are the policy's
actions. Positions and joint angles are linear; rotations are slerped.

`e < 0` shifts the reads LATER and the last one falls past the end of the chunk,
so the tail extrapolates along the final segment. In sync that is free: the
infra truncates to `P` afterwards, and `P < H` means no extrapolated action is
ever commanded — which is exactly why the shift happens BEFORE the truncation
and why a sync policy now returns its whole chunk. In async the tail can be
reached; a half-tick along a segment the policy itself produced is the mildest
thing available, but it is genuinely extrapolation.

THE INTERPOLATION POINT (eef)

Interpolating a rigid motion at the flange and at a tool point further out
disagree by the chord-versus-arc deficit of the rotation times the offset —
for a TCP 18.5 cm along the flange's +X, ~0.06 mm at 0.05 rad per tick and
~0.9 mm at 0.2. The contract speaks flange, and a TCP is each policy's own
definition, so the point is a policy attribute: `Policy.deadline_shift_point`,
a flange-frame offset in metres, default the flange itself. A policy whose
actions were labelled (or time-jittered in training) at a TCP sets it to that
TCP, so both sides interpolate at the same point.

THE BOUND

`|shift_slots|` is clipped to `MAX_SHIFT_SLOTS`. In normal operation it cannot
fire: the nearest-frame rule bounds `|e|` by half a camera period, a fraction of
a tick. It fires only when the anchor's wait cap has — a camera that stopped, so
the frame in hand is arbitrarily old — and there the honest correction would be
an arbitrarily long extrapolation. One tick is the most this will invent;
`Anchor.max_image_latency()`, printed per chunk, is what reports the camera.
"""

from __future__ import annotations

import numpy as np

from g1d_deploy.policy_contract.actions import EefActions, JointActions

#: See "THE BOUND" above.
MAX_SHIFT_SLOTS = 1.0


def deadline_shift_slots(t_anchor: float, t_trigger: float, dt: float) -> float:
    """The uniform shift, in ticks. `-(t_anchor - t_trigger)/dt`, clipped.

    Its own function because this is where the sign lives, and a flipped sign
    here is a half-tick lead where a half-tick lag belongs — perfectly
    plausible-looking on a plot.
    """
    s = -(float(t_anchor) - float(t_trigger)) / float(dt)
    return float(np.clip(s, -MAX_SHIFT_SLOTS, MAX_SHIFT_SLOTS))


def slerp(qa, qb, u):
    """Sign-corrected spherical interpolation of quaternion arrays.

    Component order is irrelevant —
    this rotates the 4-vector on its own sphere. Falls back to normalised lerp
    when the pair is nearly parallel, where slerp is ill-conditioned. `u`
    outside [0, 1] EXTRAPOLATES along the same geodesic, which the tail of a
    chunk shifted later in time needs.
    """
    qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True)
    qb = qb / np.linalg.norm(qb, axis=-1, keepdims=True)
    dot = np.sum(qa * qb, axis=-1)
    qb = np.where((dot < 0)[..., None], -qb, qb)
    dot = np.abs(dot)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t = np.sin(theta)
    near = sin_t < 1e-6
    safe = np.where(near, 1.0, sin_t)
    wa = np.where(near, 1.0 - u, np.sin((1.0 - u) * theta) / safe)
    wb = np.where(near, u, np.sin(u * theta) / safe)
    q = wa[..., None] * qa + wb[..., None] * qb
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _bracket(n: int, shift_slots: float):
    """`(lo, hi, frac)` into an `(n+1)`-row grid whose row 0 is the anchor.

    Clamp the BRACKET, not the fraction. A fraction outside [0, 1] is exactly
    what makes an end read extrapolate along its own segment instead of freezing
    on the last row, and freezing is the failure mode this exists to avoid.
    """
    f = np.arange(1, n + 1, dtype=np.float64) + float(shift_slots)
    lo = np.clip(np.floor(f).astype(np.int64), 0, n - 1)
    frac = f - lo
    return lo, lo + 1, frac


def _lerp(grid, lo, hi, frac):
    w = frac.reshape((-1,) + (1,) * (grid.ndim - 1))
    return (1.0 - w) * grid[lo] + w * grid[hi]


def _rotate(q, v):
    """Rotate `v` (..., 3) by unit wxyz quaternions `q` (..., 4)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _shift_pose(anchor_pose, poses, lo, hi, frac, point):
    """`(n, 2, 7)` flange poses, re-timed; interpolated AT `point` (flange frame).

    The rotation is the same at the flange and at any rigidly attached point, so
    only the translation moves: `p_point = p_flange + R·point`, interpolate that,
    then map back with the INTERPOLATED rotation.
    """
    grid = np.concatenate([np.asarray(anchor_pose, np.float64)[None], poses])
    quat = grid[..., 3:]
    at = grid[..., :3] + _rotate(quat, np.broadcast_to(point, grid[..., :3].shape))
    q = slerp(quat[lo], quat[hi], frac[:, None])
    xyz = _lerp(at, lo, hi, frac) - _rotate(q, np.broadcast_to(point, at[lo].shape))
    return np.concatenate([xyz, q], axis=-1)


def shift_to_deadline(actions, measured, shift_slots: float, *,
                      point=(0.0, 0.0, 0.0)):
    """`actions` re-timed by `shift_slots`, the same type back. Pure.

    `measured` is the anchor row, a mapping with:
      * `flange`  (2, 7)  measured flange poses                 (eef)
      * `arm_q`   (2, 7)  measured arm joints                   (joint)
      * `waist_q` (3,)    measured waist, slot order            (joint)
      * `hand`    (2,)    measured hand, the end effector's own units

    A zero shift returns the input unchanged — bit-identical, not merely close —
    so the opt-out and the ideal anchor are the same path as no shift at all.
    """
    if not shift_slots:
        return actions
    n = actions.n
    lo, hi, frac = _bracket(n, shift_slots)

    def grid(row0, rows):
        return _lerp(np.concatenate([np.asarray(row0, np.float64)[None], rows]),
                     lo, hi, frac)

    hand = grid(measured["hand"], actions.hand)

    if isinstance(actions, EefActions):
        pose = _shift_pose(measured["flange"], actions.pose, lo, hi, frac,
                           np.asarray(point, np.float64))
        return EefActions(pose=pose, hand=hand)

    assert isinstance(actions, JointActions)
    arm_q = grid(measured["arm_q"], actions.arm_q)
    waist_q = grid(measured["waist_q"], actions.waist_q)
    return JointActions(arm_q=arm_q, waist_q=waist_q, hand=hand)


__all__ = ["MAX_SHIFT_SLOTS", "deadline_shift_slots", "shift_to_deadline", "slerp"]

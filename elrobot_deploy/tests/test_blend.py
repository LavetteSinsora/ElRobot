"""The blend layer.

The failure this file exists for is the double cover: `-q` and `q` are the same
rotation, so a blend that averages them produces the identity and the arm snaps
to a completely different orientation for one tick. Nothing about a single
chunk, a single contributor, or a sync run can reproduce it — it needs two
overlapping chunks whose quaternions happen to have opposite signs, which is a
property of the policy's output, not of the framework.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from g1d_deploy.runtime.execution import chunk_blend as blend
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.shared_memory.rings import Contribution


def _contrib(chunk_id, quat, *, pos=0.0, hand=0.0, waist=0.0, row=-1):
    pose = np.zeros((schema.N_HANDS, schema.POSE_DIM))
    pose[:, :3] = pos
    pose[:, 3:] = np.asarray(quat, dtype=np.float64)
    return Contribution(chunk_id, pose,
                        np.full((schema.N_HANDS, schema.HAND_DIM), hand),
                        np.full((schema.WAIST_DOF,), waist), row=row)


IDENT = (1.0, 0.0, 0.0, 0.0)
# 90 degrees about z, and the same rotation written on the other sheet.
ROT_Z = (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
ROT_Z_NEG = tuple(-x for x in ROT_Z)


# --- Hemisphere alignment ---------------------------------------------------


def test_alignment_flips_into_the_newest_hemisphere():
    old, new = _contrib(1, ROT_Z_NEG), _contrib(2, ROT_Z)
    aligned = blend.hemisphere_align([old, new])
    assert np.allclose(aligned[0].pose[:, 3:], ROT_Z)
    assert np.allclose(aligned[1].pose[:, 3:], ROT_Z)


def test_alignment_reference_is_the_newest_not_the_first():
    """Whatever the method, the result stays on the sheet the most recent
    inference chose — the newest contributor is the reference."""
    aligned = blend.hemisphere_align([_contrib(1, ROT_Z), _contrib(2, ROT_Z_NEG)])
    assert np.allclose(aligned[0].pose[:, 3:], ROT_Z_NEG)
    assert np.allclose(aligned[1].pose[:, 3:], ROT_Z_NEG)


def test_alignment_is_per_arm():
    """Two independent rotations: a left flange needing a flip says nothing
    about the right."""
    old = _contrib(1, IDENT)
    old.pose[0, 3:] = ROT_Z_NEG          # left on the far sheet
    old.pose[1, 3:] = ROT_Z              # right already aligned
    new = _contrib(2, ROT_Z)
    aligned = blend.hemisphere_align([old, new])
    assert np.allclose(aligned[0].pose[0, 3:], ROT_Z)
    assert np.allclose(aligned[0].pose[1, 3:], ROT_Z)


def test_alignment_does_not_mutate_its_input():
    old = _contrib(1, ROT_Z_NEG)
    blend.hemisphere_align([old, _contrib(2, ROT_Z)])
    assert np.allclose(old.pose[:, 3:], ROT_Z_NEG)


def test_alignment_of_one_contributor_is_a_no_op():
    """Sync always yields exactly one contributor, so this is the whole of
    blending there."""
    only = _contrib(7, ROT_Z_NEG)
    assert blend.hemisphere_align([only])[0] is only


# --- The thing alignment exists to prevent ----------------------------------


def test_q_and_minus_q_are_the_same_rotation_and_average_to_nothing():
    """The arithmetic that hemisphere alignment exists for: the mean of `q` and `-q` is not a
    rotation at all, it is the zero quaternion. Renormalizing it produces
    whatever the rounding error happened to be."""
    assert np.allclose(0.5 * np.array(ROT_Z) + 0.5 * np.array(ROT_Z_NEG), 0.0)
    got = blend.apply(blend.temporal_ensemble, [_contrib(1, ROT_Z_NEG), _contrib(2, ROT_Z)])
    assert np.allclose(got.pose[:, 3:], ROT_Z)


def test_alignment_changes_the_blended_rotation():
    """The observable bug, on contributors that are not exact antipodes: an
    unaligned average lands on a rotation neither chunk asked for, and it is
    the framework's `apply()` — not the method — that prevents it."""
    pair = [_contrib(1, ROT_Z_NEG), _contrib(2, IDENT)]
    naive = blend.temporal_ensemble(pair)             # method called directly
    aligned = blend.apply(blend.temporal_ensemble, pair)   # the framework's path
    assert not np.allclose(naive.pose[:, 3:], aligned.pose[:, 3:])
    # Aligned, the answer is between the two rotations: a positive-z turn of
    # less than the 90 degrees the older chunk asked for.
    q = aligned.pose[0, 3:]
    assert q[0] > 0 and q[3] > 0 and q[3] < np.sqrt(0.5)
    # Unaligned, it turns the other way.
    assert naive.pose[0, 6] < 0


def test_the_blended_quaternion_stays_a_unit_quaternion():
    got = blend.apply(blend.temporal_ensemble, [_contrib(1, IDENT), _contrib(2, ROT_Z)])
    assert np.allclose(np.linalg.norm(got.pose[:, 3:], axis=1), 1.0)


# --- The methods ------------------------------------------------------------


def test_newest_takes_the_highest_chunk_id():
    got = blend.apply(blend.newest, [_contrib(4, IDENT, pos=1.0, hand=1.0, waist=1.0),
                                     _contrib(9, IDENT, pos=2.0, hand=2.0, waist=2.0)])
    assert np.allclose(got.pose[:, :3], 2.0)
    assert np.allclose(got.hand, 2.0)
    assert np.allclose(got.waist, 2.0)


def test_newest_copies_so_the_ring_can_be_rewritten_under_it():
    c = _contrib(1, IDENT, pos=1.0)
    got = blend.newest([c])
    c.pose[:] = 99.0
    assert np.allclose(got.pose[:, :3], 1.0)


def test_hand_and_waist_carry_the_same_weights_as_the_pose():
    """Same contributors, same weights, same commit point, so a
    crossfade never executes one chunk's grip against another's pose."""
    got = blend.apply(blend.temporal_ensemble,
                      [_contrib(1, IDENT, pos=0.0, hand=0.0, waist=0.0, row=12),
                       _contrib(2, IDENT, pos=3.0, hand=3.0, waist=3.0, row=2)])
    w_old = np.exp(0.01 * 10) / (np.exp(0.01 * 10) + 1.0)
    expected = (1 - w_old) * 3.0
    assert np.allclose(got.pose[:, :3], expected)
    assert np.allclose(got.hand, expected)
    assert np.allclose(got.waist, expected)


def test_temporal_ensembling_is_acts_weighting_by_age():
    """w_i = exp(-m i) with i = 0 the oldest prediction, ages in ticks: the
    oldest (largest row) weighs most; the weights depend only on the row gaps."""
    cs = [_contrib(1, IDENT, pos=0.0, row=20), _contrib(2, IDENT, pos=0.0, row=10),
          _contrib(3, IDENT, pos=6.0, row=0)]
    w = np.exp(0.5 * np.array([20.0, 10.0, 0.0]))
    got = blend.apply(lambda c: blend.temporal_ensemble(c, m=0.5), cs)
    assert np.allclose(got.pose[:, :3], 6.0 * w[2] / w.sum())
    plain = blend.apply(lambda c: blend.temporal_ensemble(c, m=0.0), cs)
    assert np.allclose(plain.pose[:, :3], 2.0)          # m = 0: a plain mean


def test_the_config_sets_the_ensembles_m():
    from g1d_deploy.config_loading.resolved import DeployConfig
    fn = blend.get("temporal_ensemble", types.SimpleNamespace(temporal_ensemble_m=0.5))
    cs = [_contrib(1, IDENT, pos=0.0, row=10), _contrib(2, IDENT, pos=1.0, row=0)]
    assert np.allclose(blend.apply(fn, cs).pose[0, 0], 1 / (1 + np.exp(5.0)))
    assert DeployConfig.blend == "temporal_ensemble"
    assert DeployConfig.temporal_ensemble_m == 0.01


def test_a_single_contributor_passes_through_every_method():
    """Blending is inert in sync: the one contributor is the answer."""
    only = _contrib(5, ROT_Z, pos=1.5, hand=0.25, waist=-0.5)
    for fn in (blend.newest, blend.temporal_ensemble):
        got = blend.apply(fn, [only])
        assert np.allclose(got.pose, only.pose)
        assert np.allclose(got.hand, only.hand)
        assert np.allclose(got.waist, only.waist)


# --- registry ---------------------------------------------------------------


def test_registry_resolves_the_config_default():
    from g1d_deploy.config_loading.resolved import DeployConfig
    assert blend.get(DeployConfig.blend) is blend.temporal_ensemble
    assert blend.get("newest") is blend.newest
    with pytest.raises(ValueError, match="unknown blend"):
        blend.get("recency")                            # gone


def test_registry_passes_a_callable_through_and_refuses_a_name_it_lacks():
    fn = blend.newest
    assert blend.get(fn) is fn
    with pytest.raises(ValueError, match="unknown blend"):
        blend.get("slerp-o-matic")


def test_blending_nothing_is_an_error_not_an_empty_action():
    """P6 must not be able to publish a target derived from no chunk at all."""
    with pytest.raises(ValueError):
        blend.apply(blend.newest, [])



# --- joint mode and the absent waist ------------------------------


def _joint(chunk_id, q, *, waist=None):
    return Contribution(chunk_id, None, np.zeros((2, 6)),
                        None if waist is None else np.full(3, waist),
                        np.full((2, 7), q))


def test_joint_contributors_average_their_joints_and_carry_no_pose():
    got = blend.apply(blend.temporal_ensemble, [_joint(1, 0.3), _joint(2, 0.6)])
    assert got.pose is None and got.waist is None
    assert np.allclose(got.arm_q, 0.45)           # no rows: equal weights
    newest = blend.apply(blend.newest, [_joint(1, 0.3), _joint(2, 0.6)])
    assert newest.arm_q[0, 0] == pytest.approx(0.6)


def test_a_waist_only_some_contributors_command_is_no_waist():
    got = blend.apply(blend.temporal_ensemble, [_joint(1, 0.3, waist=0.1), _joint(2, 0.6)])
    assert got.waist is None
    both = blend.apply(blend.temporal_ensemble, [_joint(1, 0.3, waist=0.1),
                                                 _joint(2, 0.6, waist=0.4)])
    assert both.waist == pytest.approx(np.full(3, 0.25))


def test_an_eef_contributor_without_a_waist_blends_as_before():
    a = Contribution(1, np.array([[0, 0, 0, 1, 0, 0, 0]] * 2, float),
                     np.zeros((2, 6)), None)
    got = blend.apply(blend.temporal_ensemble, [a, a._replace(chunk_id=2)])
    assert got.waist is None and got.arm_q is None
    assert got.pose[:, 3] == pytest.approx([1.0, 1.0])

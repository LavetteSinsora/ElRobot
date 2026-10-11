"""The deploy-a-policy skill's templates work as shipped, and its conversions
are right — an agent copies them verbatim, so a sign error here would be one
in every adapter written from them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from g1d_deploy.cli import main as cli
from g1d_deploy.policy_contract.loading import load_attr
from g1d_deploy.robot import kinematics as kin

T = Path(__file__).resolve().parents[1] / "skills" / "deploy-a-policy" / "templates"


def c(name):
    return load_attr(f"{T / 'conversions.py'}:{name}")


def test_the_templates_pass_check_policy(capfd):
    with pytest.raises(SystemExit) as got:
        cli.main(["check-policy", str(T / "deploy.toml")])
    out = capfd.readouterr().out
    assert got.value.code == 0, out
    assert "[ FAIL ]" not in out


def _random_rotations(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_quaternion_matrix_round_trip_agrees_with_the_runtime():
    q = _random_rotations(50)
    R = c("matrix_from_quat")(q)
    for i in range(len(q)):
        assert np.allclose(R[i], kin.mat_from_quat(q[i]), atol=1e-12)
    back = c("quat_from_matrix")(R)
    assert np.allclose(np.abs(np.sum(back * q, axis=1)), 1.0, atol=1e-9)   # q ~ -q


def test_xyzw_and_rotvec():
    q = np.array([0.1, 0.2, 0.3, 0.9])                  # xyzw
    assert np.allclose(c("xyzw_to_wxyz")(q), [0.9, 0.1, 0.2, 0.3])
    assert np.allclose(c("wxyz_to_xyzw")(c("xyzw_to_wxyz")(q)), q)
    got = c("quat_from_rotvec")(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(got, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    assert np.allclose(c("quat_from_rotvec")(np.zeros(3)), [1, 0, 0, 0])


def test_rot6d_columns_and_rows():
    R = c("matrix_from_quat")(_random_rotations(20, seed=1))
    cols = np.concatenate([R[..., :, 0], R[..., :, 1]], -1)
    rows = np.concatenate([R[..., 0, :], R[..., 1, :]], -1)
    assert np.allclose(c("matrix_from_rot6d")(cols, layout="columns"), R, atol=1e-12)
    assert np.allclose(c("matrix_from_rot6d")(rows, layout="rows"), R, atol=1e-12)
    with pytest.raises(TypeError):
        c("matrix_from_rot6d")(cols)                      # the layout is a choice
    with pytest.raises(ValueError):
        c("matrix_from_rot6d")(cols, layout="cols")


@pytest.mark.parametrize("axis", [(1, -1, 0), (-1, 2, -3), (0, 0, 1), (1, 0, 0)])
def test_half_turns_survive_the_matrix_round_trip(axis):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    q = np.concatenate([[0.0], axis])                     # 180 degrees
    R = c("matrix_from_quat")(q)
    assert np.allclose(c("matrix_from_quat")(c("quat_from_matrix")(R)), R, atol=1e-12)


def test_tcp_and_flange_convert_both_ways():
    tcp = np.array([0.184759, 0, 0, 1, 0, 0, 0])
    flange = np.array([[0.2, 0.149, 0.095, 1, 0, 0, 0],
                       [0.2, -0.149, 0.095, 0.7071068, 0, 0.7071068, 0]])
    tip = c("flange_to_tcp")(flange, tcp)
    assert np.allclose(tip[0, :3], [0.384759, 0.149, 0.095])      # +x of the flange
    assert np.allclose(tip[1, :3], [0.2, -0.149, 0.095 - 0.184759], atol=1e-6)
    assert np.allclose(c("tcp_to_flange")(tip, tcp), flange, atol=1e-6)


def test_deltas_local_and_world_differ_when_the_reference_is_rotated():
    ref = np.array([0.2, 0.0, 0.1, 0.7071068, 0, 0, 0.7071068])   # 90 deg about z
    step_x = np.array([0.01, 0, 0, 1, 0, 0, 0])
    local = c("apply_delta")(ref, step_x, frame="local")
    world = c("apply_delta")(ref, step_x, frame="world")
    assert np.allclose(local[:3], [0.2, 0.01, 0.1], atol=1e-6)     # along the ref's own x = world y
    assert np.allclose(world[:3], [0.21, 0.0, 0.1], atol=1e-6)
    with pytest.raises(ValueError):
        c("apply_delta")(ref, step_x, frame="tool")

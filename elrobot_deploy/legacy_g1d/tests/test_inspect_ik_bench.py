"""The IK bench: the infra's solvers re-solving a record, into `ik_bench.npz`.

The record stands at its home pose and commands that same pose, so a solver
that works reaches it: the TCP residual is ~0. What is checked is the plumbing
the viewer relies on — the file's layout (`overlays.py` reads it), the merge,
and the joint-mode rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from elrobot_deploy.dashboards.inspect import overlays
from elrobot_deploy.dashboards.inspect.producers import ik_bench

from . import inspect_fixtures as fx

pytest.importorskip("mujoco")


def test_a_solver_is_benched_into_the_sidecar_and_merged(tmp_path):
    rec = fx.write(tmp_path / "r", n_chunks=2)
    out = ik_bench.solve(rec, ["arm_qp"], log=lambda *_: None)
    assert out == rec / overlays.BENCH_NAME
    with np.load(out) as z:
        assert [str(m) for m in z["methods"]] == ["arm_qp"]
        n = len(z["k"])
        assert z["arm_qp/arm"].shape == (n, 14) and z["arm_qp/waist"].shape == (n, 3)
        assert float(z["arm_qp/tcp_err"].max()) < 2e-3
        assert float(z["recorded/tcp_err"].max()) < 1e-6

    ik_bench.solve(rec, ["arm_planner"], log=lambda *_: None)
    ov = overlays.load(rec)
    assert ov.tiers == ["arm_qp", "arm_planner"]           # merged, not replaced
    assert ov.meta["names"]["arm_planner"]["source"] == "bench"


def test_a_joint_record_benches_only_what_an_operator_commanded(tmp_path):
    with pytest.raises(ValueError, match="no operator intervention"):
        ik_bench.solve(fx.write(tmp_path / "j", example="sine_joint", n_chunks=2),
                       ["arm_qp"], log=lambda *_: None)
    rec = fx.write(tmp_path / "p", example="sine_joint", n_chunks=2,
                   intervene=[(5, 12)])
    with np.load(ik_bench.solve(rec, ["arm_qp"], log=lambda *_: None)) as z:
        assert list(z["k"]) == list(range(5, 12))

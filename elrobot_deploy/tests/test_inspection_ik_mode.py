"""The IK mode's half of the solve seam — what it reads, and what it spawns.

The mode reads `ik_bench.npz` to say what a recording already has, decides
which of the infra's solvers are worth solving, refuses a record with no
Cartesian target (a joint-mode rollout nobody intervened in), and spawns the
bench — the infra's own producer, by module path, with the dashboard's own
interpreter. The producer itself is exercised in `test_inspect_ik_bench.py`.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from g1d_deploy.dashboards.inspect.modes import ik as I


class _Rollout:
    """The `loader.Rollout` surface `applicable` reads."""

    def __init__(self, pose_finite=True, action_mode="eef"):
        dt = np.dtype([("k", "<i8"), ("pose", "<f8", (2, 7))])
        self.body_targets = np.zeros(4, dtype=dt)
        if not pose_finite:
            self.body_targets["pose"] = np.nan
        self.action_mode = action_mode


class _Loaded:
    def __init__(self, rollout):
        self.rollout = rollout


class _State:
    """The `server.State` surface the mode uses: a record and its rollout."""

    def __init__(self, record_dir, rollout=None):
        self.record_dir = record_dir
        self.current = _Loaded(rollout or _Rollout())


def _sidecar(path, names):
    np.savez(path / "ik_bench.npz", version=np.array([1]),
             k=np.arange(4), methods=np.array(names),
             meta=np.array(['{"methods": {}}']))


def _mode(record_dir, registry, rollout=None):
    mode = I.IkMode()
    mode._state = _State(record_dir, rollout)
    mode._registry = registry
    return mode


_REG = {"methods": [
    {"name": "arm_planner", "mode": "arm", "describe": "d",
     "available": True, "why": ""},
    {"name": "arm_qp", "mode": "arm", "describe": "d",
     "available": True, "why": ""},
    {"name": "waist_planner", "mode": "waist", "describe": "d",
     "available": True, "why": ""},
    {"name": "waist_qp", "mode": "", "describe": "",
     "available": False, "why": "not registered"},
], "error": ""}


def test_the_defaults_are_exactly_the_infra_solvers():
    """The bench runs the infra's own solvers and nothing an experiment owns."""
    from g1d_deploy.runtime import ik as ik_solvers
    assert I.default_methods() == tuple(ik_solvers.SOLVERS)
    assert set(I.default_methods()) >= {"arm_planner", "arm_qp", "waist_planner",
                                        "waist_qp"}


def test_solved_methods_reads_the_sidecar_in_file_order(tmp_path):
    """File order, because that is the order the viewer colours them in."""
    _sidecar(tmp_path, ["waist_planner", "arm_planner"])
    assert I.solved_methods(tmp_path) == ["waist_planner", "arm_planner"]


def test_solved_methods_is_empty_without_a_readable_sidecar(tmp_path):
    assert I.solved_methods(tmp_path) == []
    assert I.solved_methods(None) == []
    (tmp_path / "ik_bench.npz").write_bytes(b"not an npz")
    assert I.solved_methods(tmp_path) == []


def test_missing_is_the_default_set_minus_solved_and_unavailable(tmp_path):
    """What auto-solve asks for. A solver this install does not register is
    not 'missing': the picker greys it out with the reason instead."""
    _sidecar(tmp_path, ["arm_planner"])
    out = _mode(tmp_path, _REG).methods()
    assert out["missing"] == ["arm_qp", "waist_planner"]
    rows = {m["name"]: m for m in out["methods"]}
    assert rows["arm_planner"]["solved"] and rows["arm_planner"]["default"]
    assert not rows["waist_qp"]["available"]


def test_auto_solve_skips_the_solver_the_rollout_flew(tmp_path):
    """Its recorded joints are the `deployed` line already; re-solving it would
    only redraw that line."""
    rollout = _Rollout()
    rollout.manifest = {"deploy_config": {"ik": "arm_qp"}}
    out = _mode(tmp_path, _REG, rollout).methods()
    assert out["deployed"] == "arm_qp"
    assert out["missing"] == ["arm_planner", "waist_planner"]


def test_a_sidecar_method_the_registry_lost_is_still_listed(tmp_path):
    """It is drawn by the viewer either way, so the picker must not pretend it
    is not there — it just cannot be re-solved here."""
    _sidecar(tmp_path, ["some_method_from_another_build"])
    rows = {m["name"]: m for m in _mode(tmp_path, _REG).methods()["methods"]}
    row = rows["some_method_from_another_build"]
    assert row["solved"] and not row["available"] and row["why"]


def test_a_joint_rollout_nobody_intervened_in_has_nothing_to_solve(tmp_path):
    """No Cartesian target, no IK question: the mode says why and asks nothing."""
    mode = _mode(tmp_path, _REG, _Rollout(pose_finite=False, action_mode="joint"))
    out = mode.methods()
    assert not out["applicable"] and "joint-mode" in out["why"]
    assert out["missing"] == []
    with pytest.raises(RuntimeError, match="joint-mode"):
        mode.solve({"methods": ["arm_planner"]})


def test_a_joint_rollout_with_an_intervention_can_be_benched(tmp_path):
    """PICO commands are poses, so an intervened stretch has something to
    re-solve even when the policy spoke joints."""
    rollout = _Rollout(pose_finite=False, action_mode="joint")
    rollout.body_targets["pose"][1] = [[0.3, 0.2, 0.9, 1, 0, 0, 0]] * 2
    assert I.applicable(rollout) == (True, "")


def test_solve_spawns_the_infra_bench_on_the_recording(tmp_path, monkeypatch):
    """The command is the contract with `modes/ik_bench.py`: the same
    interpreter, the module by path, the record, the methods."""
    seen = {}

    def fake_submit(**kw):
        seen.update(kw)

        class _Job:
            def as_json(self):
                return {"id": "abcd1234"}
        return _Job()

    monkeypatch.setattr(I.RUNNER, "submit", fake_submit)
    monkeypatch.setattr(I.RUNNER, "busy", lambda: False)

    mode = _mode(tmp_path, _REG)
    assert mode.solve({"methods": ["arm_planner", "arm_qp"]}) \
        == {"job": {"id": "abcd1234"}}
    cmd = seen["cmd"]
    assert cmd[:3] == [sys.executable, "-m", I.PRODUCER]
    assert str(tmp_path) in cmd
    assert cmd[cmd.index("--methods") + 1] == "arm_planner,arm_qp"
    assert seen["jobs_dir"] == tmp_path / I.JOBS_DIR
    assert seen["meta"]["methods"] == ["arm_planner", "arm_qp"]


def test_solve_refuses_without_a_rollout_or_while_busy(tmp_path, monkeypatch):
    mode = _mode(tmp_path, _REG)
    monkeypatch.setattr(I.RUNNER, "busy", lambda: True)
    with pytest.raises(RuntimeError, match="one at a time"):
        mode.solve({"methods": ["arm_planner"]})
    monkeypatch.setattr(I.RUNNER, "busy", lambda: False)
    with pytest.raises(RuntimeError, match="no methods"):
        mode.solve({"methods": []})
    mode._state.current = None
    with pytest.raises(RuntimeError, match="load a rollout"):
        mode.solve({"methods": ["arm_planner"]})


def test_a_bench_that_will_not_list_is_reported_not_raised(tmp_path):
    """A broken install still gets a working panel for what the recording
    already carries."""
    out = _mode(tmp_path, {"methods": [], "error": "boom"}).methods()
    assert out["error"] == "boom" and out["methods"] == []
    assert out["missing"] == []


def test_the_registry_resolves_names_through_ik_solvers():
    """Resolved at run time by name: the bench offers exactly what
    `runtime/ik` `SOLVERS` ships, each with its mode."""
    from g1d_deploy.runtime import ik as ik_solvers
    from g1d_deploy.dashboards.inspect.producers import ik_bench

    reg = ik_bench.registry()
    assert not reg["error"]
    rows = {m["name"]: m for m in reg["methods"]}
    assert list(rows) == list(ik_solvers.SOLVERS)
    for name, row in rows.items():
        assert row["available"], row["why"]
        assert row["mode"] == ik_solvers.SOLVERS[name][1]

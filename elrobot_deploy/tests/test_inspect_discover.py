"""Discovery: a record is a directory holding P9's manifest.json, anywhere.

No naming convention, no experiment tree: the dropdown groups by the record's
path relative to the root, so whatever layout the records are kept in is the
grouping (`dashboards/inspect/discover.py`).
"""

from __future__ import annotations

import json

import pytest

from elrobot_deploy.dashboards.inspect import discover


def _record(path, **cfg):
    """The smallest manifest discovery accepts as P9's: a `state` stream."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({
        "created": "2026-09-26T10:00:00",
        "streams": {"state": {"file": "state.bin", "descr": [], "records": 0}},
        "deploy_config": {"replan_period_ticks": 25, "action_horizon": 50, **cfg},
    }))
    return path


def test_any_layout_groups_by_the_relative_path(tmp_path):
    """The old experiment/sub-experiment/rollout_record tree, a flat directory
    of stamps and an arbitrary nesting, side by side under one root."""
    _record(tmp_path / "my_task" / "run_6" / "rollout_record" / "20260926-101010")
    _record(tmp_path / "my_task" / "run_6" / "rollout_record" / "20260926-111111")
    _record(tmp_path / "20260925-090000")
    _record(tmp_path / "lab" / "tuesday" / "take 3")

    recs = discover.scan(tmp_path)
    by_key = {r.key: r for r in recs}
    assert set(by_key) == {
        "my_task/run_6/rollout_record/20260926-101010",
        "my_task/run_6/rollout_record/20260926-111111",
        "20260925-090000", "lab/tuesday/take 3"}
    assert by_key["20260925-090000"].group == ""
    assert by_key["lab/tuesday/take 3"].group == "lab/tuesday"
    assert by_key["my_task/run_6/rollout_record/20260926-101010"].group \
        == "my_task/run_6/rollout_record"

    listing = discover.as_json(recs, tmp_path)
    groups = [g["group"] for g in listing["groups"]]
    assert groups == sorted(groups)
    stamps = next(g for g in listing["groups"]
                  if g["group"] == "my_task/run_6/rollout_record")["records"]
    # Newest first within a group: the rollout you want is nearly always the last.
    assert [r["name"] for r in stamps] == ["20260926-111111", "20260926-101010"]
    assert stamps[0]["label"] == "2026-09-26 11:11:11"
    assert next(r for r in recs if r.name == "take 3").label == "take 3"


def test_a_single_record_opens_just_that_record(tmp_path):
    run = _record(tmp_path / "somewhere" / "20260926-101010")
    _record(tmp_path / "somewhere" / "20260926-121212")
    recs = discover.scan(run)
    assert [(r.key, r.group) for r in recs] == [(".", "")]
    assert discover.as_json(recs, run)["single"] is True
    assert discover.resolve(".", run) == run.resolve()


def test_a_record_is_never_searched_for_records(tmp_path):
    """A record's own results (policy replays, executor replays) carry
    manifests too; they are the record's, not recordings of their own."""
    run = _record(tmp_path / "20260926-101010")
    _record(run / "executor_replays" / "20260926-120000")
    (run / "policy_replays" / "x").mkdir(parents=True)
    (run / "policy_replays" / "x" / "manifest.json").write_text(
        json.dumps({"schema": "elrobot_deploy.policy_replay.v1", "streams": {}}))
    assert [r.key for r in discover.scan(tmp_path)] == ["20260926-101010"]


def test_a_manifest_p9_did_not_write_is_not_a_record(tmp_path):
    """A replay result, a dataset's metadata, anything else called
    manifest.json: no `state` stream, not offered."""
    other = tmp_path / "not_a_rollout"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"schema": "x", "streams": {}}))
    _record(tmp_path / "real")
    assert [r.key for r in discover.scan(tmp_path)] == ["real"]


def test_hidden_directories_are_not_walked(tmp_path):
    _record(tmp_path / ".git" / "objects" / "r")
    _record(tmp_path / "a.partial" / "r")
    _record(tmp_path / "visible")
    assert {r.key for r in discover.scan(tmp_path)} == {"a.partial/r", "visible"}


def test_resolve_walks_the_tree_and_refuses_what_is_not_in_it(tmp_path):
    """A crafted key cannot address anything outside the root."""
    _record(tmp_path / "a" / "r1")
    assert discover.resolve("a/r1", tmp_path) == (tmp_path / "a" / "r1").resolve()
    with pytest.raises(KeyError):
        discover.resolve("../../etc", tmp_path)
    with pytest.raises(KeyError):
        discover.resolve("a/nope", tmp_path)


def test_an_unreadable_manifest_is_listed_not_hidden(tmp_path):
    run = tmp_path / "broken"
    run.mkdir()
    (run / "manifest.json").write_text('{"created": "2026-')
    rec = discover.scan(tmp_path)[0]
    assert not rec.loadable and "could not be read" in rec.reason
    with pytest.raises(KeyError, match="could not be read"):
        discover.resolve("broken", tmp_path)


def test_the_server_lists_and_loads_by_key(tmp_path):
    """`/api/tree` and `/api/load` speak keys: the path under the root."""
    from elrobot_deploy.dashboards.inspect import server

    from . import inspect_fixtures as fx

    fx.write(tmp_path / "exp_x" / "rollout_record" / "20260926-120000")
    app = server.build_app(tmp_path, modes=[])
    tree = discover.as_json(discover.scan(app.options.root), app.options.root)
    assert tree["groups"][0]["group"] == "exp_x/rollout_record"
    loaded = app.state.load("exp_x/rollout_record/20260926-120000", 30.0)
    d = loaded.describe()
    assert d["key"] == "exp_x/rollout_record/20260926-120000"
    assert d["name"] == "20260926-120000"
    assert d["summary"]["robot"] == "g1d_fisheye_dex1"
    app.close()

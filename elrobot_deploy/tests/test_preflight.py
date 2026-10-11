"""`g1d_deploy check` (`preflight`) must report every check, not stop at the first failure.

The whole value of the tool is the LIST — a bring-up where you fix one thing,
re-run, and discover the next is the thing it exists to replace. So the
property under test is that a failing check does not abort the ones after it,
and that a failure always carries a remedy.
"""

from __future__ import annotations

import pytest

from g1d_deploy.cli import main as entry
from g1d_deploy.cli import check as preflight


def test_a_failing_check_does_not_stop_the_ones_after_it():
    rep = preflight.Report()
    rep.tier("T")
    rep.check("first", lambda: (_ for _ in ()).throw(RuntimeError("boom")),
              remedy="do the thing")
    rep.check("second", lambda: "fine")
    assert [r[1] for r in rep.rows] == [preflight._FAIL, preflight._PASS]
    assert rep.failed == 1


def test_every_failure_carries_a_remedy_or_says_nothing_useful_was_known():
    """A FAIL with no remedy is a dead end for whoever is standing at the box."""
    rep = preflight.Report()
    rep.tier("T")
    rep.check("no remedy", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    tier, status, what, remedy = rep.rows[0]
    assert status == preflight._FAIL
    assert remedy == ""          # allowed, but the summary must still name it


def test_skip_is_not_failure():
    rep = preflight.Report()
    rep.tier("T")
    rep.check("needs linux", lambda: "never called", skip_if="not Linux")
    assert rep.failed == 0
    assert rep.rows[0][1] == preflight._SKIP


def test_exit_status_is_zero_only_when_nothing_failed():
    rep = preflight.Report()
    rep.tier("T")
    rep.check("ok", lambda: "yes")
    assert rep.summary() == 0
    rep.check("bad", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert rep.summary() == 1


def test_the_package_list_names_import_names_not_distribution_names():
    """`pin` installs `pinocchio`, `opencv-python` installs `cv2`, `pyzmq`
    installs `zmq`. Checking the distribution name would pass on a box where
    the import fails, which is the failure this tool exists to catch early."""
    mods = {m for m, _ in preflight._PACKAGES}
    assert "pinocchio" in mods and "pin" not in mods
    assert "cv2" in mods and "opencv-python" not in mods
    assert "zmq" in mods and "pyzmq" not in mods
    # an optional extra, reported but never a failure of the base install
    assert "pyrealsense2" not in mods
    assert "pyrealsense2" in {m for m, _, _ in preflight._OPTIONAL}


def test_policy_dependencies_are_not_the_runtimes():
    """jax and openpi are a POLICY's dependencies; a box that serves a
    numpy-only policy must not fail `g1d_deploy check` for lacking them."""
    mods = {m for m, _ in preflight._PACKAGES}
    assert not {"jax", "openpi", "flax", "orbax"} & mods


def test_check_machine_reports_a_named_machine_file_that_is_missing(capsys):
    rc = preflight.check_machine(machine="/nonexistent/machine.toml")
    assert rc == 1
    assert "does not exist" in capsys.readouterr().out


def test_check_robot_stops_early_on_an_unknown_robot(capsys):
    rc = preflight.check_robot(robot="no_such_robot")
    assert rc == 1
    out = capsys.readouterr().out
    assert "no shipped robot profile" in out and "CAMERAS" not in out


# --- The `g1d_deploy` command line ------------------------------------------------------


def _help(capsys, argv):
    with pytest.raises(SystemExit):
        entry.main(argv)
    return capsys.readouterr().out


def test_help_lists_run_check_inspect_and_hides_the_replay(capsys):
    out = _help(capsys, ["--help"])
    for cmd in ("run", "check", "inspect"):
        assert f"    {cmd} " in out
    assert "replay-policy" not in out
    assert "check-machine" not in out and "check-robot" not in out


def test_replay_policy_still_parses(capsys):
    assert "--record" in _help(capsys, ["replay-policy", "--help"])


def test_check_runs_the_machine_first_then_the_robot(monkeypatch):
    calls = []
    monkeypatch.setattr(preflight, "check_machine",
                        lambda m: calls.append("machine") or 0)
    monkeypatch.setattr(preflight, "check_robot",
                        lambda **kw: calls.append(("robot", kw["robot"])) or 1)
    with pytest.raises(SystemExit) as exc:
        entry.main(["check", "--robot", "g1d_normal_dex1"])
    assert calls == ["machine", ("robot", "g1d_normal_dex1")]
    assert exc.value.code == 1, "a robot failure fails the whole check"

    calls.clear()
    with pytest.raises(SystemExit):
        entry.main(["check", "--no-robot"])
    assert calls == ["machine"]
    calls.clear()
    with pytest.raises(SystemExit):
        entry.main(["check", "--no-machine"])
    assert calls == [("robot", None)]


def test_check_refuses_to_skip_both():
    with pytest.raises(SystemExit) as exc:
        entry.main(["check", "--no-machine", "--no-robot"])
    assert exc.value.code == 2

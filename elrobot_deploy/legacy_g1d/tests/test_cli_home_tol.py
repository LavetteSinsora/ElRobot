"""`--home-tol-arm` / `--home-tol-waist`: loosening the homing check per run.

The tolerances live in the deploy file's `[home]`. The flags override them on
the deploy tree, before `validate`, so what P0 enforces and what the rollout
record carries are the same number — and each flag left off keeps the deploy
file's own value.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from elrobot_deploy.cli import main as entry

DEPLOY = Path(__file__).resolve().parents[1] / "examples" / "deploy.toml"


def _cfg_launched_with(monkeypatch, argv):
    seen = {}

    class _Launcher:
        def __init__(self, cfg, hw, **kwargs):
            seen["cfg"] = cfg

        def run(self):
            pass

    monkeypatch.setattr(entry, "Launcher", _Launcher)
    entry.main(["run", str(DEPLOY), "--no-record", *argv])
    return seen["cfg"]


def test_without_the_flags_the_deploy_files_tolerances_stand(monkeypatch):
    cfg = _cfg_launched_with(monkeypatch, [])
    assert (cfg.home_tol_arm, cfg.home_tol_waist) == (0.15, 0.05)


def test_both_tolerances_can_be_raised_for_one_run(monkeypatch):
    cfg = _cfg_launched_with(
        monkeypatch, ["--home-tol-arm", "0.25", "--home-tol-waist", "0.08"])
    assert (cfg.home_tol_arm, cfg.home_tol_waist) == (0.25, 0.08)


def test_one_override_leaves_the_other_alone(monkeypatch):
    cfg = _cfg_launched_with(monkeypatch, ["--home-tol-arm", "0.3"])
    assert (cfg.home_tol_arm, cfg.home_tol_waist) == (0.3, 0.05)


@pytest.mark.parametrize("flag", ["--home-tol-arm", "--home-tol-waist"])
@pytest.mark.parametrize("value", ["0", "-0.1", "nan"])
def test_a_tolerance_that_is_not_positive_is_refused(flag, value, monkeypatch):
    """Zero refuses every homing and NaN passes every one (`x > nan` is False),
    so neither is a loosening."""
    with pytest.raises(SystemExit):
        _cfg_launched_with(monkeypatch, [flag, value])

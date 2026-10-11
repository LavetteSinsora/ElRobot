"""`--intervene` is refused with any IK but `arm_qp`, the per-tick arm QP.

The intervention and the waist-mode IK (docs/infra_logic/ik_solvers.md) meet in P6 and
nobody has specified what a planned trunk should do when an operator abandons
the chunk it was planned for; the arm PLANNER likewise holds a per-chunk plan
(and an escape toward it) that an operator's targets would be steered by. Until
somebody specifies either, the combination exits before anything is loaded,
and says it is unsupported rather than failing somewhere stranger later.

`arm_qp` is what an intervention runs when nobody named an IK: the default
(`arm_planner`) becomes `arm_qp` under `--intervene`, as the servo becomes the
linear one — written as an override, so the record says why.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from elrobot_deploy.cli import main as entry
from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.config_loading.validate import check_intervene_ik
from elrobot_deploy.runtime.launcher import Launcher

DEPLOY = Path(__file__).resolve().parents[1] / "examples" / "deploy.toml"


def _launch(monkeypatch, argv, *, sdk=True):
    """Run the CLI with a fake Launcher. Returns the kwargs it got, or None if
    it was never constructed. `sdk` fakes whether `xrobotoolkit_sdk` imports."""
    from elrobot_deploy.runtime.processes import p11_pico
    monkeypatch.setattr(p11_pico, "sdk_available", lambda: sdk)
    seen = {}

    class _Launcher:
        def __init__(self, cfg, hw, **kwargs):
            seen.update(kwargs, cfg=cfg)

        def run(self):
            pass

    monkeypatch.setattr(entry, "Launcher", _Launcher)
    entry.main(["run", str(DEPLOY), "--no-record", *argv])
    return seen or None


@pytest.mark.parametrize("ik", ["waist", "waist_planner", "waist_qp",
                                "arm_planner"])
def test_intervene_with_any_other_ik_exits_as_unsupported(ik, monkeypatch,
                                                          capsys):
    with pytest.raises(SystemExit) as exc:
        _launch(monkeypatch, ["--intervene", "--ik", ik])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "currently unsupported" in err and repr(ik) in err


def test_the_refusal_comes_before_the_launcher_is_built(monkeypatch):
    built = []
    monkeypatch.setattr(cfgmod, "load_hardware",
                        lambda *a, **k: built.append("hw"))
    with pytest.raises(SystemExit):
        _launch(monkeypatch, ["--intervene", "--ik", "waist"])
    assert built == [], "nothing past the refusal ran"


@pytest.mark.parametrize("argv", [[], ["--ik", "arm"], ["--ik", "arm_qp"]])
def test_intervene_runs_the_arm_qp(argv, monkeypatch):
    """Unsaid, the bare "arm" (the arm mode's default) and `arm_qp` itself
    all launch — on `arm_qp`, never on the `arm_planner` default."""
    seen = _launch(monkeypatch, ["--intervene", *argv])
    assert seen is not None and seen["pico"] is True
    assert seen["cfg"].ik == "arm_qp"


def test_without_intervene_the_default_is_the_arm_planner(monkeypatch):
    seen = _launch(monkeypatch, [])
    assert seen["cfg"].ik == "arm_planner" and seen["pico"] is False


def test_the_launcher_refuses_it_too(cfg, hw):
    """A programmatic launch does not go through `__main__`, so the launcher
    carries the same check rather than trusting its caller — including for the
    `arm_planner` default, which `__main__` would have turned into `arm_qp`."""
    for ik in ("waist_qp", "arm_planner"):
        with pytest.raises(ConfigError, match="currently unsupported"):
            Launcher(cfgmod.replace(cfg, ik=ik), hw, pico=True)


def test_the_check_is_silent_without_intervene(cfg):
    check_intervene_ik(cfgmod.replace(cfg, ik="waist"), intervene=False)
    check_intervene_ik(cfg, intervene=False)
    check_intervene_ik(cfgmod.replace(cfg, ik="arm_qp"), intervene=True)


def test_intervene_without_the_sdk_says_how_to_build_it(monkeypatch, capsys):
    """`xrobotoolkit_sdk` is a native extension no package install provides;
    the refusal names the build script in this checkout, and it exists."""
    from elrobot_deploy.runtime.processes import p11_pico

    with pytest.raises(SystemExit) as exc:
        _launch(monkeypatch, ["--intervene"], sdk=False)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "xrobotoolkit_sdk" in err and p11_pico.build_script() in err
    assert Path(p11_pico.build_script()).is_file()

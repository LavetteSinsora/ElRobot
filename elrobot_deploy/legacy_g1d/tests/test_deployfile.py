"""The deploy file, the policy entry, checkpoint discovery and the layering.

    infra defaults < the policy's trained_* attributes < deploy.toml < command line

Each layer is pinned on its own, and so is the record of which one won: a run
whose startup banner or manifest misattributes a value is one that cannot be
compared with another.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from elrobot_deploy.policy_contract import checkpoints, loading as entry
from elrobot_deploy.config_loading import deploy_file as deployfile, machine_file as machine
from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.config_loading.validate import validate

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DEPLOY = EXAMPLES / "deploy.toml"


def _policy_file(tmp_path, body: str, name="mypolicy.py") -> Path:
    f = tmp_path / name
    f.write_text(textwrap.dedent(body))
    return f


def _deploy(tmp_path, body: str) -> Path:
    f = tmp_path / "deploy.toml"
    f.write_text(textwrap.dedent(body))
    return f


HOME = "[home]\narm = [0.0,0,0,0,0,0,0, 0,0,0,0,0,0,0]\n"


# --- entry --------------------------------------------------------------------


def test_a_module_entry_imports_the_module():
    cls = entry.load_entry("elrobot_deploy.policy_contract:EchoPolicy")
    assert cls.__name__ == "EchoPolicy"


def test_a_file_entry_is_relative_to_the_deploy_file(tmp_path):
    _policy_file(tmp_path, "class P:\n    anchor_camera = 'head_left'\n")
    spec = entry.normalize("mypolicy.py:P", base_dir=tmp_path)
    assert spec == f"{(tmp_path / 'mypolicy.py').resolve()}:P"
    assert entry.load_entry(spec).anchor_camera == "head_left"


def test_a_packaged_file_is_imported_under_its_dotted_name():
    cls = entry.load_entry(f"{EXAMPLES / 'policies' / 'hold.py'}:Policy")
    assert cls.__module__ == "examples.policies.hold"


@pytest.mark.parametrize("spec, match", [
    ("no_colon", "must be"),
    ("x.py:", "must be"),
    ("elrobot_deploy.policy_contract:NoSuchThing", "defines no"),
    ("elrobot_deploy.policy_contract:LATCH", "defines no"),
    ("elrobot_deploy.no_such_module:X", "no module"),
    ("elrobot_deploy.policy_contract:__all__", "not a class"),
    ("/nonexistent/p.py:P", "does not exist"),
])
def test_a_bad_entry_is_refused_by_name(spec, match):
    with pytest.raises(ConfigError, match=match):
        entry.load_entry(spec)


# --- checkpoints ----------------------------------------------------------------


def _run(tmp_path, *steps):
    run = tmp_path / "run"
    for s in steps:
        (run / s).mkdir(parents=True)
    return run


def test_a_path_is_taken_as_is_and_empty_means_none(tmp_path):
    assert checkpoints.resolve(path=str(tmp_path)) == str(tmp_path)
    assert checkpoints.resolve(path="rel", base_dir=tmp_path) == str(tmp_path / "rel")
    assert checkpoints.resolve() == ""


def test_latest_is_the_highest_numeric_step(tmp_path):
    run = _run(tmp_path, "999", "10000", "2000", "best")
    assert checkpoints.resolve(run=str(run)) == str(run / "10000")
    assert checkpoints.resolve(run=str(run), step="latest") == str(run / "10000")
    assert checkpoints.resolve(run=str(run), step=2000) == str(run / "2000")


def test_a_missing_step_and_an_unknown_alias_are_refused(tmp_path):
    run = _run(tmp_path, "10")
    with pytest.raises(ConfigError, match="not in"):
        checkpoints.resolve(run=str(run), step=11)
    with pytest.raises(ConfigError, match="resolve_checkpoint"):
        checkpoints.resolve(run=str(run), step="crowned")


def test_the_policy_resolves_its_own_layout_first(tmp_path):
    seen = []

    class P:
        @classmethod
        def resolve_checkpoint(cls, run, step):
            seen.append((run, step))
            return run / "best" / step

    run = _run(tmp_path, "5")
    out = checkpoints.resolve(P, run=str(run), step="crowned")
    assert out == str(run / "best" / "crowned")
    assert seen == [(run, "crowned")] and isinstance(seen[0][0], Path)


def test_path_and_run_together_are_ambiguous(tmp_path):
    with pytest.raises(ConfigError, match="both"):
        checkpoints.resolve(path="a", run="b")


# --- The example deploy file ------------------------------------------------------


def test_the_example_resolves_to_the_hold_policy_on_the_fisheye_rig():
    r = deployfile.resolve(DEPLOY)
    cfg = r.cfg
    assert cfg.robot == "g1d_fisheye_dex1"
    assert cfg.policy_entry.endswith("examples/policies/hold.py:Policy")
    assert cfg.primary_camera == "head_left", "from the class's anchor_camera"
    assert sorted(cfg.cameras) == ["head", "lwrist", "rwrist"]
    assert (cfg.P, cfg.H, cfg.control_hz, cfg.sync) == (25, 50, 30.0, True)
    assert cfg.home_q.shape == (14,) and cfg.checkpoint == ""
    assert np.all(cfg.exit_q == cfg.home_q), "exit defaults to home"
    assert r.record_dir == EXAMPLES / "rollout_record"
    assert r.sources["replan_period_ticks"] == "deploy"
    assert r.sources["primary_camera"] == "trained"
    validate(cfg, r.hw)


def test_the_machine_file_env_reaches_the_policy_process():
    r = deployfile.resolve(DEPLOY)
    assert r.cfg.policy_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"


def test_cli_flags_and_set_override_the_file_and_say_so():
    r = deployfile.resolve(DEPLOY, flags={"replan_period": 10, "prompt": "go"},
                           sets=["control.sample_steps=4", "labels.arm=b",
                                 "control.replan_period=12"])
    assert (r.cfg.P, r.cfg.sample_steps, r.cfg.prompt) == (12, 4, "go")
    assert r.cfg.labels == {"arm": "b"}
    assert r.sources["replan_period_ticks"] == "cli"
    assert r.sources["control_hz"] == "deploy"
    assert r.tree["control"]["replan_period"] == 12, "the tree that ran is kept"


def test_checkpoint_flag_replaces_the_files_run(tmp_path):
    run = _run(tmp_path, "3")
    f = _deploy(tmp_path, f'robot = "g1d_normal_dex1"\n[checkpoint]\nrun = "{run}"\n{HOME}')
    assert deployfile.resolve(f).cfg.checkpoint == str(run / "3")
    r = deployfile.resolve(f, checkpoint=str(tmp_path))
    assert r.cfg.checkpoint == str(tmp_path) and r.sources["checkpoint"] == "cli"


def test_step_flag_picks_a_step_of_the_files_run(tmp_path):
    run = _run(tmp_path, "3", "7")
    f = _deploy(tmp_path, f'robot = "g1d_normal_dex1"\n[checkpoint]\nrun = "run"\n{HOME}')
    assert deployfile.resolve(f, flags={"step": "3"}).cfg.checkpoint == str(run / "3")


@pytest.mark.parametrize("body, match", [
    ("[control]\nreplan_peroid = 3\n" + HOME, "replan_peroid"),
    ("colour = 1\n" + HOME, "colour"),
    ("[home]\ntol = 1\n", "tol"),
    ("robot = \"nope\"\n" + HOME, "no shipped robot"),
    # Gone, and said so by name rather than as an unknown key.
    ("[control]\nik_solver = \"arm_planner\"\n" + HOME, "ik_solver is gone"),
    ("[intervene]\ntcp = [0.18, 0, 0, 1, 0, 0, 0]\n" + HOME,
     r"\[end_effector\] tcp"),
])
def test_a_wrong_deploy_file_is_refused_by_name(tmp_path, body, match):
    with pytest.raises(ConfigError, match=match):
        deployfile.resolve(_deploy(tmp_path, body))


def test_without_home_arm_the_arm_goes_to_the_infras_default_home(tmp_path):
    r = deployfile.resolve(_deploy(tmp_path, 'robot = "g1d_normal_dex1"\n'))
    assert np.allclose(r.cfg.home_q, deployfile.DEFAULT_HOME_ARM)
    assert np.allclose(r.cfg.home_waist, 0.0)
    assert r.sources["home_q"] == "infra"
    assert np.allclose(r.cfg.exit_q, r.cfg.home_q), "exit defaults to home"


@pytest.mark.parametrize("ik, want", [
    (None, "arm_planner"), ("arm", "arm_planner"), ("arm_qp", "arm_qp"),
    ("waist", "waist_planner"), ("waist_qp", "waist_qp"),
])
def test_ik_names_the_solver_and_the_shorthands_resolve(tmp_path, ik, want):
    """What the record will say ran. `ik = "arm"` (or nothing) is the arm
    mode's default, `arm_planner` — the pre-2026-09-26 default is `arm_qp`."""
    line = "" if ik is None else f'ik = "{ik}"\n'
    robot = 'robot = "g1d_normal_dex1"\n'
    f = _deploy(tmp_path, f"{robot}[control]\n{line}{HOME}")
    assert deployfile.resolve(f).cfg.ik == want
    assert deployfile.resolve(_deploy(tmp_path, robot + HOME),
                              flags={"ik": ik}).cfg.ik \
        == want


def test_no_anchor_on_a_multi_camera_robot_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="anchor"):
        deployfile.resolve(_deploy(tmp_path, HOME))
    one = deployfile.resolve(_deploy(tmp_path, 'robot = "g1d_normal_dex1"\n' + HOME))
    assert one.cfg.primary_camera == "head_left", "the only camera needs no naming"


def test_hand_values_are_in_the_hands_own_units(tmp_path):
    f = _deploy(tmp_path, 'robot = "g1d_normal_brainco"\n' + HOME
                + 'hand = "closed"\n[exit]\nhand = [0.2, 0.3]\n')
    cfg = deployfile.resolve(f).cfg
    assert list(cfg.home_hand[:, 0]) == [1.0, 1.0]        # BrainCo: 1 = closed
    assert list(cfg.exit_hand[:, 0]) == [0.2, 0.3]


def test_no_record_and_a_relative_out(tmp_path):
    f = _deploy(tmp_path, 'robot = "g1d_normal_dex1"\n' + HOME
                + '[record]\nout = "recs"\n')
    assert deployfile.resolve(f).record_dir == tmp_path / "recs"
    assert deployfile.resolve(f, record=False).record_dir is None
    assert deployfile.resolve(f, flags={"out": "/tmp/x"}).record_dir == Path("/tmp/x")


# --- The policy's trained rates ------------------------------------------------------


def _trained(tmp_path, rates: str, extra: str = "") -> Path:
    _policy_file(tmp_path, f'''
        class P:
            anchor_camera = "head_left"
            {rates}
        ''')
    return _deploy(tmp_path, 'robot = "g1d_normal_dex1"\n'
                   '[policy]\nentry = "mypolicy.py:P"\n' + HOME + extra)


def test_trained_rates_sit_between_infra_and_the_deploy_file(tmp_path):
    f = _trained(tmp_path, """trained_action_horizon = 32
            trained_control_hz = 50.0""")
    r = deployfile.resolve(f)
    assert (r.cfg.H, r.cfg.control_hz, r.cfg.P) == (32, 50.0, 25)
    assert r.sources["action_horizon"] == "trained"
    assert r.sources["replan_period_ticks"] == "infra"
    assert not r.warnings
    r2 = deployfile.resolve(f, flags={"action_horizon": 16})
    assert r2.cfg.H == 16 and r2.sources["action_horizon"] == "cli"


def test_trained_rates_are_read_from_the_class_never_an_instance(tmp_path):
    """No construction: an __init__ that would raise is never reached."""
    f = _trained(tmp_path, """trained_action_horizon = 40
            def __init__(self, *a):
                raise RuntimeError("constructed")""")
    assert deployfile.resolve(f).cfg.H == 40


def test_a_trained_rate_of_the_wrong_type_is_refused(tmp_path):
    f = _trained(tmp_path, 'trained_action_horizon = "50"')
    with pytest.raises(ConfigError, match="trained_action_horizon"):
        deployfile.resolve(f)


def test_a_horizon_past_the_trained_one_is_refused(tmp_path):
    f = _trained(tmp_path, "trained_action_horizon = 32")
    with pytest.raises(ConfigError, match="trained_action_horizon"):
        deployfile.resolve(f, flags={"action_horizon": 40})


def test_a_control_rate_off_the_trained_one_is_a_warning(tmp_path):
    f = _trained(tmp_path, """trained_action_horizon = 50
            trained_control_hz = 30.0""")
    r = deployfile.resolve(f, flags={"control_hz": 20.0})
    assert r.cfg.control_hz == 20.0
    assert len(r.warnings) == 1 and "declares 30 Hz" in r.warnings[0]


def test_replan_period_above_the_horizon_is_refused_by_validate():
    r = deployfile.resolve(DEPLOY, flags={"replan_period": 60})
    with pytest.raises(ConfigError, match="replan"):
        validate(r.cfg, r.hw)


# --- The machine file ------------------------------------------------------------------


def test_the_machine_file_supplies_the_nic(monkeypatch, tmp_path):
    f = tmp_path / "m.toml"
    f.write_text('[network]\niface = "eth9"\n')
    m = machine.load_machine(f)
    assert m.iface == "lo0"
    assert m.sources["iface"] == "env", "ELROBOT_IFACE wins over the file"
    monkeypatch.delenv("ELROBOT_IFACE")
    assert machine.load_machine(f).iface == "eth9"


def test_the_board_address_and_login_are_the_robot_files(monkeypatch, tmp_path):
    """A fact about one robot: factory values by default, a robot file's
    `[board]` for a unit that differs, `ELROBOT_*` for one shell."""
    hw = cfgmod.load_hardware("g1d_fisheye_dex1", machine.load_machine())
    assert (hw.camera_host, hw.board_ssh_user) == ("192.168.123.164", "unitree")
    rig = tmp_path / "rig.toml"
    rig.write_text('extends = "g1d_fisheye_dex1"\n[board]\nip = "10.1.2.3"\n'
                   'ssh_user = "me"\n')
    hw = cfgmod.load_hardware(str(rig), machine.load_machine())
    assert (hw.camera_host, hw.board_ssh_user) == ("10.1.2.3", "me")
    monkeypatch.setenv("ELROBOT_ROBOT_IP", "10.9.9.9")
    assert cfgmod.load_hardware(str(rig), machine.load_machine()).camera_host == "10.9.9.9"
    rig.write_text('extends = "g1d_fisheye_dex1"\n[board]\nadress = "x"\n')
    with pytest.raises(ConfigError, match="unknown keys"):
        cfgmod.load_hardware(str(rig), machine.load_machine())


def test_a_machine_file_still_carrying_the_moved_keys_is_refused(tmp_path):
    f = tmp_path / "m.toml"
    f.write_text('robot_ip = "10.1.2.3"\n')
    with pytest.raises(ConfigError, match="belong to the robot file"):
        machine.load_machine(f)


def test_a_named_machine_file_that_is_missing_is_an_error(monkeypatch):
    monkeypatch.setenv("ELROBOT_MACHINE", "/nonexistent.toml")
    with pytest.raises(ConfigError, match="ELROBOT_MACHINE"):
        machine.load_machine()


def test_no_machine_file_means_the_defaults(monkeypatch, tmp_path):
    """The file is optional, and nothing is created for anyone."""
    monkeypatch.delenv("ELROBOT_MACHINE")
    monkeypatch.setattr(machine, "DEFAULT_PATH", tmp_path / "machine.toml")
    assert machine.machine_path() is None
    m = machine.load_machine()
    assert (m.path, m.text, m.dashboard_port, m.records) == (None, None, 8080, "")
    assert m.iface == "lo0" and m.sources["dashboard_port"] == "default"
    assert not (tmp_path / "machine.toml").exists()


def test_the_default_machine_file_is_the_project_roots(monkeypatch):
    assert machine.DEFAULT_PATH == machine.PROJECT_ROOT / "machine.toml"
    assert (machine.PROJECT_ROOT / "pyproject.toml").is_file()
    assert machine.TEMPLATE.is_file()


def test_the_template_parses_and_derives_its_cpu_table(monkeypatch):
    """The committed template states no pin table: every box derives its own."""
    m = machine.load_machine(machine.TEMPLATE)
    assert "pin" not in m.cpu


def test_auto_iface_refuses_when_no_nic_is_on_the_subnet(monkeypatch, tmp_path):
    monkeypatch.delenv("ELROBOT_IFACE")
    f = tmp_path / "m.toml"
    f.write_text('[network]\niface = "auto"\nsubnet = "203.0.113."\n')
    with pytest.raises(ConfigError, match="203.0.113"):
        machine.load_machine(f)
    assert machine.load_machine(f, resolve_iface=False).iface == ""


# --- The hand-over to P5 ------------------------------------------------------------


def test_p5_builds_the_class_the_entry_names(cfg, hw, rings):
    """The resolved entry travels in the config; P5's construction step loads
    it from there (`policy_contract.make_policy`)."""
    from elrobot_deploy.policy_contract import make_policy

    spec = f"{EXAMPLES / 'policies' / 'hold.py'}:Policy"
    # The example anchors on `head_left`: the head as every robot file declares
    # it, side_by_side, so split into two views.
    head = cfgmod.CameraSpec(8, 16, layout="side_by_side", binocular=True)
    cfg = cfgmod.replace(cfg, cameras={**cfg.cameras, "head": head},
                         primary_camera="head_left")
    pol = make_policy(cfgmod.replace(cfg, policy_entry=spec), hw)
    assert type(pol).__module__ == "examples.policies.hold"


def test_latest_uses_the_numbered_rule_for_a_policy_without_its_own_layout(tmp_path):
    """The base `Policy.resolve_checkpoint` answers None, so a class that does
    not override it gets the highest numbered step, not `run/latest`."""
    from elrobot_deploy.policy_contract import checkpoints
    from elrobot_deploy.policy_contract import Policy

    for step in (1000, 20000, 3000):
        (tmp_path / str(step)).mkdir()

    class Plain(Policy):
        action_mode = "eef"
        end_effector = "dex1"
        anchor_camera = "head_left"

        def infer(self, obs, reader):
            raise NotImplementedError

    assert checkpoints.resolve(Plain, run=str(tmp_path), step="latest") == str(tmp_path / "20000")

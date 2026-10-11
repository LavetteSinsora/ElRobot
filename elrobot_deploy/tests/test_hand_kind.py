"""`--hand`: the BrainCo hand instead of the Dex-1.

The wire facts are Unitree's own (xr_teleoperate
`teleop/robot_control/robot_hand_brainco.py`): `rt/brainco/{left,right}/cmd`
and `/state`, the same `unitree_go` MotorCmds_/MotorStates_ as the Dex-1, SIX
motors per hand, `q` normalized in [0, 1] with 0 fully OPEN.

The hand's first dimension stays one grip value — what the policy emits and
the joystick integrates — and only its units and which end is closed change.
These tests pin that the swap reaches every consumer, and that leaving the flag
off changes nothing.
"""

from __future__ import annotations


from pathlib import Path
import numpy as np
import pytest

from g1d_deploy.cli import main as entry
from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.runtime.teleop import intervention as iv
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.drivers import gripper as wire_hand
from g1d_deploy.runtime.drivers import robot_state as wire_state

DEPLOY = Path(__file__).resolve().parents[1] / "examples" / "deploy.toml"
DEX1_CMD = ("rt/dex1/left/cmd", "rt/dex1/right/cmd")
BRAINCO_CMD = ("rt/brainco/left/cmd", "rt/brainco/right/cmd")
BRAINCO_STATE = ("rt/brainco/left/state", "rt/brainco/right/state")


# --- The robot file and the swap ---------------------------------------------


def test_dex1_is_the_default_and_changes_nothing(hw):
    assert cfgmod.for_hand(hw, "dex1", tcp=hw.tcp) == hw
    assert hw.hand == "dex1" and not hw.hand_close_is_max
    assert hw.topic_hand_cmd == DEX1_CMD
    assert (hw.hand_min, hw.hand_max, hw.hand_open) == (0.0, 5.4, 5.4)


def test_brainco_swaps_topics_and_travel_and_open_is_the_other_end(hw):
    b = cfgmod.for_hand(hw, "brainco")
    assert b.topic_hand_cmd == BRAINCO_CMD
    assert b.topic_hand_state == BRAINCO_STATE
    assert (b.hand_min, b.hand_max) == (0.0, 1.0)
    assert b.hand_close_is_max and b.hand_open == 0.0
    assert hw.topic_hand_cmd == DEX1_CMD, "the original is untouched"


def test_the_brainco_robot_file_is_the_same_swap():
    b = cfgmod.load_hardware("g1d_normal_brainco")
    assert b.hand == "brainco" and b.topic_hand_cmd == BRAINCO_CMD
    assert (b.hand_min, b.hand_max) == (0.0, 1.0)


def test_an_unknown_hand_is_refused(hw, tmp_path):
    with pytest.raises(ConfigError, match="inspire"):
        cfgmod.for_hand(hw, "inspire")
    f = tmp_path / "r.toml"
    f.write_text('extends = "g1d_base"\n[end_effector]\ntype = "inspire"\n')
    from g1d_deploy.config_loading import robot_file as robot
    with pytest.raises(ConfigError, match="inspire"):
        robot.load_robot(str(f))


@pytest.mark.parametrize("tcp, match", [
    ("[0.1, 0, 0]", "7"),
    ("[0.1, 0, 0, 0, 0, 0, 1, 0]", "7"),
    ("[0.1, 0, 0, 2, 0, 0, 0]", "unit quaternion"),
    ('"x"', "7"),
])
def test_a_bad_tcp_is_refused_at_load(tmp_path, tcp, match):
    f = tmp_path / "r.toml"
    f.write_text(f'extends = "g1d_normal_dex1"\n[end_effector]\ntcp = {tcp}\n')
    with pytest.raises(ConfigError, match=match):
        cfgmod.load_hardware(str(f))


def test_a_rig_states_its_own_tcp(tmp_path):
    """A different tool on the wrist overrides the base's value in its own
    robot file; the quaternion is w first and comes back normalized."""
    f = tmp_path / "r.toml"
    f.write_text('extends = "g1d_normal_dex1"\n[end_effector]\n'
                 'tcp = [0.2, 0.01, 0.0, 0.7071068, 0.0, 0.0, 0.7071068]\n')
    tcp = cfgmod.load_hardware(str(f)).tcp
    assert tcp[:3] == pytest.approx([0.2, 0.01, 0.0])
    assert np.linalg.norm(tcp[3:]) == pytest.approx(1.0, abs=1e-12)
    b = tmp_path / "b.toml"
    b.write_text('extends = "g1d_normal_brainco"\n[end_effector]\n'
                 'tcp = [0.15, 0, 0, 1, 0, 0, 0]\n')
    assert cfgmod.load_hardware(str(b)).tcp[0] == pytest.approx(0.15)


def test_changing_type_drops_the_inherited_tcp(tmp_path):
    """A TCP belongs to its end effector: a file that changes `type` without
    stating a `tcp` does not inherit the Dex-1's jaw midpoint."""
    f = tmp_path / "r.toml"
    f.write_text('extends = "g1d_fisheye_dex1"\n[end_effector]\ntype = "brainco"\n')
    hw = cfgmod.load_hardware(str(f))
    assert hw.hand == "brainco" and hw.tcp is None
    g = tmp_path / "g.toml"
    g.write_text(f'extends = "{f}"\n[end_effector]\ntype = "brainco"\n'
                 'tcp = [0.1, 0, 0, 1, 0, 0, 0]\n')
    assert cfgmod.load_hardware(str(g)).tcp[0] == pytest.approx(0.1)
    # Restating the same type keeps the inherited TCP.
    h = tmp_path / "h.toml"
    h.write_text('extends = "g1d_fisheye_dex1"\n[end_effector]\ntype = "dex1"\n')
    assert cfgmod.load_hardware(str(h)).tcp[0] == pytest.approx(0.184759)


def test_end_effector_keys_other_than_type_and_tcp_are_refused(tmp_path):
    """Per-type specifics are built in; only `type` and `tcp` belong here."""
    from g1d_deploy.config_loading import robot_file as robot
    for body in ('[end_effector]\nhand_max = 3.0\n',
                 '[end_effector.dex1]\ntcp = [0.2, 0, 0, 1, 0, 0, 0]\n',
                 '[end_effector]\nkind = "dex1"\n'):
        f = tmp_path / "r.toml"
        f.write_text('extends = "g1d_normal_dex1"\n' + body)
        with pytest.raises(ConfigError, match="unknown keys"):
            robot.load_robot(str(f))


def test_a_rig_with_no_end_effector_is_refused(tmp_path):
    f = tmp_path / "r.toml"
    f.write_text('extends = "g1d_base"\n')
    from g1d_deploy.config_loading import robot_file as robot
    with pytest.raises(ConfigError, match="names no end effector"):
        robot.load_robot(str(f))


# --- The wire ---------------------------------------------------------------


def test_one_grip_value_drives_six_motors_as_unitrees_trigger_mode_does():
    """thumb-ish motor 0 waits for the second half of the squeeze, motor 1
    closes over the first half and stops at 0.7, the fingers follow 1:1."""
    t = wire_hand.brainco_targets
    np.testing.assert_allclose(t(0.0), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(t(0.25), [0.0, 0.5, 0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(t(0.5), [0.0, 0.7, 0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(t(0.75), [0.5, 0.7, 0.75, 0.75, 0.75, 0.75])
    np.testing.assert_allclose(t(1.0), [0.98, 0.7, 0.98, 0.98, 0.98, 0.98])


class _FakePublisher:
    def __init__(self, topic, _type):
        self.topic = topic
        self.written: list[list[float]] = []

    def Init(self):
        pass

    def Write(self, msg):
        self.written.append([float(c.q) for c in msg.cmds])


def _publisher(monkeypatch, hw):
    pytest.importorskip("cyclonedds")      # the message types are real
    from g1d_deploy.runtime.drivers.dds import channel

    monkeypatch.setattr(channel, "ChannelPublisher", _FakePublisher)
    monkeypatch.setattr(channel, "ChannelFactoryInitialize",
                        lambda *a, **k: None)
    out = wire_hand.DdsHandCmd(hw)
    out.connect()
    return out


def test_brainco_publishes_six_motors_per_hand_from_dim_zero(monkeypatch, hw):
    out = _publisher(monkeypatch, cfgmod.for_hand(hw, "brainco"))
    assert [p.topic for p in out._pubs] == list(BRAINCO_CMD)
    for msg in out._msgs:
        assert len(msg.cmds) == wire_hand.BRAINCO_MOTORS
        assert all(c.dq == wire_hand.BRAINCO_DQ and c.kp == 0.0
                   for c in msg.cmds)

    cmd = np.zeros((schema.N_HANDS, schema.HAND_DIM))
    cmd[0, 0], cmd[1, 0] = 1.0, 0.5
    cmd[:, 1:] = 9.0                    # only the grip value reaches the wire
    out.publish(cmd)

    assert out._pubs[0].written[0] == pytest.approx(
        [0.98, 0.7, 0.98, 0.98, 0.98, 0.98])
    assert out._pubs[1].written[0] == pytest.approx(
        [0.0, 0.7, 0.5, 0.5, 0.5, 0.5])


def test_dex1_still_publishes_one_jaw_with_its_gains(monkeypatch, hw):
    out = _publisher(monkeypatch, hw)
    assert [p.topic for p in out._pubs] == list(DEX1_CMD)
    for msg in out._msgs:
        assert len(msg.cmds) == 1
        assert msg.cmds[0].kp == wire_hand.GRIPPER_KP
    cmd = np.zeros((schema.N_HANDS, schema.HAND_DIM))
    cmd[:, 0] = [5.4, 1.2]
    out.publish(cmd)
    assert [p.written[0] for p in out._pubs] == [[5.4], [1.2]]


class _Sub6:
    def __init__(self, topic, _type):
        self.topic = topic
        self.q = [0.0] * 6

    def Init(self):
        pass

    def Read(self, *_a, **_k):
        motors = [type("M", (), {"q": v})() for v in self.q]
        return type("Msg", (), {"states": motors})()

    def Close(self):
        pass


def test_brainco_state_reports_the_grip_as_the_mean_of_the_four_fingers(
        monkeypatch, hw):
    """So the measured grip is comparable to the commanded one, which the
    fingers follow 1:1 — the squeeze bound compares exactly these two."""
    pytest.importorskip("cyclonedds")      # the message types are real
    from g1d_deploy.runtime.drivers.dds import channel

    monkeypatch.setattr(channel, "ChannelSubscriber", _Sub6)
    monkeypatch.setattr(channel, "ChannelFactoryInitialize",
                        lambda *a, **k: None)
    hand = wire_state.DdsHandState(cfgmod.for_hand(hw, "brainco"))
    hand.connect()
    hand._subs[0].q = [0.9, 0.7, 0.4, 0.5, 0.6, 0.7]
    hand._subs[1].q = [0.0, 0.0, 0.1, 0.1, 0.1, 0.1]

    rec = wire_state.record(hand.dtype)
    assert hand.read(rec)
    block = rec["grip"].reshape(schema.N_HANDS, schema.HAND_DIM)
    assert [s.topic for s in hand._subs] == list(BRAINCO_STATE)
    assert block[0, 0] == pytest.approx(0.55)
    assert block[1, 0] == pytest.approx(0.1)
    assert not block[:, 1:].any()


# --- The joystick -----------------------------------------------------------


def test_on_brainco_up_still_closes_and_the_squeeze_bound_is_a_ceiling():
    """1 is closed there, so closing RAISES the value and an object stopping the
    fingers sits BELOW the command — the bound must cap it from above."""
    g = iv.Gripper(3.0 / 5.4, 0.0, 1.0, squeeze_margin=0.6 / 5.4,
                   up_closes=True, close_is_max=True)
    g.engage(0.3)
    g.latch(0.0)
    assert g.step(1.0, 0.1) > 0.3, "stick up closes"
    for _ in range(50):
        cmd = g.step(1.0, 0.1, measured=0.4)       # fingers stopped by an object
    assert cmd == pytest.approx(0.4 + 0.6 / 5.4)


def test_p11_gives_the_brainco_the_dex1s_feel_in_its_own_units(rings, cfg, hw):
    from g1d_deploy.runtime.processes.p11_pico import PicoLoop

    dex = PicoLoop(rings, cfg, hw, None).grippers[0]
    brainco = cfgmod.for_hand(hw, "brainco")
    bc = PicoLoop(rings, cfg, brainco, None).grippers[0]

    assert dex.rate == cfg.intervene_gripper_rate, "exactly unchanged on Dex-1"
    assert not dex.close_is_max
    assert bc.rate == pytest.approx(cfg.intervene_gripper_rate / 5.4)
    assert bc.squeeze_margin == pytest.approx(
        cfg.intervene_squeeze_margin_rad / 5.4)
    assert bc.close_is_max and (bc.lo, bc.hi) == (0.0, 1.0)


def test_the_tcp_is_the_end_effectors_own_and_pico_does_not_need_it(rings, cfg, hw):
    """`Hardware.tcp` comes from `[end_effector] tcp`: the Dex-1's measured jaw
    midpoint, and NOT carried over to a hand that states none. The
    intervention drives the flange, so a hand without a TCP is not refused."""
    from g1d_deploy.runtime.processes.p11_pico import PicoLoop

    assert hw.tcp[:3] == pytest.approx([0.184759, 0.0, 0.0])
    bc = cfgmod.for_hand(hw, "brainco")
    assert bc.tcp is None
    PicoLoop(rings, cfg, bc, None)


# --- The deploy file and the CLI ----------------------------------------------


@pytest.mark.parametrize("argv, hand, topics, open_", [
    ([], "dex1", DEX1_CMD, 5.4),
    (["--robot", "g1d_normal_dex1"], "dex1", DEX1_CMD, 5.4),
])
def test_the_cli_carries_the_hand_to_the_wire_and_the_home_posture(
        argv, hand, topics, open_, monkeypatch):
    """`hand = "open"` in the deploy file means the chosen hand's own open end:
    5.4 rad on the Dex-1."""
    seen = {}

    class _Launcher:
        def __init__(self, cfg, hw, **kwargs):
            seen.update(cfg=cfg, hw=hw, **kwargs)

        def run(self):
            pass

    monkeypatch.setattr(entry, "Launcher", _Launcher)
    entry.main(["run", str(DEPLOY), "--no-record", *argv])

    assert seen["hw"].hand == hand
    assert seen["hw"].topic_hand_cmd == topics
    assert np.all(np.asarray(seen["cfg"].home_hand)[:, 0] == open_)
    assert np.all(np.asarray(seen["cfg"].exit_hand)[:, 0] == open_)


def test_a_policy_is_refused_on_a_brainco_rig(monkeypatch, capsys):
    """The BrainCo is not in the policy action contract yet: the run stops at
    resolution, before any process, saying so."""
    class _Launcher:
        def __init__(self, *a, **k):
            raise AssertionError("must not launch")

    monkeypatch.setattr(entry, "Launcher", _Launcher)
    with pytest.raises(SystemExit):
        entry.main(["run", str(DEPLOY), "--no-record", "--robot",
                    "g1d_normal_brainco"])
    assert "not support" in capsys.readouterr().err

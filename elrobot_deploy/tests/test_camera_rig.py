"""Robot files: which cameras exist, and which server on the board serves them.

What used to be three flags (`--camera`, `--exo`, and the experiment's own
camera table) is now one robot file per rig. Three things still have to agree,
and nothing at runtime compares them: the ports P2 connects to
(`hw.camera_ports`), the shapes the config declares (P2 refuses any other),
and whether P0 SSHes to the board to stop teleimager. These tests pin that one
robot file sets all three.
"""

from __future__ import annotations

from dataclasses import astuple
from pathlib import Path

import pytest

from g1d_deploy.cli import main as entry
from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.config_loading import robot_file as robotmod

DEPLOY = Path(__file__).resolve().parents[1] / "examples" / "deploy.toml"

FISHEYE_PORTS = {"head": 55565, "lwrist": 55566, "rwrist": 55567}


def _shape(spec):
    return astuple(spec)[:2]


# --- The shipped profiles ---------------------------------------------------


def test_the_fisheye_rig_is_ours_and_distinct_from_teleimagers(hw):
    """The default robot: our server on 55565/6/7, never the vendor's 5555x —
    running the wrong server must give nothing, not a plausible stream."""
    assert hw.camera_ports == FISHEYE_PORTS
    assert hw.image_server == "fisheye"
    normal = cfgmod.load_hardware("g1d_normal_dex1")
    assert normal.camera_ports == {"head": 55555}
    assert normal.image_server == "teleimager"
    assert normal.camera_host == hw.camera_host


@pytest.mark.parametrize("name, cameras, rig", [
    ("g1d_fisheye_dex1", {"rwrist": (720, 1280), "lwrist": (720, 1280),
                          "head": (480, 1280)}, "fisheye"),
    # The robot without the fisheye rig has NO wrist cameras — only the head
    # unit is on its USB bus, and teleimager serves only 55555. A declared wrist
    # there is a CHECK that waits forever.
    ("g1d_normal_dex1", {"head": (480, 1280)}, "teleimager"),
    ("g1d_fisheye_dex1_exo", {"rwrist": (720, 1280), "lwrist": (720, 1280),
                              "head": (480, 1280), "exo": (480, 640)}, "fisheye"),
])
def test_a_robot_file_lists_exactly_the_cameras_that_exist(name, cameras, rig):
    r = robotmod.load_robot(name)
    assert {n: _shape(s) for n, s in r.cameras.items()} == cameras
    assert r.image_server == rig


def test_the_exo_is_a_local_realsense_found_automatically():
    exo = robotmod.load_robot("g1d_fisheye_dex1_exo").cameras["exo"]
    assert exo.source == "realsense" and exo.device is None     # serial = "auto"


def test_the_head_is_declared_side_by_side_and_the_wrists_fisheye():
    cams = robotmod.load_robot("g1d_fisheye_dex1").cameras
    assert cams["head"].layout == "side_by_side" and cams["head"].binocular
    for w in ("lwrist", "rwrist"):
        assert cams[w].kind == "fisheye180"
        assert cams[w].fallback_lens_circle and cams[w].crop is None


def test_extends_merges_tables_and_the_child_wins(tmp_path):
    f = tmp_path / "mine.toml"
    f.write_text('extends = "g1d_normal_dex1"\n'
                 '[end_effector]\ntype = "brainco"\n'
                 '[cameras.head]\nport = 60000\n'
                 '[cameras.table]\nsource = "v4l2"\n'
                 'device = "/dev/v4l/by-path/x"\nshape = [480, 640]\n'
                 '[cameras.table.intrinsics]\nfx = 1.0\n')
    r = robotmod.load_robot(str(f))
    assert r.end_effector == "brainco"
    assert r.cameras["head"].port == 60000
    assert r.cameras["head"].layout == "side_by_side", "inherited, not replaced"
    assert r.cameras["table"].intrinsics == {"fx": 1.0}
    assert [Path(x.path).stem for x in r.files] == ["g1d_base", "g1d_normal_dex1", "mine"]


@pytest.mark.parametrize("table, match", [
    ('source = "zmq"\nshape = [8, 8]', "port"),
    ('source = "zmq"\nport = 1\nshape = [8, 8]\ncrop_defualt = [0, 0, 8, 8]', "crop_defualt"),
    ('source = "usb"\nshape = [8, 8]', "source"),
    ('source = "v4l2"\ndevice = "auto"\nshape = [8, 8]', "by-path"),
    ('source = "zmq"\nport = 1\nshape = [8, 8]\ncrop_default = [0, 0, 8, 8]', "crop_default"),
])
def test_a_wrong_camera_table_is_refused_by_name(table, match, tmp_path):
    f = tmp_path / "bad.toml"
    f.write_text(f'extends = "g1d_base"\n[cameras.c]\n{table}\n')
    with pytest.raises(ConfigError, match=match):
        robotmod.load_robot(str(f))


def test_an_unknown_profile_name_lists_the_shipped_ones():
    with pytest.raises(ConfigError, match="g1d_fisheye_dex1"):
        robotmod.load_robot("g1d_fisheye")


# --- The CLI carries the robot file to all three -----------------------------


@pytest.mark.parametrize("argv, rig, ports, cameras", [
    (["--robot", "g1d_normal_dex1"], "teleimager", {"head": 55555}, ["head"]),
    ([], "fisheye", FISHEYE_PORTS, ["head", "lwrist", "rwrist"]),
])
def test_the_cli_sets_ports_cameras_and_board_handling_together(
        argv, rig, ports, cameras, monkeypatch):
    seen = {}

    class _Launcher:
        def __init__(self, cfg, hw, **kwargs):
            seen.update(cfg=cfg, hw=hw, **kwargs)

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr(entry, "Launcher", _Launcher)
    entry.main(["run", str(DEPLOY), "--no-record", *argv])

    assert seen["ran"]
    assert seen["hw"].image_server == rig           # P0: SSH to the board or not
    assert seen["hw"].camera_ports == ports         # P2: which server
    assert sorted(seen["cfg"].cameras) == cameras   # P2: which cameras exist


def test_a_realsense_camera_without_pyrealsense2_is_refused_by_name(monkeypatch, capsys):
    """pyrealsense2 is the `realsense` extra, imported only when P2 opens the
    camera; a run that declares one without it is refused before anything
    starts, with the command that installs it."""
    monkeypatch.setattr(cfgmod, "_importable", lambda m: m != "pyrealsense2")
    with pytest.raises(SystemExit):
        entry.main(["run", str(DEPLOY), "--no-record",
                    "--robot", "g1d_fisheye_dex1_exo"])
    err = capsys.readouterr().err
    assert "'exo'" in err and "uv sync --extra runtime --extra realsense" in err
    assert cfgmod.missing_camera_driver("head", "zmq") is None


def test_a_robot_file_uncrosses_the_wrists_by_exchanging_their_ports(tmp_path):
    """The two wrist cameras cannot be told apart in software; a unit whose
    cables are crossed gets a robot file that exchanges the ports."""
    base = cfgmod.load_hardware("g1d_fisheye_dex1")
    rig = tmp_path / "rig.toml"
    rig.write_text('extends = "g1d_fisheye_dex1"\n'
                   f'[cameras.lwrist]\nport = {base.camera_ports["rwrist"]}\n'
                   f'[cameras.rwrist]\nport = {base.camera_ports["lwrist"]}\n')
    hw = cfgmod.load_hardware(str(rig))
    assert hw.camera_ports["lwrist"] == base.camera_ports["rwrist"]
    assert hw.camera_ports["rwrist"] == base.camera_ports["lwrist"]
    assert hw.camera_ports["head"] == base.camera_ports["head"]


def test_a_zmq_camera_may_name_its_own_host(hw, cfg):
    from g1d_deploy.runtime.processes import p2_camera

    spec = cfgmod.CameraSpec(8, 8, source="zmq", host="10.0.0.9")
    c = cfgmod.replace(cfg, cameras={"rwrist": spec})
    src = p2_camera.make_source(c, hw, "rwrist")
    assert src.host == "10.0.0.9" and src.port == hw.camera_ports["rwrist"]
    plain = cfgmod.replace(cfg, cameras={"rwrist": cfgmod.CameraSpec(8, 8)})
    assert p2_camera.make_source(plain, hw, "rwrist").host == hw.camera_host

"""The robot a record is drawn as comes from the record's own robot file text.

P9 copies the robot file chain into every manifest verbatim; inspection reads
the joints, the hand's travel, the virtual TCP and the URDF from there, not
from whatever the named profile (or this machine's default) says today.
"""

from __future__ import annotations

import numpy as np
import pytest

from elrobot_deploy.dashboards.inspect import loader, record_robot, scene, timeline

from . import inspect_fixtures as fx

pytest.importorskip("mujoco")

#: A TCP no shipped profile has: 20 cm out and 1 cm up, rotated.
_TCP = [0.2, 0.0, 0.01, 0.7071068, 0.0, 0.7071068, 0.0]


def _custom(body: str):
    """A manifest edit: a custom robot file extending the recorded profile,
    appended to the recorded chain the way P9 records it."""
    def edit(m):
        m["robot"]["files"].append({"name": "my_rig.toml", "shipped": False,
                                    "path": "/rig/my_rig.toml",
                                    "text": 'extends = "g1d_fisheye_dex1"\n' + body})
    return edit


_with_tcp = _custom("[end_effector]\ntcp = [" + ", ".join(map(str, _TCP)) + "]\n")


def test_the_robot_is_read_from_the_manifests_robot_file(tmp_path):
    run = fx.write(tmp_path / "r")
    m = loader.load(run).manifest
    assert m["robot"]["shipped"] and m["robot"]["model"]["shipped"]
    assert m["robot"]["extends"] == ["g1d_base", "g1d_fisheye_dex1"]
    rr = record_robot.from_manifest(m)
    assert rr.source == "record"
    assert rr.name == "g1d_fisheye_dex1"
    assert (rr.hand_min, rr.hand_max) == (0.0, 5.4)
    assert rr.joints_arm == list(range(15, 29))
    assert rr.urdf_source == "package" and rr.urdf.endswith("g1_d.urdf")


def test_the_virtual_tcp_comes_from_the_robot_file(tmp_path):
    """`[end_effector] tcp` (flange frame) moves every TCP marker: the
    measured one sits exactly at that offset from the wrist."""
    import mujoco

    run = fx.write(tmp_path / "r", manifest_edit=_with_tcp)
    ro = loader.load(run)
    try:
        assert ro.model.tcp_source == "record"
        assert np.allclose(ro.model.tcp[:3], _TCP[:3])
        tl = timeline.build(ro)
        sc = scene.build(ro, tl)
        model, data = scene.build_model(ro.model.urdf)
        jmap = scene._JointMap(model, ro.model)
        q = tl.q[0]
        data.qpos[:] = 0.0
        data.qpos[jmap.arm_adr] = q[jmap.arm_cols]
        data.qpos[jmap.waist_adr] = q[jmap.waist_cols]
        mujoco.mj_forward(model, data)
        for h, bid in enumerate(jmap.wrist_body):
            local = data.xmat[bid].reshape(3, 3).T @ (sc.tcp[0, h] - data.xpos[bid])
            assert np.allclose(local, _TCP[:3], atol=1e-5)
        assert ro.summary()["tcp_source"] == "record"
    finally:
        ro.close()


def test_a_hand_with_no_tcp_draws_no_tcp_marker(tmp_path):
    """A robot file that changes the end effector type without a TCP (the
    BrainCo: nobody has measured one) is not given the Dex-1's: `tcp` is
    None, and the page is told to draw no marker."""
    edit = _custom('[end_effector]\ntype = "brainco"\n')
    ro = loader.load(fx.write(tmp_path / "r", manifest_edit=edit))
    try:
        assert ro.model.hand == "brainco"
        assert ro.model.tcp is None and ro.model.tcp_source == "none"
        assert ro.summary()["tcp"] is None
        sc = scene.build(ro, timeline.build(ro))
        assert sc.meta()["tcp"] is False
        assert np.allclose(scene.tcp_point(ro.model), 0.0)
    finally:
        ro.close()


def test_a_record_without_a_robot_file_is_unsupported(tmp_path):
    run = fx.write(tmp_path / "r", manifest_edit=lambda m: m.pop("robot"))
    with pytest.raises(record_robot.UnsupportedRecord, match="unsupported"):
        loader.load(run)


def test_an_unparseable_robot_file_is_unsupported(tmp_path):
    run = fx.write(tmp_path / "r", robot_edit=lambda t: t + "\n[[[ not toml")
    with pytest.raises(record_robot.UnsupportedRecord, match="does not parse"):
        loader.load(run)


def test_a_custom_urdf_is_unsupported(tmp_path):
    """A robot whose model is not the package's own cannot be drawn here: the
    record says so instead of guessing."""
    edit = _custom('[model]\nurdf = "/home/someone/my_robot.urdf"\n')
    with pytest.raises(record_robot.UnsupportedRecord,
                       match=r"custom URDF\) isn't available here"):
        loader.load(fx.write(tmp_path / "r", manifest_edit=edit))


def test_a_record_opens_anywhere(tmp_path):
    """Copied to another machine: every path the recording box wrote is gone,
    and the record still opens — nothing in it has to exist here."""
    import json
    import shutil

    src = fx.write(tmp_path / "box" / "r")
    dst = tmp_path / "Downloads" / "r"
    shutil.copytree(src, dst)
    shutil.rmtree(tmp_path / "box")
    m = json.loads((dst / "manifest.json").read_text())
    for f in m["robot"]["files"]:
        f["path"] = "/nowhere/" + f["name"]
    (dst / "manifest.json").write_text(json.dumps(m))
    ro = loader.load(dst)
    try:
        assert ro.model.urdf_source == "package"
        assert m["model"]["urdf"] == "assets/unitree_g1d/g1_d.urdf"
    finally:
        ro.close()


def test_hardware_is_built_offline_from_the_record(tmp_path):
    """The replay constructs a policy with the recorded robot, on a machine
    with no NIC: no refusal about DDS, the record's own CPU table."""
    run = fx.write(tmp_path / "r")
    hw = record_robot.hardware(loader.load(run).manifest)
    assert (hw.hand_min, hw.hand_max) == (0.0, 5.4)
    assert hw.dds_iface == ""
    assert list(hw.joints_arm) == list(range(15, 29))

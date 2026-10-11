"""`Policy.cameras`: what the infra knows about each camera view — size, rate,
lens kind, the fisheye crop this run decided, calibration — bound before
`warmup()`. A fisheye view is already the square around its lens circle
(the infra cut it); `crop` says where on the sensor it sits."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from g1d_deploy.config_loading import deploy_file
from g1d_deploy.policy_contract import (CameraInfo, Policy, bind_runtime,
                                        camera_info)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _cfg():
    return deploy_file.resolve(EXAMPLES / "policies" / "sine_eef.toml",
                               record=False).cfg


def test_every_view_is_described_and_the_head_halves_know_their_columns():
    info = camera_info(_cfg())
    assert set(info) == {"lwrist", "rwrist", "head_left", "head_right"}
    left, right = info["head_left"], info["head_right"]
    assert (left.source, left.x0, left.width) == ("head", 0, 640)
    assert (right.source, right.x0) == ("head", 640)
    assert left.kind == "normal" and left.crop is None


def test_without_a_fit_a_fisheye_gets_the_robot_files_circle():
    from g1d_deploy.runtime.drivers.fisheye_circle_fit import crop_for

    cfg = _cfg()
    rw = camera_info(cfg)["rwrist"]
    assert rw.kind == "fisheye180" and rw.crop_source == "fallback_lens_circle"
    assert rw.crop == crop_for(*cfg.cameras["rwrist"].fallback_lens_circle)
    assert rw.height == rw.width == rw.crop[2] - rw.crop[0]
    assert rw.lens_circle is not None


def test_the_launch_fit_wins():
    cfg = _cfg()
    fitted = replace(cfg.cameras["rwrist"], crop=(1, 2, 11, 12), lens_circle=(6.0, 7.0, 3.0))
    cfg = replace(cfg, cameras={**cfg.cameras, "rwrist": fitted})
    rw = camera_info(cfg)["rwrist"]
    assert rw.crop == (1, 2, 11, 12) and rw.crop_source == "fitted"
    assert (rw.height, rw.width) == (10, 10) and rw.lens_circle == (6.0, 7.0, 3.0)


def test_bound_before_warmup_and_empty_before():
    class P(Policy):
        action_mode, end_effector, anchor_camera = "eef", "dex1", "rwrist"

        def infer(self, obs, reader):
            raise AssertionError("not called")

    p = P(None, "x", None)
    assert p.cameras == {}
    bind_runtime(p, None, None, camera_info(_cfg()))
    assert isinstance(p.cameras["rwrist"], CameraInfo)

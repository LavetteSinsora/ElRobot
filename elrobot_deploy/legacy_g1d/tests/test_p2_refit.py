"""The fisheye lens fit at every launch (`runtime/drivers/fisheye_circle_fit.py`,
`Launcher._fit_lenses`).

A stored lens fit was once measured stale — 53 px out in cy,
which had every rollout serving a crop with the fisheye disc clipped at the
top. The fit decides the circle and so the square; the robot file's optional
`fallback_lens_circle` is a FALLBACK, used explicitly and with a warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.runtime.drivers import fisheye_circle_fit as cr
from elrobot_deploy.config_loading import robot_file as robotmod


def a_spec(cx=648.0, cy=386.0, r=375.0, *, fallback=True):
    return cfgmod.CameraSpec(
        720, 1280, source="zmq", kind="fisheye180",
        fallback_lens_circle=(cx, cy, r) if fallback else None)


def disc_frames(cx, cy, r, *, n=10, h=720, w=1280, moving=True):
    """A lit disc with a dark bar across it — the gripper's stand-in."""
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    out = []
    for i in range(n):
        f = np.zeros((h, w), np.uint8)
        f[disc] = 180
        x0 = int(cx - r + (2 * r - 120) * (i / max(n - 1, 1))) if moving else int(cx - 60)
        f[:, max(x0, 0):x0 + 120] = 0
        out.append(f)
    return out


# --- The crop rule ---------------------------------------------------------

def test_crop_is_square_and_centred_on_the_circle():
    x0, y0, x1, y1 = cr.crop_for(648.0, 386.0, 375.0)
    assert x1 - x0 == y1 - y0 == 2 * (375 + cr.CROP_MARGIN_PX)
    assert (x0 + x1) / 2 == pytest.approx(648.0, abs=1)
    assert (y0 + y1) / 2 == pytest.approx(386.0, abs=1)


def test_the_crop_may_run_off_the_sensor():
    """It has to. These circles overflow the top of a 720-row sensor, and
    clamping would put the two wrists' discs at different scales."""
    assert cr.crop_for(648.0, 330.0, 375.0)[1] < 0


@pytest.mark.parametrize("name", ["lwrist", "rwrist"])
def test_the_robot_file_gives_only_a_fallback_circle(name):
    spec = robotmod.load_robot("g1d_fisheye_dex1").cameras[name]
    assert spec.kind == "fisheye180" and spec.fallback_lens_circle is not None
    assert spec.crop is None, "the square is the launch's fit, never the file's"


# --- The decision ------------------------------------------------------------

def test_a_good_fit_decides_the_square():
    d = cr.decide("lwrist", a_spec(648.0, 386.0, 375.0),
                  disc_frames(660.0, 330.0, 372.0))       # the camera drifted
    assert d.used == "fit" and not d.warning
    cx, cy, r = d.lens_circle
    assert (cx, cy, r) == pytest.approx((660.0, 330.0, 372.0), abs=4)
    assert d.crop == cr.crop_for(cx, cy, r), "centre AND size from the fit"


def test_it_catches_the_drift_that_was_actually_missed():
    """REGRESSION for the real event: cy 53 px out, radius fine."""
    d = cr.decide("lwrist", a_spec(648.5, 386.5, 375.0),
                  disc_frames(662.9, 333.5, 374.2))
    assert d.used == "fit"
    assert d.drift[1] == pytest.approx(-53.0, abs=6), "should see the 53 px"
    assert abs(d.drift[2]) < 6, "and should NOT invent a radius change"


def test_without_fallback_lens_circle_the_fit_stands_alone():
    d = cr.decide("lwrist", a_spec(648.0, 386.0, 375.0, fallback=False),
                  disc_frames(650.0, 380.0, 370.0))
    assert d.used == "fit" and d.drift is None


def test_a_refused_fit_falls_back_to_the_default_and_says_so():
    spec = a_spec()
    dark = [np.zeros((720, 1280), np.uint8) for _ in range(8)]
    d = cr.decide("lwrist", spec, dark)
    assert d.used == "fallback_lens_circle" and d.warning
    assert d.crop == cr.crop_for(*spec.fallback_lens_circle)
    assert d.why and d.line().startswith("WARNING")


def test_a_plausible_circle_that_drifted_too_far_is_refused():
    """The drift guard's own case: a circle `fit_circle` is happy with, but far
    from where this camera's lens was. It has to be the RADIUS that moved: the
    fitter's centre prior catches a moved centre before this guard sees it."""
    d = cr.decide("lwrist", a_spec(648.0, 386.0, 375.0),
                  disc_frames(648.0, 386.0, 200.0))       # 175 px smaller
    assert d.used == "fallback_lens_circle"
    assert "cables" in d.why or "covered" in d.why
    assert d.fit is not None, "the refused fit is still reported"


def test_a_wildly_distant_fit_is_refused_rather_than_used():
    spec = a_spec(648.0, 386.0, 375.0)
    d = cr.decide("lwrist", spec, disc_frames(420.0, 386.0, 370.0))
    assert d.used == "fallback_lens_circle" and d.crop == cr.crop_for(*spec.fallback_lens_circle)


def test_with_a_fallback_decide_never_raises():
    """A launch is not stopped by a camera that serves fine on its known circle."""
    spec = a_spec()
    for bad in ([], [object()], "not frames"):
        d = cr.decide("lwrist", spec, bad)
        assert d.used == "fallback_lens_circle"


def test_the_line_says_what_it_did():
    d = cr.decide("lwrist", a_spec(), disc_frames(650.0, 380.0, 370.0))
    line = d.line()
    assert "lens lwrist" in line and "from fallback_lens_circle" in line and "resid" in line


def test_the_decision_is_json_able_for_the_manifest():
    import json

    d = cr.decide("lwrist", a_spec(), disc_frames(650.0, 380.0, 370.0))
    back = json.loads(json.dumps(d.as_dict()))
    assert back["used"] == "fit" and len(back["crop"]) == 4


def test_luma_is_rec601_over_rgb_from_a_bgr_frame():
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 2] = 200                             # red, in BGR order
    assert cr.luma(bgr)[0, 0] == int(0.299 * 200)


# --- The INIT hook -------------------------------------------------------------


def _launcher_stub(cfg, frames):
    from elrobot_deploy.runtime.launcher import Launcher

    lch = Launcher.__new__(Launcher)
    lch.cfg = cfg
    lch.hw = None
    lch.crops = []
    lch.logged = []
    lch.log = lch.logged.append
    lch._lens_frames = lambda cam: frames[cam]
    return lch


def test_init_fits_only_fisheye_cameras_and_hands_the_square_on(cfg):
    fish = a_spec(648.0, 386.0, 375.0)
    head = cfgmod.CameraSpec(480, 1280, source="zmq", layout="side_by_side")
    c = cfgmod.replace(cfg, cameras={"lwrist": fish, "head": head},
                       primary_camera="lwrist")
    lch = _launcher_stub(c, {"lwrist": disc_frames(655.0, 380.0, 372.0)})
    lch._fit_lenses()

    assert [d.camera for d in lch.crops] == ["lwrist"]
    assert lch.cfg.cameras["lwrist"].crop == lch.crops[0].crop
    assert lch.cfg.cameras["lwrist"].lens_circle == lch.crops[0].lens_circle
    assert lch.cfg.cameras["head"].crop is None, "rectilinear: the whole frame"
    assert any("lens lwrist" in m for m in lch.logged)


def test_init_falls_back_explicitly_with_a_warning(cfg):
    fish = a_spec()
    c = cfgmod.replace(cfg, cameras={"lwrist": fish}, primary_camera="lwrist")
    lch = _launcher_stub(c, {"lwrist": [np.zeros((720, 1280), np.uint8)] * 4})
    lch._fit_lenses()
    assert lch.cfg.cameras["lwrist"].crop == cr.crop_for(*fish.fallback_lens_circle)
    assert any("WARNING" in m and "fallback_lens_circle" in m for m in lch.logged)

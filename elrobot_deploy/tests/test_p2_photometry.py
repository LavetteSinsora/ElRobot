"""The exposure servo's law, tested without a camera.

Everything here is `photometry.step` and its helpers, which are pure by
construction. The integration into the publisher's loop is tested separately in
`test_p2_camera.py` against a fake capture; the one thing that genuinely needs
the robot is whether the driver accepts a mid-stream write, and that was
measured (it does, read-back exact, settling 2 frames).
"""

from __future__ import annotations

import math

import pytest

from g1d_deploy.runtime.drivers.board.fisheye_image_server import photometry as ph


def meas(p995, *, mean=140.0, clip=0.0, pixels=90_000):
    return ph.Measurement(p995=p995, mean=mean, clip_frac=clip, pixels=pixels)


def true_gain(g):
    """The TOY CAMERA's gain curve — the measured one, 0/32/64 -> 1.0/2.50/4.68.

    Deliberately NOT the servo's `1 + g/21` model, so every convergence test
    below runs the loop against a plant it models wrongly. That is the real
    case: the servo's model is a rough fit and is meant to be.
    """
    return 1.0 + 2.05 * (g / 32.0) ** 1.15


#: The toy camera's tone exponent, `p995 ~ true_light ** k`, unless a test says
#: otherwise. 1.1 is what a 10bb:2b08 camera measured live
#: 2026-09-23 (see `ph.PLANT_K_MAX`). The toy used to use the servo's own
#: gamma-2.2 exponent, so every test here ran against a plant that agreed with
#: the controller's assumption, and the overshoot that flickered the box could
#: not show up.
K_LIVE = 1.1

#: How much steeper the toy's gain curve is than the servo's `gain_multiplier`,
#: at worst, in log-log slope. The loop sees `k` times this, so it is what
#: turns a tone exponent into the exponent `ph.PLANT_K_MAX` has to bound.
TOY_SLOPE_MAX = max(
    (math.log(true_gain(g + 1)) - math.log(true_gain(g)))
    / (math.log(ph.gain_multiplier(g + 1)) - math.log(ph.gain_multiplier(g)))
    for g in range(ph.GAIN_MAX))

#: The whole range the law claims to handle, ends included: from a gamma-2.2
#: camera to the steepest tone curve whose EFFECTIVE exponent, through the
#: toy's gain-curve error, is still inside `ph.PLANT_K_MAX`.
K_RANGE = (ph.PLANT_K_MIN, 0.7, 1.0, K_LIVE, ph.PLANT_K_MAX / TOY_SLOPE_MAX)


def luma_of(state, scene_light, k=K_LIVE):
    """What the toy camera reports at this operating point."""
    light = state.exposure * true_gain(state.gain) * scene_light
    return min(255.0, 255.0 * (light ** k))


#: Scene brightnesses the rig can actually hold the target at, derived rather
#: than guessed: for every `k` in `K_RANGE` the servable band is ~2.9e-4 ..
#: 9.1e-3 (below is a room no exposure can fix, above is one where even 10 ms
#: at gain 0 overexposes).
SCENE_DIM, SCENE_MID, SCENE_BRIGHT = 1e-3, 3e-3, 7e-3


def converge(start, scene_light, *, ticks=40, target=ph.TARGET_P995,
             k=K_LIVE):
    """Run the loop against the toy camera for `ticks` ticks."""
    st = start
    seen = []
    for _ in range(ticks):
        luma = luma_of(st, scene_light, k)
        clip = 0.35 if luma >= 255.0 else 0.0
        m = meas(luma, clip=clip)
        seen.append((st, m))
        st = ph.step(m, st, target=target)
    return st, seen


# --- The setpoint and the knee ---------------------------------------------

def test_target_is_below_the_measured_clipping_knee():
    """245 is not a taste. Corpus median p99.5 is 243/249 at 0.03%/0.28%
    clipping, and the knee where clipping takes off is 250."""
    assert 240 <= ph.TARGET_P995 <= 250


def test_bands_cannot_drop_the_frame_rate():
    """Past 330 units (33 ms) the sensor outputs 1/exposure instead of 30 fps."""
    assert max(ph.EXPOSURE_BANDS) <= 330
    assert ph.PREFERRED_BAND in ph.EXPOSURE_BANDS


# --- The law ---------------------------------------------------------------

def test_deadband_holds_and_writes_nothing():
    st = ph.initial_state()
    out = ph.step(meas(ph.TARGET_P995), st)
    assert (out.exposure, out.gain) == (st.exposure, st.gain)
    assert out.held == st.held + 1


def test_too_bright_reduces_light():
    st = ph.initial_state()
    out = ph.step(meas(252.0, clip=0.01), st)
    assert out.light < st.light


def test_too_dark_raises_light():
    st = ph.initial_state()
    out = ph.step(meas(150.0), st)
    assert out.light > st.light


def test_a_pegged_highlight_cuts_blind():
    """THE failure this law exists to avoid.

    Once `p99.5` is on the rail the measurement saturates: 1.1x over and 5x
    over both read 255, so the ratio law asks for the SAME small correction
    either way and has no idea how bad it is. `PANIC_CUT` replaces a measured
    correction with a fixed one precisely because there is nothing to measure.
    """
    st = ph.initial_state()
    naive = (ph.TARGET_P995 / 255.0) ** (1.0 / ph.PLANT_K_MIN)
    assert 0.9 < naive < 1.0, (
        "the premise: however far over the camera is, a pegged highlight asks "
        f"for only a {(1 - naive) * 100:.0f}% cut")
    assert naive > ph.PANIC_CUT, "the panic cut must be the harder of the two"

    out = ph.step(meas(255.0, clip=0.35), st)
    assert out.saturated
    assert out.light < st.light * naive, "must cut harder than the ratio law would"


def test_pegged_highlight_keeps_cutting_until_it_comes_off_the_rail():
    st = ph.initial_state(band=300, gain=128)
    for _ in range(12):
        st = ph.step(meas(255.0, clip=0.30), st)
    assert st.light < ph.light_index(300, 128) * 0.25


def test_step_size_is_bounded():
    """A huge demand must not be answered in one jump.

    The bound is on what the loop ASKS for; what it lands on is then quantised
    to a (band, gain) pair, which can overshoot the request slightly — 1295
    asked for, 1300 available at 30 ms / gain 70. A few percent of quantisation
    slack is allowed; a 3x jump is not.
    """
    st = ph.initial_state()
    out = ph.step(meas(20.0), st)          # absurdly dark -> huge demand
    assert out.light <= st.light * ph.MAX_STEP * 1.05


def test_a_broken_measurement_holds_rather_than_servoing():
    """A circle that measured almost nothing is a blocked lens or a bad refit,
    not a dark room. Servoing on it would drive the camera to maximum gain."""
    st = ph.initial_state()
    out = ph.step(ph.Measurement(p995=3.0, mean=1.0, clip_frac=0.0, pixels=12), st)
    assert (out.exposure, out.gain) == (st.exposure, st.gain)
    assert not out.dark


# --- band selection --------------------------------------------------------

def test_prefers_the_preferred_band_when_it_can():
    light = ph.light_index(ph.PREFERRED_BAND, 64)
    band, gain = ph._split(light, ph.PREFERRED_BAND)
    assert band == ph.PREFERRED_BAND
    assert gain == pytest.approx(64, abs=1)


def test_leaves_the_band_when_gain_would_leave_its_range():
    bright = ph.light_index(ph.PREFERRED_BAND, 0) * 0.5      # very bright room
    band, gain = ph._split(bright, ph.PREFERRED_BAND)
    assert band < ph.PREFERRED_BAND or gain <= ph.GAIN_COMFORT_LO

    dim = ph.light_index(ph.PREFERRED_BAND, ph.GAIN_MAX) * 1.4
    band, gain = ph._split(dim, ph.PREFERRED_BAND)
    assert band > ph.PREFERRED_BAND or gain >= ph.GAIN_COMFORT_HI


def test_hysteresis_stops_boundary_flapping():
    """A scene sitting exactly where two bands meet must not switch every tick;
    each switch is a visible brightness step in the recording."""
    on_edge = ph.light_index(ph.PREFERRED_BAND, ph.GAIN_COMFORT_HI + 4)
    band_a, _ = ph._split(on_edge, ph.PREFERRED_BAND)
    band_b, _ = ph._split(on_edge, band_a)
    band_c, _ = ph._split(on_edge, band_b)
    assert band_b == band_c, "band must settle, not alternate"


def test_gain_never_leaves_the_register_range():
    for light in (1e-3, 1.0, 50.0, 1e6):
        for cur in ph.EXPOSURE_BANDS:
            band, gain = ph._split(light, cur)
            assert band in ph.EXPOSURE_BANDS
            assert ph.GAIN_MIN <= gain <= ph.GAIN_MAX


# --- convergence, under a deliberately wrong plant model -------------------

@pytest.mark.parametrize("k", K_RANGE)
@pytest.mark.parametrize("scene", [SCENE_DIM, SCENE_MID, SCENE_BRIGHT])
def test_converges_from_a_wrong_plant_model(scene, k):
    st, _ = converge(ph.initial_state(), scene, k=k)
    luma = luma_of(st, scene, k)
    assert abs(luma - ph.TARGET_P995) < 12, f"settled at {luma:.0f}"


@pytest.mark.parametrize("k", K_RANGE)
def test_settles_on_the_safe_side_of_the_knee(k):
    """Converged is not enough — it has to converge BELOW 250, where the
    measured clipping knee is. The asymmetric deadband is what buys this."""
    for scene in (SCENE_DIM, SCENE_MID, SCENE_BRIGHT):
        st, _ = converge(ph.initial_state(), scene, k=k)
        assert luma_of(st, scene, k) < 250.0, f"settled hot at scene {scene:g}"


@pytest.mark.parametrize("k", K_RANGE)
@pytest.mark.parametrize("scene", [3.5e-4, 1e-3, 3e-3, 7e-3, 8.5e-3])
@pytest.mark.parametrize("start_gain", [0, 30, 64])
def test_a_climb_never_overshoots_into_the_clip(scene, k, start_gain):
    """THE property the flicker violated. Approaching from below, no tick may
    measure past the knee — for ANY plant exponent inside the law's bound.

    Clipping is the one irreversible failure, and near a flat highlight one
    overshoot is not a small error: it trips `PANIC_CUT`, and a loop that
    overshoots once will do it again on the next climb. Sizing climbs for
    `PLANT_K_MAX` is what makes the approach monotone.
    """
    start = ph.ServoState(exposure=100, gain=start_gain)
    if luma_of(start, scene, k) >= ph.TARGET_P995:
        pytest.skip("this start is already at or above the target")
    _, seen = converge(start, scene, k=k, ticks=40)
    hottest = max(m.p995 for _, m in seen)
    assert hottest < 250.0, f"k={k:g}: a climb overshot to {hottest:.0f}"


def test_converges_quickly():
    """At 10 Hz, "quickly" has to mean a handful of ticks, not a minute."""
    st, _ = converge(ph.initial_state(), SCENE_MID, ticks=10)
    assert abs(luma_of(st, SCENE_MID) - ph.TARGET_P995) < 12


def test_recovers_from_the_blown_state_the_robot_was_actually_in():
    """Starting where `photometry: auto` left the robot — 30 ms, gain 64,
    p99.5 pegged, 26% clipped — the loop must come down, not sit there."""
    st = ph.ServoState(exposure=300, gain=64)
    assert luma_of(st, SCENE_MID) >= 254, "the premise: this start IS blown"
    st, seen = converge(st, SCENE_MID, ticks=25)
    assert seen[0][1].p995 >= 254
    assert luma_of(st, SCENE_MID) < 250
    assert seen[-1][1].clip_frac == 0.0


def test_does_not_oscillate_once_settled():
    st, _ = converge(ph.initial_state(), SCENE_MID, ticks=30)
    _, seen = converge(st, SCENE_MID, ticks=10)
    lights = {(s.exposure, s.gain) for s, _ in seen}
    assert len(lights) <= 2, f"settled loop still moving through {lights}"


# --- The dark end ----------------------------------------------------------

def test_dark_room_is_flagged_not_silently_accepted():
    st = ph.ServoState(exposure=max(ph.EXPOSURE_BANDS), gain=ph.GAIN_MAX)
    out = ph.step(meas(120.0), st)
    assert out.dark
    verdict, message = ph.dark_verdict(meas(120.0), out)
    assert verdict == "dark"
    assert "light" in message.lower()


def test_noisy_is_distinguished_from_dark():
    st = ph.ServoState(exposure=300, gain=ph.NOISY_GAIN)
    verdict, message = ph.dark_verdict(meas(246.0), st)
    assert verdict == "noisy"
    assert message


def test_a_good_operating_point_passes():
    st = ph.initial_state()
    verdict, message = ph.dark_verdict(meas(246.0, clip=0.002), st)
    assert verdict == "ok" and message == ""


# --- measurement -----------------------------------------------------------

def test_measure_reads_only_inside_the_circle():
    np = pytest.importorskip("numpy")
    img = np.zeros((360, 640), np.uint8)
    m = ph.circle_mask(img.shape, cx=640.0, cy=360.0, r=300.0, scale=2)
    img[m] = 200
    img[~m] = 255                      # a blown surround must not be counted
    got = ph.measure(img, m)
    assert got.p995 == pytest.approx(200, abs=1)
    assert got.clip_frac == 0.0


def test_circle_mask_scales_with_the_decode():
    np = pytest.importorskip("numpy")
    full = ph.circle_mask((720, 1280), 648.0, 386.0, 375.0, scale=1)
    half = ph.circle_mask((360, 640), 648.0, 386.0, 375.0, scale=2)
    assert half.sum() == pytest.approx(full.sum() / 4, rel=0.02)


def test_measure_reports_an_empty_circle_as_unusable():
    np = pytest.importorskip("numpy")
    img = np.zeros((90, 160), np.uint8)
    empty = np.zeros_like(img, bool)
    assert not ph.measure(img, empty).usable


# --- The limit cycle, pinned by name ---------------------------------------

@pytest.mark.parametrize("k", K_RANGE)
def test_does_not_limit_cycle_between_clipped_and_dark(k):
    """REGRESSION, twice over: ~208 <-> 255 on the robot, 190 <-> 255 on a
    second rig, both a climb sized for a flatter plant than the real one.

    The mechanism is worth keeping written down because it is a property of
    the scene, not a tuning accident: the highlight is a big flat wall, so the
    recorded frames show clipping going 0.02% -> 26.6% within a few luma
    levels. A step that overshoots the target overshoots into that, trips
    `PANIC_CUT`, drops far below, climbs, and overshoots again.
    """
    _, seen = converge(ph.initial_state(), SCENE_MID, ticks=60, k=k)
    tail = [luma_of(s, SCENE_MID, k) for s, _ in seen[-20:]]
    assert max(tail) - min(tail) < 15, (
        f"oscillating over {min(tail):.0f}..{max(tail):.0f}")
    assert not any(l >= 254 for l in tail), "still touching the rail once settled"


@pytest.mark.parametrize("scene", [3.5e-4, 6e-4, 1e-3, 2e-3, 3e-3,
                                   5e-3, 7e-3, 8.5e-3])
def test_every_servable_room_settles_below_the_knee(scene):
    """Across the whole servable range, not just three hand-picked points."""
    _, seen = converge(ph.initial_state(), scene, ticks=60)
    tail = [luma_of(s, scene) for s, _ in seen[-15:]]
    assert max(tail) < 250.0, f"scene {scene:g} settles hot at {max(tail):.0f}"
    assert max(tail) - min(tail) < 20, f"scene {scene:g} oscillates"
    assert min(tail) > 195.0, f"scene {scene:g} settles dark at {min(tail):.0f}"


def test_the_collection_box_flicker_replayed():
    """The scene that flickered, 2026-09-23, left wrist, static, dim room.

    Measured at 30 ms: gain 63 -> p99.5 194, gain 99 -> 255 with 31% of the
    circle clipped (mean 101 -> 149, `k` ~1.1 against `light_index`). The
    target sits near gain 83. The old law climbed from 194 straight to 99,
    panicked back to 63, and did it again, forever. This one must arrive
    without a single clipped tick and then stop writing.
    """
    ref = ph.light_index(300, 63)

    def box(st):
        p = 194.0 * (st.light / ref) ** K_LIVE
        return meas(min(p, 255.0), clip=0.31 if p >= 255.0 else 0.0)

    st = ph.ServoState(exposure=300, gain=63)
    seen = []
    for _ in range(60):
        m = box(st)
        seen.append(((st.exposure, st.gain), m.p995))
        st = ph.step(m, st)
    assert max(p for _, p in seen) < 250.0, f"clipped on the way: {seen[:8]}"
    assert ph.TARGET_P995 - ph.DEADBAND_DARK <= seen[-1][1] < 250.0
    assert len({reg for reg, _ in seen[-40:]}) == 1, "still writing registers"


def test_a_room_below_the_servable_range_reports_dark():
    st, _ = converge(ph.initial_state(), 5e-5, ticks=40)
    assert st.dark
    assert st.exposure == max(ph.EXPOSURE_BANDS) and st.gain == ph.GAIN_MAX


def test_tracks_a_room_that_changes_mid_take():
    """Someone turns a lamp on. The loop must re-converge, not sit blown.

    This is the whole reason for running live rather than freezing per take.
    """
    st, _ = converge(ph.initial_state(), SCENE_DIM, ticks=40)
    brighter = SCENE_DIM * 4
    st2, seen = converge(st, brighter, ticks=40)
    assert luma_of(st2, brighter) < 250
    settle = next(i for i, (s, _) in enumerate(seen)
                  if luma_of(s, brighter) < 250)
    assert settle <= 12, f"took {settle} ticks ({settle * 100} ms) to recover"


# --- The runner: the only part that touches a camera -----------------------

CIRCLE = dict(cx=648.49, cy=386.54, r=375.06)


def a_frame(luma: int) -> bytes:
    """A 1280x720 JPEG whose lens circle is a flat `luma`, encoded the way the
    camera encodes: MJPG at the rig's native geometry."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    img = np.zeros((720, 1280, 3), np.uint8)
    m = ph.circle_mask(img.shape, scale=1, **CIRCLE)
    img[m] = luma
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


class FakeControls:
    """Stands in for the pyuvc control dict. Records the write ORDER, which is
    part of the contract — see `ServoRunner._apply`."""

    def __init__(self):
        self.writes: list[tuple[str, int]] = []
        self.fail_on: str | None = None

    def __call__(self, name, value):
        if self.fail_on == name:
            raise RuntimeError("driver said no")
        self.writes.append((name, int(value)))


def test_runner_primes_before_the_first_frame():
    """A camera coming out of `auto` holds blown registers until the loop
    first writes. Without priming the opening ~100 ms of every session is
    overexposed, and at 30 fps that is three frames of every rollout."""
    w = FakeControls()
    r = ph.ServoRunner("w", **CIRCLE, write=w)
    r.prime()
    assert dict(w.writes) == {"Gain": r.state.gain,
                              "Absolute Exposure Time": r.state.exposure}


def test_gain_is_written_before_exposure():
    """Both are multipliers, so a frame caught between the two writes sees one
    applied and not the other. Gain first makes that frame DARK; exposure
    first makes it BRIGHT, and bright can clip. Same asymmetry as the rest."""
    w = FakeControls()
    ph.ServoRunner("w", **CIRCLE, write=w).prime()
    assert [n for n, _ in w.writes] == ["Gain", "Absolute Exposure Time"]


def test_only_one_frame_in_every_is_decoded():
    w = FakeControls()
    r = ph.ServoRunner("w", **CIRCLE, write=w, every=3)
    frame = a_frame(140)
    acted = [r.offer(frame) for _ in range(9)]
    assert sum(acted) == 3, acted
    assert r.ticks == 3


def test_the_runner_measures_the_circle_it_was_given():
    w = FakeControls()
    r = ph.ServoRunner("w", **CIRCLE, write=w, every=1)
    r.offer(a_frame(200))
    assert r.last is not None
    assert r.last.p995 == pytest.approx(200, abs=4)
    assert r.last.pixels > 10_000


def test_a_bright_frame_makes_the_runner_reduce_exposure():
    w = FakeControls()
    r = ph.ServoRunner("w", **CIRCLE, write=w, every=1)
    before = r.state.light
    for _ in range(4):
        r.offer(a_frame(254))
    assert r.state.light < before
    assert w.writes, "it must actually have written to the camera"


def test_a_decode_fault_does_not_escape_into_the_capture_loop():
    """A photometry bug must cost a photometry tick, never an image server."""
    w = FakeControls()
    r = ph.ServoRunner("w", **CIRCLE, write=w, every=1)
    assert r.offer(b"not a jpeg at all") is False
    assert r.faults == 1
    assert r.offer(a_frame(140)) is True, "and it must recover afterwards"


def test_a_refused_control_write_does_not_escape_either():
    w = FakeControls()
    w.fail_on = "Absolute Exposure Time"
    r = ph.ServoRunner("w", **CIRCLE, write=w, every=1)
    for _ in range(3):
        r.offer(a_frame(254))
    assert r.faults >= 1


def test_runner_for_skips_a_camera_with_no_circle():
    assert ph.runner_for("head", {"image_shape": [480, 1280]}, write=None) is None


def test_runner_for_refuses_a_half_declared_circle():
    """Silently ignoring an incomplete circle would drop that camera back to
    `auto` — the failure mode this mode exists to remove — with no message."""
    with pytest.raises(RuntimeError, match="cx|cy|r"):
        ph.runner_for("w", {"circle": {"cx": 1.0, "cy": 2.0}}, write=None)


# --- finding the circle ----------------------------------------------------

def synthetic(cx, cy, r, *, n=12, h=720, w=1280, moving=True, glare=0):
    """`n` frames of a lit disc with a dark object moving across it.

    The moving object is the point: a wrist camera looks at its own gripper,
    and a stationary dark object is what breaks a naive fit.
    """
    np = pytest.importorskip("numpy")
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    out = []
    for i in range(n):
        f = np.zeros((h, w), np.uint8)
        f[disc] = 180
        # a black bar sweeping across, or parked in one place
        x0 = int(cx - r + (2 * r - 120) * (i / max(n - 1, 1))) if moving else int(cx - 60)
        f[:, max(x0, 0):x0 + 120] = 0
        if glare:
            f[: glare, int(cx) - 200:int(cx) + 200] = 60      # light outside the disc
        out.append(f)
    return out


def test_fit_recovers_a_synthetic_circle():
    fit = ph.fit_circle(synthetic(648.0, 386.0, 375.0))
    assert fit.ok, fit.why
    assert fit.cx == pytest.approx(648.0, abs=3)
    assert fit.cy == pytest.approx(386.0, abs=3)
    assert fit.r == pytest.approx(375.0, abs=3)
    assert fit.resid_p95 < 5.0


def test_fit_survives_an_object_that_moves_across_the_lens():
    """The per-pixel maximum is what makes this work: the bar is dark in any
    one frame but every pixel it crosses is bright in some other frame."""
    fit = ph.fit_circle(synthetic(640.0, 360.0, 300.0, n=16, moving=True))
    assert fit.ok, fit.why
    assert fit.r == pytest.approx(300.0, abs=4)


def test_fit_survives_an_object_parked_over_the_lens():
    """REGRESSION, and the one I got backwards.

    The gripper is bolted to the wrist and never moves out of view, so part of
    the boundary traces ITS silhouette rather than the lens rim. I first
    concluded from this that a circle could not be fitted on the robot at all.
    That was wrong: three points define a circle, the occlusion leaves ~305
    degrees of arc, and the failure was using plain least squares — which any
    minority of outliers destroys. RANSAC fits the dominant arc and ignores
    the silhouette.
    """
    fit = ph.fit_circle(synthetic(648.0, 386.0, 375.0, moving=False))
    assert fit.ok, fit.why
    assert fit.cx == pytest.approx(648.0, abs=4)
    assert fit.cy == pytest.approx(386.0, abs=4)
    assert fit.r == pytest.approx(375.0, abs=4)


def test_a_partly_occluded_circle_still_fits():
    """Explicitly: black out the bottom third, as the gripper does."""
    np = pytest.importorskip("numpy")
    frames = synthetic(640.0, 360.0, 300.0)
    frames = [f.copy() for f in frames]
    for f in frames:
        f[480:, :] = 0
    fit = ph.fit_circle(frames)
    assert fit.ok, fit.why
    assert fit.r == pytest.approx(300.0, abs=5)
    # Under the sharp-edge rule an occluded arc contributes NO points rather
    # than wrong ones -- a ray into the black-out finds no drop -- so the fit
    # simply rests on less of the rim.
    whole = ph.fit_circle(synthetic(640.0, 360.0, 300.0))
    assert fit.points < 0.9 * whole.points


def test_fit_refuses_a_blocked_lens():
    np = pytest.importorskip("numpy")
    fit = ph.fit_circle([np.zeros((720, 1280), np.uint8) for _ in range(8)])
    assert not fit.ok


def test_glare_outside_the_lens_is_rejected_as_outliers():
    """Overexposure spills light outside the image circle — measured on the
    robot, 354 lit pixels in row 0 where the circle predicts none.

    A least-squares fit absorbs that and moves; RANSAC leaves it out. This is
    the property that makes a fit trustworthy on frames that are not perfectly
    exposed, which is every frame before the servo has converged.
    """
    frames = synthetic(648.0, 386.0, 375.0, glare=90)
    fit = ph.fit_circle(frames)
    assert fit.ok, fit.why
    assert fit.cx == pytest.approx(648.0, abs=4)
    assert fit.cy == pytest.approx(386.0, abs=4)
    assert fit.r == pytest.approx(375.0, abs=4)


def test_a_lit_region_that_is_no_circle_at_all_is_refused():
    """Two disjoint blobs are not a lens. The guard has to be the shape of the
    inliers, not the presence of outliers."""
    np = pytest.importorskip("numpy")
    frames = []
    for _ in range(8):
        f = np.zeros((720, 1280), np.uint8)
        f[100:600, 100:500] = 200          # a big rectangle
        f[100:600, 800:1200] = 200         # and another
        frames.append(f)
    fit = ph.fit_circle(frames)
    assert not fit.ok
    assert fit.why


def test_fit_reports_enough_to_judge_it():
    """A refit button needs a number to gate on, not just a boolean."""
    fit = ph.fit_circle(synthetic(648.0, 386.0, 375.0))
    d = fit.as_dict()
    for key in ("cx", "cy", "r", "points", "resid_p95", "occupancy", "ok"):
        assert key in d
    assert 0.20 <= fit.occupancy <= 0.60

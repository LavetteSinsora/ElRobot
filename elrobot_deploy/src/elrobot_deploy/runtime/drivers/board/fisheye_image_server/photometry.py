"""Closed-loop exposure for a fisheye camera, metered on the lens circle.

WHY THIS EXISTS
---------------

The wrist fisheye projects its image circle onto a rectangular sensor, so
**~60% of the frame the camera meters is the lens's black surround** — pixels no
exposure can brighten. The camera's own auto-exposure averages the whole frame,
finds it dark, and raises exposure and gain until the part that IS a picture
goes through the ceiling.

Measured on the reference robot, 2026-09-20, `photometry: auto`, both wrists:

    full-frame mean      67.7 / 67.9      <- what AE is regulating
    disk mean           150.4 / 163.4
    disk clipped >=250   26.3% / 26.3%    <- a quarter of the view is flat white

The head camera is the control that settles it: same room, same light, same
`Auto Exposure Mode: 8`, but rectilinear with no black surround — full-frame
mean 138.7, clipped 0.16%. The room is fine. The metering window is wrong.

**There is no metering-ROI control on this camera.** Probed live: the list is
the standard UVC set (exposure, gain, backlight compensation,
brightness/contrast/gamma, white balance, power line, pan/tilt/roll, sharpness,
saturation, hue) and nothing region-, window- or zone-shaped. So the loop cannot
be pushed into the firmware; it has to run here.

WHAT IT CONTROLS, AND WHY NOT THE MEAN
--------------------------------------

**The setpoint is a high percentile of luma inside the circle, not the mean.**
The mean is a weak predictor of clipping — measured across 1798 recorded frames:

    mean 130-145   median clip 0.01%   p90 clip  0.83%
    mean 145-160   median clip 0.02%   p90 clip  1.35%
    mean 160-175   median clip 0.82%   p90 clip 10.05%    <- still a fine mean
    mean 175+      median clip 14.6%   p90 clip 29.33%

while `p99.5` is direct, with a knee exactly at the ceiling:

    p99.5 < 240    median clip 0.00%
    p99.5 240-250  median clip 0.02%   p90 clip  0.26%
    p99.5 250-254  median clip 3.63%   p90 clip 19.14%    <- the cliff
    p99.5 >= 254   median clip 26.6%

Holding `p99.5` just under the ceiling is "expose to the right": collect as much
light as possible without squashing distinct brightnesses into one identical
255. That squash is the one *irreversible* failure — a later tone curve can
darken a blown region but never restore the texture inside it, and a learned
policy's brightness augmentation cannot manufacture it either (a monotone
[0,1]->[0,1] jitter never clips and never flattens).

`TARGET_P995 = 245` sits just under the knee, and matches well-exposed
recordings made with tuned manual controls on the same lens: their median
`p99.5` is 243 (left) and 249 (right), at 0.03% and 0.28% clipping.

THE PLANT MODEL IS DELIBERATELY ROUGH
-------------------------------------

`gain_multiplier` below is a one-parameter fit to a nine-point sweep, and it is
not accurate — measured gain 0/32/64 at a fixed exposure gives relative light
1.0 / 2.50 / 4.68, where the model says 1.0 / 2.52 / 4.05. It does not matter.
**The loop closes on the image**, so a model that is 15% off changes how many
ticks convergence takes and nothing else — PROVIDED the error is on the safe
side. A loop that believes the plant is flatter than it is steps too far, and
near a clipping knee "too far" is a limit cycle, not a slow convergence. That
is exactly how an earlier version of this loop flickered (see `PLANT_K_MAX`),
and why the step is sized from a bound on the plant rather than a fit of it.
The alternative — a per-unit
calibration table — is a thing that goes stale the first time a camera is
swapped and then lies silently. A rough model that is re-measured every tick
does not have that failure mode.

The same reasoning is why exposure is treated as exactly linear in its register
even though the sweep says 300 units returns ~2.55x the light of 100 rather
than 3x.

WHY 10 Hz
---------

**The plant settles in 2 frames (~67 ms), measured.** Sampling every 100 ms
therefore measures a fully settled response to the previous command: there are
no hidden dynamics between samples, so the loop has no phase lag to oscillate
on. Sampling at frame rate would measure frames still in transit and would need
detuning to stay stable. 10 Hz is the right rate, not a concession to cost.

Cost, measured on the board (Orin NX): a 1/2-scale grayscale decode of one
1280x720 frame is 2.30 ms, so two wrists at 10 Hz is 46 ms/s — 1.5% of the
publisher's three pinned cores. The pass-through path is untouched; 1 frame in
3 gets an extra decode *after* its send.

Runs on the robot board inside `publisher.py` (and its lens-circle fit is used
on the deploy box by `runtime/drivers/fisheye_circle_fit.py`). See README.md
beside this file.
"""

from __future__ import annotations

import dataclasses
import math


#: Luma percentile the loop holds. See the knee table above: below 250 there is
#: essentially no clipping, and well-exposed reference recordings sit at 243/249.
TARGET_P995 = 245.0

#: Exposure register values the loop may choose, in this driver's unit of
#: 100 us. The camera offers [50..500]; these are the three that matter.
#:
#: **10 / 20 / 30 ms, and they are whole multiples of the 10 ms flicker period
#: of 50 Hz mains** (China), which cancels banding. Anything past 330 would let
#: the sensor output 1/exposure instead of 30 fps — measured on this model:
#: 33 ms -> 30.10 fps, 40 -> 24.93, 50 -> 19.94. So 300 is the ceiling and the
#: frame rate becomes a property of this list rather than of the room's light.
EXPOSURE_BANDS = (100, 200, 300)

#: The band the loop prefers to sit in, and returns to when it can.
#:
#: 20 ms is a compromise: 13 ms of margin below the frame-rate cliff, a whole
#: number of flicker periods, and enough light that gain stays near 60 rather
#: than near 100. Against a ~10 ms exposure it doubles motion blur — at wrist
#: speeds, on an image a policy downsizes to a few hundred pixels, that is
#: sub-pixel, but it IS an assumption,
#: and `PREFERRED_BAND = 100` is the change that removes it at the cost of
#: roughly 1.6x the gain (and therefore the noise).
PREFERRED_BAND = 200

#: Gain register range this camera exposes, and the window the loop keeps gain
#: inside before it would rather change exposure band.
GAIN_MIN, GAIN_MAX = 0, 128
GAIN_COMFORT_LO, GAIN_COMFORT_HI = 8, 112

#: Hysteresis: once in a band, stay there until gain leaves THIS wider window.
#: Without the gap between the two windows a scene sitting exactly on a band
#: boundary switches every tick, and each switch is a visible brightness step.
GAIN_HYST_LO, GAIN_HYST_HI = 2, 124

#: Gain register -> relative light, as `1 + gain / GAIN_KNEE`. Fitted to the
#: sweep; see "the plant model is deliberately rough" above.
GAIN_KNEE = 21.0

#: Fraction of the circle at >=250 above which `p99.5` is uninformative.
#:
#: THIS IS THE ONE CASE THE RATIO LAW CANNOT HANDLE. When the highlight is
#: pegged at 255 the measured error is zero however far over the camera is, so
#: the loop would sit at maximum overexposure believing it had converged. Above
#: this fraction the loop stops measuring and just cuts by `PANIC_CUT` until
#: the highlight comes off the rail.
CLIP_PANIC = 0.02
PANIC_CUT = 0.70

#: Ignore `p99.5` this close to the target, in luma levels. Stops the loop
#: rewriting registers every tick over sensor noise; each write is a USB
#: control transfer and a visible (if small) brightness step.
#:
#: **ASYMMETRIC, AND THAT IS THE POINT.** A symmetric band wide enough to stop
#: chatter on the dark side lets `p99.5` sit at 252 on the bright side — past
#: the measured knee at 250, where median clipping is already 3.63%. So the band
#: is tight against the ceiling and loose away from it, which is just the design
#: principle restated: overexposure is irreversible and darkness is not. With
#: the target at 245: "correct as soon as we exceed 247, tolerate down to 235".
DEADBAND_DARK = 10.0       # p995 below target: we need more light. Tolerate.
DEADBAND_BRIGHT = 2.0      # p995 above target: too bright. Barely tolerate.

#: Largest single-tick change in `light_index`, up or down. At 10 Hz a 1.6x
#: ceiling crosses the camera's whole useful range in under a second while
#: keeping any one step small enough not to look like a flash in the recording.
MAX_STEP = 1.60

#: How steeply `p99.5` answers the registers: `p995 ~ light_index ** k`. The
#: loop does not know `k` and does not need to — it needs BOUNDS on it, and it
#: uses the bound that makes each direction's mistake the recoverable one.
#:
#: **Climbing assumes the STEEPEST plant (`PLANT_K_MAX`).** Every step up is
#: then at most the step that lands exactly on the target, so the approach
#: from below is monotone and never overshoots into the clip, whatever the
#: true `k` is inside the bound. A flatter camera just takes a few more ticks.
#: **Descending assumes the FLATTEST (`PLANT_K_MIN`)**, so a cut is at least
#: big enough; overshooting DOWN is into darkness, which the next climb fixes.
#:
#: WHY A BOUND AND NOT A GAMMA MODEL: a gamma model causes flicker. Converting
#: luma to light with gamma 2.2 (`k = 0.45`) and under-relaxing by 0.70
#: assumes `k = 0.65`. Measured live (2026-09-23) on a 10bb:2b08 camera, three
#: step responses at 30 ms on a static scene
#: (gain 73->91, 73->57, 63->99): `k = 1.0 .. 1.2` — this camera's JPEG luma
#: is close to LINEAR in exposure x gain over the range the loop uses, and
#: the gain register is steeper at the top than `gain_multiplier` says. So
#: every climb under the gamma model goes ~1.7x too far: 219 aimed at 245
#: lands at 255 (11% clipped), `PANIC_CUT` drops it, the next climb clips
#: again — a flash to 255 and back to ~190 several times a second. The target
#: is reachable all along (gain ~83 at that scene); the step is just sized
#: for a plant that is not this one. 2.0 is the measured 1.2 with room for the top of the gain
#: register, where `gain_multiplier` underestimates the curve most, and for a
#: tone curve steeper than linear. The cost of the margin is only speed: a
#: gamma-2.2 camera would close ~1/4 of its error per climb, ~0.5 s from 20%
#: dark, and the target is still approached from below.
PLANT_K_MAX = 2.0
PLANT_K_MIN = 1.0 / 2.2

#: `p99.5` the rig must be able to reach before a take is worth recording, and
#: the gain above which it can only reach it by being noisy.
DARK_REFUSE_P995 = 230.0
NOISY_GAIN = 96


def gain_multiplier(gain: int) -> float:
    """Gain register -> relative light. Rough by design."""
    return 1.0 + max(gain, 0) / GAIN_KNEE


def gain_for(light_index: float, band: int) -> float:
    """The gain register that reaches `light_index` at this exposure band.

    May fall outside `[GAIN_MIN, GAIN_MAX]`; the caller decides whether that
    means "change band" or "this room is too dark".
    """
    return ((light_index / band) - 1.0) * GAIN_KNEE


def light_index(exposure: int, gain: int) -> float:
    """The scalar the loop actually servos. Light is proportional to
    exposure time times gain, so one number stands for both registers and the
    split back into (band, gain) is a separate, purely mechanical decision."""
    return exposure * gain_multiplier(gain)


@dataclasses.dataclass(frozen=True)
class Measurement:
    """What one decoded frame says about the light inside the lens circle."""

    p995: float
    mean: float
    clip_frac: float
    #: Pixels the measurement was taken over. A circle that has shrunk to
    #: nothing — a blocked lens, a bad refit — must not be servoed on.
    pixels: int

    @property
    def usable(self) -> bool:
        return self.pixels >= 256


@dataclasses.dataclass(frozen=True)
class ServoState:
    """Where the loop currently is. Immutable; `step` returns the next one."""

    exposure: int
    gain: int
    #: Set when the loop wanted more light than the camera can give. The take
    #: is then noisy at best and underexposed at worst; the caller decides.
    dark: bool = False
    #: Set while the highlight is pegged and the loop is cutting blind.
    saturated: bool = False
    #: Ticks since a register was actually written. Only for logging.
    held: int = 0

    @property
    def light(self) -> float:
        return light_index(self.exposure, self.gain)

    @property
    def noisy(self) -> bool:
        return self.gain >= NOISY_GAIN


def initial_state(band: int = PREFERRED_BAND, gain: int = 64) -> ServoState:
    """Where to start before the first measurement.

    `200 / 64` is not arbitrary: on the reference rig it is the sweep point
    that landed closest to the target — right wrist 153.2 mean / p99.5 248 /
    0.18% clipped. Starting there means the first frame is already close, and
    the loop only trims.
    """
    return ServoState(exposure=int(band), gain=int(gain))


def _split(light: float, current_band: int) -> tuple[int, int]:
    """`light_index` -> (exposure band, gain), preferring to stay put.

    Band order is the preferred band first, then the rest ascending, so a room
    that can be served from several bands is always served from the one whose
    noise and motion blur were reasoned about.
    """
    # Stay in the current band while its gain is merely outside the comfort
    # window but still inside the hysteresis window. This is what stops a
    # scene sitting on a boundary from switching band every tick.
    g_here = gain_for(light, current_band)
    if GAIN_HYST_LO <= g_here <= GAIN_HYST_HI:
        if GAIN_COMFORT_LO <= g_here <= GAIN_COMFORT_HI:
            return current_band, int(round(g_here))
        # Outside comfort but inside hysteresis: only move if another band can
        # actually do better, otherwise hold.
        for band in _band_order():
            if band == current_band:
                continue
            g = gain_for(light, band)
            if GAIN_COMFORT_LO <= g <= GAIN_COMFORT_HI:
                return band, int(round(g))
        return current_band, int(round(g_here))

    for band in _band_order():
        g = gain_for(light, band)
        if GAIN_COMFORT_LO <= g <= GAIN_COMFORT_HI:
            return band, int(round(g))

    # Nothing comfortable: saturate against the nearest reachable extreme.
    if light > light_index(max(EXPOSURE_BANDS), GAIN_MAX):
        return max(EXPOSURE_BANDS), GAIN_MAX
    if light < light_index(min(EXPOSURE_BANDS), GAIN_MIN):
        return min(EXPOSURE_BANDS), GAIN_MIN
    band = current_band
    return band, int(round(min(max(gain_for(light, band), GAIN_MIN), GAIN_MAX)))


def _band_order() -> tuple[int, ...]:
    rest = tuple(b for b in sorted(EXPOSURE_BANDS) if b != PREFERRED_BAND)
    return (PREFERRED_BAND,) + rest


def step(meas: Measurement, state: ServoState, *,
         target: float = TARGET_P995) -> ServoState:
    """One control tick. Pure: no camera, no clock, no I/O.

    Returns the next state. When nothing should change it returns a state equal
    to `state` except for `held`, and the caller writes no registers.
    """
    if not meas.usable:
        # A circle that measured almost nothing is a broken measurement, not a
        # dark room. Hold, and let the caller notice `held` climbing.
        return dataclasses.replace(state, held=state.held + 1)

    if meas.clip_frac > CLIP_PANIC and meas.p995 >= 254.0:
        # Blind: the highlight is on the rail, so the error is unmeasurable.
        # Cut by a fixed factor and re-measure. See CLIP_PANIC.
        band, gain = _split(state.light * PANIC_CUT, state.exposure)
        return ServoState(band, gain, dark=False, saturated=True, held=0)

    p995 = max(meas.p995, 1.0)
    if target - DEADBAND_DARK <= p995 <= target + DEADBAND_BRIGHT:
        return dataclasses.replace(state, saturated=False, held=state.held + 1)

    # See PLANT_K_MAX: climb as if the plant were as steep as it can be, cut
    # as if it were as flat as it can be. Each direction's error is darkness.
    k = PLANT_K_MAX if p995 < target else PLANT_K_MIN
    ratio = (target / p995) ** (1.0 / k)
    ratio = min(max(ratio, 1.0 / MAX_STEP), MAX_STEP)
    wanted = state.light * ratio
    band, gain = _split(wanted, state.exposure)

    reachable = light_index(max(EXPOSURE_BANDS), GAIN_MAX)
    dark = wanted > reachable * 1.001 and meas.p995 < DARK_REFUSE_P995

    if (band, gain) == (state.exposure, state.gain):
        return dataclasses.replace(state, dark=dark, saturated=False,
                                   held=state.held + 1)
    return ServoState(band, gain, dark=dark, saturated=False, held=0)


def measure(gray, circle_mask) -> Measurement:
    """`p99.5`, mean and clipped fraction inside the lens circle.

    `gray` is a single-channel array at whatever scale it was decoded; the mask
    must already be at that same scale. Kept separate from `step` so the law
    can be tested without numpy images and so the caller owns the decode.

    Scale costs almost nothing here: measured on real frames at the operating
    point, a 1/2-scale decode shifts `p99.5` by -0.16 luma levels and a
    1/8-scale one by -0.98. Averaging DOES hide small clipped clusters, which
    is why the default is 1/2 rather than the cheapest option — the difference
    in decode cost is 0.6 ms and the difference in bias is the whole point.
    """
    import numpy as np

    v = gray[circle_mask]
    if v.size == 0:
        return Measurement(0.0, 0.0, 0.0, 0)
    v = v.astype(np.float32)
    return Measurement(
        p995=float(np.percentile(v, 99.5)),
        mean=float(v.mean()),
        clip_frac=float((v >= 250).mean()),
        pixels=int(v.size))


def circle_mask(shape, cx: float, cy: float, r: float, scale: int = 1,
                inset: float = 0.92):
    """Boolean mask of the lens circle, for an image decoded at `1/scale`.

    `inset` keeps the measurement off the rim, where the lens's own falloff and
    the JPEG's ringing both live — neither is scene content and both would drag
    the mean down and the loop up.
    """
    import numpy as np

    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx / scale) ** 2 + (yy - cy / scale) ** 2) <= (r * inset / scale) ** 2


# --- finding the circle -----------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CircleFit:
    """A fitted lens circle, with enough to judge whether to believe it."""

    cx: float
    cy: float
    r: float
    #: Boundary points the fit was taken over, the fraction of them that ended
    #: up ON the fitted circle, and the 95th percentile of THEIR distance from
    #: it. A good fit on this rig runs 3-5 px over 55%+ of the boundary.
    #:
    #: `inlier_frac` is well below 1 even for a good fit and that is expected:
    #: the gripper occludes the bottom of the lens, so a real arc of the
    #: boundary is the gripper's silhouette rather than the lens rim. Judging
    #: the fit on ALL points instead of the inliers is what made an earlier
    #: version reject a perfectly good circle.
    points: int
    inlier_frac: float
    resid_p95: float
    #: Fraction of the frame the lens circle covers. ~35-41% on these wrists;
    #: far from that means the threshold found something that is not the lens.
    occupancy: float
    ok: bool
    why: str = ""

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


#: THE RULE IS THE SHARP EDGE, chosen by comparing three rules side by side on
#: two lenses (a 210 and a 180 degree one). Walk rays out from a centre and take, on each, the STEEPEST
#: brightness drop: that is where the image physically ends.
#:
#: A brightness-threshold rule ("every pixel ever brighter than 12 is lens")
#: measures something else, and it shows. Fitted radius, px:
#:
#:                          threshold         a stored fit     sharp edge
#:     210, clean take           327-334          325-329         323.0-323.3
#:     210, glare take           364, 389         331-332         323.8-324.7
#:     180 camera                refused          513-655 (junk)  391.9, 394.0
#:
#: Glare puts a halo PAST the edge, bright enough to pass any fixed threshold,
#: so the threshold circle grows with it; a halo only softens the drop, it does
#: not move it. And on the 180 degree camera the surround is not black at all
#: -- an illuminated lens barrel at luma 30-60, with a black border ring between
#: -- so "lit" is most of the frame and no threshold separates lens from
#: surround. The edge rule does not care what is outside, only where the image
#: stops.

#: Rays per frame (every 360/N degrees), and the blur applied first so a single
#: noisy pixel cannot be the steepest drop.
EDGE_ANGLES = 180
EDGE_BLUR_SIGMA = 1.5
#: A drop only counts if it is at least this steep (luma per pixel, after the
#: blur) and the image just inside it is at least this bright. Scene edges pass
#: too -- a white wall against the black gripper -- but they do not agree on a
#: circle across rays and frames, and RANSAC leaves them out.
EDGE_MIN_DROP = 8.0
EDGE_MIN_INNER = 40.0
#: ...and the band just OUTSIDE it (3-12 px out) must fall below this fraction
#: of the band just inside (3-8 px in). That is the lens's black border -- true
#: of both lenses, the 210's black surround and the 180's black ring before its
#: barrel -- and rarely true of a scene edge, which has more scene beyond it.
EDGE_DARK_RATIO = 0.5
#: The first pass searches from the frame centre over this fraction of the frame
#: HEIGHT; the second searches +-EDGE_REFINE_BAND px about the first answer, from
#: its centre. Two passes because a ray from the wrong centre meets the rim at a
#: slant, where the drop is smeared.
EDGE_SEARCH = (0.30, 0.95)
EDGE_REFINE_BAND = 40.0
#: Edge points are precise to a pixel or two, so the inlier band is tighter
#: than the old rule's 4 px, which was sized for a threshold contour.
EDGE_RANSAC_TOL = 2.5
#: A circle whose radius is not within this fraction of the frame height is not
#: this rig's lens: measured 0.45 (210 degree, 323 px at 720) to 0.54 (180
#: degree, 392 px at 720 and 588 at 1080).
FIT_RADIUS_RANGE = (0.25, 0.80)
#: ...nor one whose centre is further than this fraction of the width / height
#: from the frame's centre: measured within 0.05 / 0.07 on every camera of
#: both lenses. APPLIED INSIDE RANSAC, not after it -- on a static scene a
#: curved scene edge (a window's top, seen through a fisheye) is a sharp drop
#: in the same place in every frame, and on dark frames where the rim barely
#: shows it out-voted the rim with a circle centred at y=646 on a 720-row
#: frame. As a post-check that refuses; as a candidate filter it cannot win.
FIT_CENTRE_RANGE = (0.12, 0.15)

#: RANSAC: a boundary point this far from a candidate circle is ON it.
#:
#: THE FIT IS ROBUST, AND IT HAS TO BE. The outliers are guaranteed: the gripper
#: is bolted to the wrist and occludes the bottom of the lens, and scene edges
#: inside the disc produce steep drops of their own. RANSAC finds the arc that
#: the rays and frames AGREE on and ignores the rest.
FIT_RANSAC_TOL = 4.0
FIT_RANSAC_ITERS = 400

#: Reject a fit supported by less of the edge points than this, or whose inliers
#: scatter more than `FIT_MAX_RESID`. Low on purpose: occlusion and in-scene
#: edges are always a large minority. What a bad fit looks like is inliers that
#: do not lie on a circle, not a low fraction.
FIT_MIN_INLIERS = 0.35
FIT_MAX_RESID = 8.0


def _edge_points(frames, cx, cy, r_lo, r_hi):
    """`(x, y)` of the lens rim on each ray of each frame: the OUTERMOST drop
    that is steep, has image inside it and goes dark outside it.

    Outermost, not steepest: every scene edge lies inside the rim, and on a
    static scene they sit in the same place in every frame, so pooling does
    not average them away. The rim is the last real edge before the border.

    A ray is walked only while it is on the sensor: where the circle overflows
    the frame (the 180 degree camera's top and bottom) the ray leaves before it
    reaches the rim and contributes nothing, instead of a false edge at the
    sensor's border.
    """
    import cv2
    import numpy as np

    angles = np.linspace(0.0, 2 * np.pi, EDGE_ANGLES, endpoint=False)
    radii = np.arange(max(r_lo, 8.0), r_hi, 1.0)
    n = radii.size
    if n < 30:
        return np.empty((0, 2))
    xs = cx + radii[None, :] * np.cos(angles)[:, None]
    ys = cy + radii[None, :] * np.sin(angles)[:, None]
    idx = np.arange(n)[None, :]
    rows = np.arange(len(angles))

    def band_mean(csum, a, b):
        lo = np.broadcast_to(np.clip(idx + a, 0, n), csum[:, 1:].shape)
        hi = np.broadcast_to(np.clip(idx + b, 0, n), csum[:, 1:].shape)
        total = np.take_along_axis(csum, hi, 1) - np.take_along_axis(csum, lo, 1)
        return total / np.maximum(hi - lo, 1)

    out = []
    for frame in frames:
        h, w = frame.shape
        on = (xs >= 2) & (xs < w - 2) & (ys >= 2) & (ys < h - 2)
        length = np.where(on.all(axis=1), n, np.argmin(on, axis=1))
        smooth = cv2.GaussianBlur(np.asarray(frame, np.float32), (0, 0), EDGE_BLUR_SIGMA)
        prof = smooth[np.clip(ys, 0, h - 1).astype(int), np.clip(xs, 0, w - 1).astype(int)]
        drop = np.gradient(prof, axis=1)
        csum = np.concatenate([np.zeros((len(angles), 1), np.float64),
                               np.cumsum(prof, axis=1, dtype=np.float64)], axis=1)
        inner = band_mean(csum, -8, -3)
        outer = band_mean(csum, 3, 13)
        edge = ((drop < -EDGE_MIN_DROP) & (inner > EDGE_MIN_INNER)
                & (outer < EDGE_DARK_RATIO * inner)
                & (idx >= 8) & (idx < (length - 13)[:, None]))
        found = edge.any(axis=1) & (length >= 30)
        last = n - 1 - np.argmax(edge[:, ::-1], axis=1)
        # the steepest point of THAT edge, not of the whole ray
        near = (idx >= (last - 8)[:, None]) & (idx <= last[:, None])
        i = np.argmin(np.where(near, drop, np.inf), axis=1)
        out.append(np.stack([xs[rows, i][found], ys[rows, i][found]], axis=1))
    return np.concatenate(out) if out else np.empty((0, 2))


def fit_circle(frames) -> CircleFit:
    """Find the lens circle from a handful of frames: the SHARP-EDGE rule.

    `frames` are grayscale arrays at FULL sensor scale, all the same shape.
    Several frames, pooled, because a dark object against the rim (a gripper
    jaw) hides the edge in one frame and not in the next; the rim is the one
    edge every frame agrees on.

    Returns a fit with `ok=False` and a reason rather than raising: at camera
    open the caller wants to carry on without a circle, not die.
    """
    import numpy as np

    if not len(frames):
        return CircleFit(0, 0, 0, 0, 0.0, 0.0, 0.0, False, "no frames")
    stack = [np.asarray(f) for f in frames]
    if any(f.ndim != 2 for f in stack) or len({f.shape for f in stack}) != 1:
        return CircleFit(0, 0, 0, 0, 0.0, 0.0, 0.0, False,
                         "expected grayscale frames of one shape")
    h, w = stack[0].shape
    lo, hi = FIT_RADIUS_RANGE

    def refused(why, fx=0.0, fy=0.0, fr=0.0, n=0, frac=0.0, p95=0.0):
        return CircleFit(fx, fy, fr, n, frac, p95, 0.0, False, why)

    ax, ay = FIT_CENTRE_RANGE

    def plausible(cx, cy, r):
        return (lo * h <= r <= hi * h and abs(cx - w / 2) <= ax * w
                and abs(cy - h / 2) <= ay * h)

    pts = _edge_points(stack[::3] or stack, w / 2.0, h / 2.0,
                       EDGE_SEARCH[0] * h, EDGE_SEARCH[1] * w / 2.0 + 0.5 * h)
    if len(pts) < 64:
        return refused(f"only {len(pts)} sharp edges found; the lens may be "
                       "blocked or the image too dark to show its rim")
    first = _ransac_circle(pts[:, 0], pts[:, 1], tol=EDGE_RANSAC_TOL,
                           plausible=plausible)
    if first is None:
        return refused("no circle was supported by enough edge points", n=len(pts))

    pts = _edge_points(stack, first[0], first[1], first[2] - EDGE_REFINE_BAND,
                       first[2] + EDGE_REFINE_BAND)
    got = (_ransac_circle(pts[:, 0], pts[:, 1], tol=EDGE_RANSAC_TOL,
                          plausible=plausible) if len(pts) >= 64 else None)
    if got is None:
        return refused("the edge did not hold up from the fitted centre", n=len(pts))
    fx, fy, fr, inlier_frac, p95 = got

    yy, xx = np.mgrid[0:h:4, 0:w:4]
    occupancy = float(np.mean((xx - fx) ** 2 + (yy - fy) ** 2 <= fr * fr))
    if not plausible(fx, fy, fr):
        return CircleFit(fx, fy, fr, len(pts), inlier_frac, p95, occupancy, False,
                         f"circle ({fx:.0f}, {fy:.0f}) r {fr:.0f} is not where "
                         f"this rig's lens sits (centre near the frame's, radius "
                         f"{lo:.2f}-{hi:.2f} of its height)")
    if inlier_frac < FIT_MIN_INLIERS:
        return CircleFit(fx, fy, fr, len(pts), inlier_frac, p95, occupancy, False,
                         f"only {inlier_frac * 100:.0f}% of the edges lie on any "
                         f"one circle; this is not a lens rim")
    if p95 > FIT_MAX_RESID:
        return CircleFit(fx, fy, fr, len(pts), inlier_frac, p95, occupancy, False,
                         f"even the supporting edges scatter {p95:.1f} px "
                         f"(a rim fits within 1-3); this is not a circle")
    return CircleFit(float(fx), float(fy), float(fr), len(pts), inlier_frac,
                     p95, occupancy, True)


def _kasa(x, y):
    """Algebraic circle fit: minimises |x^2+y^2 - (2cx.x + 2cy.y + c)|, which
    is linear in the unknowns and needs no initial guess."""
    import numpy as np

    a = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    return cx, cy, math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))


def _ransac_circle(x, y, *, tol: float = FIT_RANSAC_TOL,
                   iters: int = FIT_RANSAC_ITERS, seed: int = 0,
                   plausible=None):
    """`(cx, cy, r, inlier_frac, resid_p95)` for the circle most of the
    boundary agrees on, or None.

    Three points define a circle, so a minimal sample is three. `seed` is
    fixed: the same frames must always produce the same circle, or two people
    looking at the same take get different crops.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(x)
    best_inl, best_n = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            cx, cy, r = _kasa(x[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        if not (50.0 < r < 2000.0):        # three near-collinear points
            continue
        if plausible is not None and not plausible(cx, cy, r):
            continue
        inl = np.abs(np.hypot(x - cx, y - cy) - r) < tol
        if int(inl.sum()) > best_n:
            best_inl, best_n = inl, int(inl.sum())

    if best_inl is None or best_n < 64:
        return None

    # Refit on the inliers, then re-select — twice. The minimal sample only
    # has to be good enough to find the right arc; these passes are what make
    # the answer precise.
    cx, cy, r = _kasa(x[best_inl], y[best_inl])
    inl = best_inl
    for _ in range(3):
        inl = np.abs(np.hypot(x - cx, y - cy) - r) < tol
        if inl.sum() < 3:
            return None
        cx, cy, r = _kasa(x[inl], y[inl])

    resid = np.abs(np.hypot(x[inl] - cx, y[inl] - cy) - r)
    return (float(cx), float(cy), float(r), float(inl.mean()),
            float(np.percentile(resid, 95)))


def describe(state: ServoState, meas: Measurement) -> str:
    """One log line. Says what it did AND what it saw, because a photometry
    log that records only the registers cannot be read after the fact."""
    flags = "".join(c for c, on in
                    (("D", state.dark), ("S", state.saturated),
                     ("N", state.noisy)) if on)
    return (f"exp={state.exposure} gain={state.gain}"
            f"{' [' + flags + ']' if flags else ''} "
            f"p99.5={meas.p995:.0f} mean={meas.mean:.0f} "
            f"clip={meas.clip_frac * 100:.2f}%")


def dark_verdict(meas: Measurement, state: ServoState) -> tuple[str, str]:
    """`(verdict, message)` for a preflight check — "ok", "noisy" or "dark".

    The message says what to DO. An operator reading "exposure target not met"
    learns nothing; an operator reading "turn the overhead lights on" acts.
    """
    if state.dark or meas.p995 < DARK_REFUSE_P995:
        return ("dark",
                f"Too dark to record: the wrist camera reaches only "
                f"p99.5={meas.p995:.0f} at {state.exposure // 10} ms and gain "
                f"{state.gain}, against {DARK_REFUSE_P995:.0f} needed. Turn on "
                f"more room light — no camera setting fixes this.")
    if state.noisy:
        return ("noisy",
                f"Usable but noisy: gain is {state.gain} of {GAIN_MAX}. The "
                f"images will be grainy. More room light would let the camera "
                f"drop the gain.")
    return ("ok", "")


# --- the glue ---------------------------------------------------------------
#
# Everything above is pure. This is the part that touches a camera, and it is
# kept to one class with one injected `write` so that the law can be tested
# without hardware and so that a second rig can reuse the shape without
# inheriting this one's driver.


#: Decode one frame in this many. At 30 fps that is 10 Hz — see the module
#: docstring: the plant settles in 2 frames, so a 100 ms sample measures a
#: settled response and the loop has no phase lag to oscillate on.
DECODE_EVERY = 3

#: Decode at 1/this. Costs 2.30 ms on the board against 1.72 ms at 1/8, and
#: biases `p99.5` by -0.16 luma levels against -0.98. The extra 0.6 ms buys
#: a measurement that does not quietly under-read the highlight.
DECODE_SCALE = 2

#: Log a servo line at most this often, in ticks. The loop runs at 10 Hz and
#: nobody wants 10 lines a second; this gives one every ~5 s, plus every
#: actual register change.
LOG_EVERY = 50


class ServoRunner:
    """Drives one camera's exposure from its own frames.

    `write(name, value)` is the only thing it does to the world, and it is
    injected: this class never imports a camera library. The publisher passes
    a closure over its `uvc` control dict; a test passes a list.

    **Nothing here may raise into the capture loop.** A photometry fault must
    degrade to "the camera keeps whatever setting it has and says so", never
    to a stopped image server. Every entry point is wrapped.
    """

    def __init__(self, topic: str, cx: float, cy: float, r: float, *,
                 write, state: ServoState | None = None,
                 every: int = DECODE_EVERY, scale: int = DECODE_SCALE,
                 log=None):
        self.topic = topic
        self.cx, self.cy, self.r = float(cx), float(cy), float(r)
        self._write = write
        self.state = state or initial_state()
        self.every = int(every)
        self.scale = int(scale)
        self._log = log
        self._mask = None
        self._mask_shape = None
        self._n = 0
        self.ticks = 0
        self.faults = 0
        self.last: Measurement | None = None

    # -- what the publisher calls ------------------------------------------

    def prime(self) -> None:
        """Write the starting operating point, before the first frame.

        Without this the camera sits at whatever the last run left in its
        registers until the first tick — which for a camera coming out of
        `auto` means the first ~100 ms of every session is blown.
        """
        self._apply(self.state, force=True)

    def offer(self, jpeg: bytes) -> bool:
        """Hand the runner one frame. Returns True if it acted on it.

        Called once per captured frame from the hot loop; decodes one in
        `every` and ignores the rest, so the pass-through path pays a counter
        increment and nothing else.
        """
        self._n += 1
        if self._n % self.every:
            return False
        try:
            self._tick(jpeg)
            return True
        except Exception as exc:                # noqa: BLE001 - see the docstring
            self.faults += 1
            if self.faults in (1, 10) or self.faults % 200 == 0:
                self._say("photometry fault #%d (%s: %s) — holding exp=%s "
                          "gain=%s", self.faults, type(exc).__name__, exc,
                          self.state.exposure, self.state.gain)
            return False

    # -- internals ----------------------------------------------------------

    def _tick(self, jpeg: bytes) -> None:
        import cv2
        import numpy as np

        flag = {1: cv2.IMREAD_GRAYSCALE,
                2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
                4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
                8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[self.scale]
        gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), flag)
        if gray is None:
            raise RuntimeError("frame did not decode")

        if self._mask_shape != gray.shape:
            self._mask = circle_mask(gray.shape, self.cx, self.cy, self.r,
                                     scale=self.scale)
            self._mask_shape = gray.shape

        self.last = measure(gray, self._mask)
        self.ticks += 1
        nxt = step(self.last, self.state)
        changed = (nxt.exposure, nxt.gain) != (self.state.exposure, self.state.gain)
        self.state = nxt
        if changed:
            self._apply(nxt)
        elif self.ticks % LOG_EVERY == 0:
            self._say("%s", describe(nxt, self.last))

    def _apply(self, st: ServoState, *, force: bool = False) -> None:
        # Exposure before gain is not arbitrary: both are pure multipliers, but
        # writing the longer one first means any frame caught mid-pair is too
        # BRIGHT rather than too dark, and a frame that is briefly too bright
        # can clip. Gain first makes the straddled frame dark, which is
        # recoverable. That is the same asymmetry as everywhere else here.
        self._write("Gain", int(st.gain))
        self._write("Absolute Exposure Time", int(st.exposure))
        if self.last is not None:
            self._say("%s", describe(st, self.last))
        elif force:
            self._say("primed exp=%d gain=%d", st.exposure, st.gain)

    def _say(self, fmt: str, *args) -> None:
        if self._log is not None:
            self._log.info("[%s] " + fmt, self.topic, *args)


def runner_for(topic: str, spec: dict, *, write, log=None) -> ServoRunner | None:
    """A runner for this camera, or None if it does not declare a circle.

    **A camera with no `circle` is not served by the loop, deliberately.** The
    head is rectilinear: its own auto-exposure meters a frame that is entirely
    picture and gets it right (measured: full-frame mean 138.7, 0.16%
    clipped). There is nothing here for it to fix and a circle it does not
    have for the loop to meter.
    """
    circle = spec.get("circle")
    if not circle:
        return None
    missing = [k for k in ("cx", "cy", "r") if k not in circle]
    if missing:
        raise RuntimeError(
            f"camera {topic!r} has a `circle` missing {missing}. The servo "
            f"meters inside the lens circle, so all three are required; they "
            f"come from the same fit that wrote the crop box.")
    return ServoRunner(topic, circle["cx"], circle["cy"], circle["r"],
                       write=write, log=log)


__all__ = [
    "Measurement", "ServoState", "ServoRunner", "runner_for",
    "step", "measure", "circle_mask",
    "initial_state", "describe", "dark_verdict", "light_index",
    "gain_multiplier", "gain_for", "TARGET_P995", "EXPOSURE_BANDS",
    "PREFERRED_BAND", "GAIN_MIN", "GAIN_MAX", "DARK_REFUSE_P995", "NOISY_GAIN",
    "DECODE_EVERY", "DECODE_SCALE",
]

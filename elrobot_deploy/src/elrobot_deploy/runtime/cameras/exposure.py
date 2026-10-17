"""Exposure for a fisheye camera, metered on the lens circle: the law, and the modes.

THE LAW below (`step`, `measure`, the constants) is g1d_deploy's, ported unchanged from
`legacy_g1d/.../board/fisheye_image_server/photometry.py`, where it ran on the robot
board against the same HX200S 210 deg camera; its measurements are kept as they were
written. THE MODES (`Controller`, at the end) are new: which law runs, and when, on
this Mac. See "THE MODES" there.

g1d's rationale follows.


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

Here it runs inside the camera process (P2) on frames it has already decoded: one in three
is converted to half-scale grayscale and measured inside the lens circle. The lens-circle fit
is `lens.py`.
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


# --- THE MODES ---------------------------------------------------------------------------
#
# Which law runs, and when. One `Controller` per camera, in P2. It is pure: it is handed a
# measurement and the time, and answers with the control writes to make; P2 owns the camera,
# the clock and the USB control channel (`uvc.py`).
#
#   camera_first  (the default) THE CAMERA'S OWN SETTINGS FIRST. Its controls are put back to
#                 its factory defaults (auto-exposure on) and left alone, while we meter inside
#                 the lens circle. Only if the camera's loop is clearly failing for a sustained
#                 spell (`Patience.takeover_*`: a large part of the disc clipped, or the disc
#                 as a whole too bright, or dark) do we take over, from the exposure and gain
#                 the camera had reached, and run `sticky` for the rest of the session.
#
#                 "Clearly failing" is deliberately NOT sticky's own trigger (1% clipped).
#                 Measured on our wrist camera, 2026-10-17, a desk with a lit monitor in view:
#                 the camera's own loop held the disc at mean 157 with 7.0% clipped (all of it
#                 the monitor) and the workspace well exposed; taking over at the 1% rule held
#                 p99.5 at 235 by darkening everything else to mean 48 at gain 97 (noisy) to
#                 save the monitor's contents. For the arm, the camera's image was the better
#                 one. A light source in a 210 deg view always clips at an exposure that suits
#                 the workspace; what g1d measured on its rig (26% of the disc clipped, the
#                 camera metering the black surround) is the failure this mode takes over from.
#                 `takeover_clip` (profile: `takeover_clip_pct`) sets where that line is.
#   camera        the camera's own settings, always; we only measure (and say what we would
#                 have done).
#   sticky        ours from the start (PORTING_PLAN.md section 6, "Exposure: sticky"): hold
#                 exposure and gain while the image is acceptable, correct only on a sustained
#                 need (clipping 0.5 s, darkness 2 s, heavy clipping 0.2 s), decisively with
#                 the servo law until p99.5 is back in 235..247, then hold again; a cooldown
#                 between corrections. A lamp swept through the 210 deg view changes nothing.
#   servo         g1d's continuous loop (`step` every tick).
#   pinned        fixed manual exposure and gain.
#
# Every mode but `camera` and `camera_first` needs the control channel; `camera_first` needs it
# only to take over, and without one it stays on the camera's settings and says it wanted to.

MODES = ("camera_first", "camera", "sticky", "servo", "pinned")

#: UVC auto-exposure mode values (bmAutoExposureMode): 1 manual, 2 auto, 4 shutter priority,
#: 8 aperture priority (what this camera's "auto" is).
AE_MANUAL = 1

#: A write P2 makes on the control channel: (control name, value). Value None = that
#: control's own default (GET_DEF), which is how "the camera's settings" are restored.
Write = tuple

#: The camera's own settings back: gain, then its auto-exposure mode, then auto white balance.
#: NOT the exposure time: in auto-exposure the camera owns it and refuses a write (measured on
#: our wrist camera: a USB STALL, "Pipe error"), and once auto is back on its value is moot.
HAND_BACK: list = [("gain", None), ("ae_mode", None), ("wb_auto", None)]


@dataclasses.dataclass(frozen=True)
class Patience:
    """When a measured need becomes an action. First guesses from the knee table above
    (clipping rises sharply once p99.5 passes 250); tune them from recordings."""

    bright_p995: float = 250.0      # too bright: p99.5 at least this ...
    bright_clip: float = 0.01       # ... AND this much of the disc at >= 250 (not one lamp)
    bright_s: float = 0.5
    dark_p995: float = 200.0        # too dark: p99.5 below this
    dark_s: float = 2.0
    panic_clip: float = 0.10        # heavily clipped
    panic_s: float = 0.2
    cooldown_s: float = 1.5         # between the end of one correction and the next (not panic)
    correct_max_s: float = 3.0      # a correction that has not converged by now stops anyway
    # camera_first: when the CAMERA'S loop counts as failing, and for how long it may before
    # we take over (longer than ours: its own loop gets the time to react to a scene first).
    takeover_grace_s: float = 2.0   # after the stream starts, not judged at all
    takeover_clip: float = 0.15     # this much of the disc at >= 250 (a lamp or a screen: less)
    takeover_mean: float = 190.0    # ... or the disc as a whole this bright
    takeover_bright_s: float = 2.0
    takeover_dark_p995: float = 180.0   # ... or its highlights this dark
    takeover_dark_s: float = 4.0


def need_of(meas: Measurement, p: Patience) -> str | None:
    """What this frame asks for: "panic", "bright", "dark" or None (acceptable)."""
    if not meas.usable:
        return None
    if meas.clip_frac >= p.panic_clip:
        return "panic"
    if meas.p995 >= p.bright_p995 and meas.clip_frac >= p.bright_clip:
        return "bright"
    if meas.p995 < p.dark_p995:
        return "dark"
    return None


def camera_failing(meas: Measurement, p: Patience) -> str | None:
    """camera_first's judgement of the CAMERA'S loop on this frame: "bright" (a large part of
    the disc clipped, or the whole disc too bright), "dark", or None (good enough)."""
    if not meas.usable:
        return None
    if meas.clip_frac >= p.takeover_clip or meas.mean >= p.takeover_mean:
        return "bright"
    if meas.p995 < p.takeover_dark_p995:
        return "dark"
    return None


class _Spell:
    """How long the current need has lasted, unbroken. Any acceptable tick (or a different
    need) starts it over, so a lamp swept briefly through the view changes nothing. "panic"
    also counts as "bright": heavy clipping is clipping."""

    def __init__(self):
        self.need: str | None = None
        self.since = 0.0
        self.bright_since: float | None = None

    def update(self, need: str | None, now: float) -> None:
        if need != self.need:
            self.need, self.since = need, now
        if need in ("bright", "panic"):
            if self.bright_since is None:
                self.bright_since = now
        else:
            self.bright_since = None

    def lasted(self, need: str, now: float) -> float:
        if need == "bright":
            return 0.0 if self.bright_since is None else now - self.bright_since
        return now - self.since if self.need == need else 0.0


class Controller:
    """One camera's exposure: which law runs, and when. Pure (no camera, no clock, no I/O).

        ctl = Controller("camera_first", can_control=True)
        writes = ctl.begin(t)                       # at stream start
        writes = ctl.update(meas, t, current=read)  # every metered frame (10 Hz)

    `current()` returns the camera's present (exposure, gain), or None; it is called only at
    a takeover, so we start from what the camera's own loop had reached and the hand-over is
    not a visible step.
    """

    def __init__(self, mode: str = "camera_first", *, patience: Patience | None = None,
                 can_control: bool = True, pinned: tuple[int, int] = (200, 64),
                 white_balance: int | None = None):
        if mode not in MODES:
            raise ValueError(f"exposure mode {mode!r}: expected one of {MODES}")
        self.mode = mode
        self.p = patience or Patience()
        self.can_control = bool(can_control)
        self.pinned = (int(pinned[0]), int(pinned[1]))
        #: Kelvin to pin white balance at when we own the camera; None = leave it auto.
        self.white_balance = white_balance
        self.owner = "camera" if mode in ("camera_first", "camera") else "ours"
        self.state = (ServoState(*self.pinned) if mode == "pinned" else initial_state())
        self.phase = "hold"             # ours: "hold" or "correct"
        self.t_begin = 0.0
        self.t_correct = 0.0            # when the correction in progress began
        self.t_settled = -math.inf      # when the last correction ended
        self.corrections = 0
        self.spell = _Spell()           # what WE would act on (sticky's needs)
        self.judge = _Spell()           # camera_first: is the camera's own loop failing?
        self.last: Measurement | None = None
        #: camera_first: why we took over (or wanted to, without a control channel).
        self.takeover: str = ""
        self.t_takeover: float | None = None
        self.wants_takeover = False
        self.events: list[str] = []     # lines for P2 to log; P2 empties it

    # --- writes ----------------------------------------------------------------------------

    def _ours(self, st: ServoState) -> list[Write]:
        # Gain before exposure: a frame caught between the two writes is then too dark rather
        # than too bright (g1d's `ServoRunner._apply`, same asymmetry as everywhere here).
        return [("gain", int(st.gain)), ("exposure", int(st.exposure))]

    def _take(self) -> list[Write]:
        w: list[Write] = [("ae_mode", AE_MANUAL)]
        if self.white_balance is not None:
            w += [("wb_auto", 0), ("wb_temp", int(self.white_balance))]
        return w + self._ours(self.state)

    def begin(self, now: float) -> list[Write]:
        """The writes at stream start."""
        self.t_begin = now
        if self.owner == "camera":
            # The camera's own settings, whatever a previous run (ours, or another program)
            # left in its registers: they persist until it is unplugged.
            return list(HAND_BACK)
        return self._take()

    def update(self, meas: Measurement, now: float, *, current=None) -> list[Write]:
        self.last = meas
        need = need_of(meas, self.p)
        self.spell.update(need, now)
        if self.mode == "pinned":
            return []
        if self.owner == "camera":
            return self._judge_camera(now, current) if self.mode == "camera_first" else []
        if self.mode == "servo":
            return self._step(meas)
        return self._sticky(meas, now)

    def _step(self, meas: Measurement) -> list[Write]:
        nxt = step(meas, self.state)
        changed = (nxt.exposure, nxt.gain) != (self.state.exposure, self.state.gain)
        self.state = nxt
        return self._ours(nxt) if changed else []

    def _sticky(self, meas: Measurement, now: float) -> list[Write]:
        p, s = self.p, self.spell
        if self.phase == "correct":
            out = self._step(meas)
            inside = target_band(meas)
            if (inside and not self.state.saturated) or now - self.t_correct >= p.correct_max_s \
                    or (self.state.dark and not out):
                self.phase, self.t_settled = "hold", now
                self.events.append(f"settled: {describe(self.state, meas)}")
            return out
        panic = s.lasted("panic", now) >= p.panic_s
        bright = s.lasted("bright", now) >= p.bright_s
        dark = s.lasted("dark", now) >= p.dark_s
        if not (panic or ((bright or dark) and now - self.t_settled >= p.cooldown_s)):
            return []
        why = "panic" if panic else ("bright" if bright else "dark")
        self.phase, self.t_correct = "correct", now
        self.corrections += 1
        self.events.append(f"correcting ({why}): {describe(self.state, meas)}")
        return self._step(meas)

    def _judge_camera(self, now: float, current) -> list[Write]:
        p, s, m = self.p, self.judge, self.last
        s.update(camera_failing(m, p), now)
        if now - self.t_begin < p.takeover_grace_s:
            return []
        why = None
        if s.lasted("bright", now) >= p.takeover_bright_s:
            why = (f"{m.clip_frac * 100:.1f}% of the disc clipped, mean {m.mean:.0f}, for "
                   f"{s.lasted('bright', now):.1f} s")
        elif s.lasted("dark", now) >= p.takeover_dark_s:
            why = f"p99.5 only {m.p995:.0f} for {s.lasted('dark', now):.1f} s"
        if why is None:
            return []
        if not self.can_control:
            if not self.wants_takeover:
                self.wants_takeover = True
                self.takeover = why
                self.events.append(f"the camera's own exposure is not good enough ({why}), "
                                   f"but there is no control channel to take over with: "
                                   f"staying on the camera's settings")
            return []
        cur = current() if current is not None else None
        if cur is not None and cur[0] > 0:
            band, gain = _split(light_index(int(cur[0]), int(cur[1])), PREFERRED_BAND)
            self.state = ServoState(band, gain)
        self.owner, self.takeover, self.t_takeover = "ours", why, now
        self.phase, self.t_correct = "correct", now
        self.corrections += 1
        self.events.append(f"taking over from the camera's own exposure ({why}); "
                           f"from exp={self.state.exposure} gain={self.state.gain}"
                           + ("" if cur is None else f" (the camera was at exp={cur[0]} "
                                                     f"gain={cur[1]})"))
        return self._take() + self._step(self.last)

    # --- what P2 publishes -------------------------------------------------------------------

    def status(self, now: float) -> dict:
        m = self.last
        # While the camera owns exposure, what is shown as the need is camera_first's judgement
        # of it (the takeover's trigger); once we own it, sticky's.
        s = self.judge if (self.owner, self.mode) == ("camera", "camera_first") else self.spell
        return {
            "mode": self.mode, "owner": self.owner,
            "phase": self.phase if self.owner == "ours" else "",
            "exposure": self.state.exposure if self.owner == "ours" else None,
            "gain": self.state.gain if self.owner == "ours" else None,
            "need": s.need or "",
            "need_s": round(now - s.since, 1) if s.need else 0.0,
            "corrections": self.corrections,
            "takeover": self.takeover, "wants_takeover": self.wants_takeover,
            "dark": bool(self.state.dark), "noisy": bool(self.state.noisy),
            "p995": None if m is None else round(m.p995, 1),
            "mean": None if m is None else round(m.mean, 1),
            "clip_pct": None if m is None else round(m.clip_frac * 100, 2),
        }


def target_band(meas: Measurement, target: float = TARGET_P995) -> bool:
    """Is p99.5 inside the servo law's own deadband (235..247 at the default target)?"""
    return target - DEADBAND_DARK <= meas.p995 <= target + DEADBAND_BRIGHT


__all__ = [
    "AE_MANUAL",
    "DARK_REFUSE_P995",
    "EXPOSURE_BANDS",
    "GAIN_MAX",
    "GAIN_MIN",
    "MODES",
    "NOISY_GAIN",
    "PREFERRED_BAND",
    "TARGET_P995",
    "Controller",
    "Measurement",
    "Patience",
    "ServoState",
    "circle_mask",
    "dark_verdict",
    "describe",
    "gain_for",
    "gain_multiplier",
    "initial_state",
    "light_index",
    "measure",
    "need_of",
    "step",
    "target_band",
]


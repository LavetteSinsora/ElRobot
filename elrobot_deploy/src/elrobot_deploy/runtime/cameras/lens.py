"""The fisheye lens circle: found from live frames, and the square cut around it.

A 210 deg fisheye projects a circle onto the rectangular sensor; outside it is black. The
camera process (P2) finds the circle `(cx, cy, r)` from its own first frames at every start
(`fit_circle`, then `decide`), and every frame it publishes carries the square around it
(`crop_for`: the circle plus 2 px of air). A reader cuts that square (`cut`, black where it
falls off the sensor), so a consumer sees a square-cropped disc. The profile's
`fallback_lens_circle` is only a fallback for a fit that fails, and a sanity reference a fit
must land near.

Ported from g1d_deploy (`legacy_g1d/.../drivers/fisheye_circle_fit.py` and the fit half of
`board/fisheye_image_server/photometry.py`), unchanged in substance. The rationale, measured
on g1d's fisheye wrists (the same HX200S 210 deg lens as ours):

* **Fit every start.** A stored fit goes stale silently: one measured three weeks after it was
  taken agreed in radius to 0.8 px while the CENTRE was 53 px out (the camera had moved in its
  mount).
* **Repeatable** (three captures minutes apart agreed to 0.06 px) and **exposure-independent**:
  the rim is the steepest brightness DROP on rays from the centre, not a luma threshold, so
  glare softens the edge without moving it.

Rectilinear cameras (`lens = "normal"`, e.g. a future exocentric camera) have no circle and are
never fitted: their crop is the whole frame (`full_frame`).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


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



# --- the square ------------------------------------------------------------

#: Air between the fitted circle and the square, in sensor pixels.
CROP_MARGIN_PX = 2

#: Refuse a fit that disagrees with `fallback_lens_circle` by more than this, in any
#: of cx, cy, r — when the profile gives one. Not a tolerance on correctness
#: (the stored geometry has been wrong by 53 px); a guard against a fit that
#: found something that is not this lens at all: a reflection, a covered lens,
#: a swapped cable.
MAX_DRIFT_PX = 150.0

#: Frames per camera for the fit. It takes a per-pixel maximum, so more frames
#: fill in more of what a dark object happens to cover; 24 gave 0.06 px
#: repeatability on the live rig and costs under a second at 30 fps.
REFIT_FRAMES = 24

#: Seconds of frames discarded before the fit: a camera's auto-exposure (or ours) converges
#: within a few hundred ms of the stream starting.
SETTLE_S = 1.0

#: While a camera runs without a fitted circle (the fit failed at start: a covered lens, a
#: dark room), P2 tries again this often from its newest frames, and switches to the fitted
#: square when one succeeds. Every frame carries its own square, so the switch is safe.
REFIT_RETRY_S = 5.0


def crop_for(cx: float, cy: float, r: float, margin: int = CROP_MARGIN_PX
             ) -> tuple[int, int, int, int]:
    """The square a circle implies, `(x0, y0, x1, y1)` (end exclusive):
    centred on it, `margin` px of air, so the disc nearly fills it.

    The corner comes from the UNROUNDED side. May extend past the sensor, and
    must be allowed to: these circles overflow the top of a 720-row sensor, and
    clamping would put the two wrists' discs at different scales.
    """
    side = 2 * (r + margin)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    s = int(round(side))
    return (x0, y0, x0 + s, y0 + s)


def placeholder_crop(height: int, width: int) -> tuple[int, int, int, int]:
    """The square assumed when no circle is known and none was fitted (a failed fit and no
    `fallback_lens_circle`): the sensor-height square in the middle of the frame."""
    x0 = (width - height) // 2
    return (x0, 0, x0 + height, height)


def full_frame(height: int, width: int) -> tuple[int, int, int, int]:
    """A rectilinear camera's "crop": the whole sensor. Not square."""
    return (0, 0, width, height)


def cut(frame: np.ndarray, x0: int, y0: int, side: int,
        out: np.ndarray | None = None) -> np.ndarray:
    """`frame[y0:y0+side, x0:x0+side]`, with whatever falls off the sensor
    black — the square a fisheye view holds. Into `out` when given; a new array otherwise."""
    h, w = frame.shape[:2]
    if out is None:
        out = np.empty((side, side) + frame.shape[2:], frame.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx0 >= sx1 or sy0 >= sy1:
        out[...] = 0
        return out
    if (sx0, sy0, sx1, sy1) != (x0, y0, x0 + side, y0 + side):
        out[...] = 0
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


def luma(frame: np.ndarray, *, bgr: bool = True) -> np.ndarray:
    """Rec. 601 luma, uint8 (0.299/0.587/0.114 over RGB). Wire frames are BGR."""
    if frame.ndim == 2:
        return frame
    c = frame[..., ::-1] if bgr else frame
    return (0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]).astype(np.uint8)


@dataclasses.dataclass(frozen=True)
class CropDecision:
    """What P2 decided for one camera: the square it uses, and why. Logged, and shown on the
    dashboard."""

    camera: str
    used: str                                   # "fit", "fallback_lens_circle", "placeholder"
    crop: tuple[int, int, int, int]
    lens_circle: tuple[float, float, float]
    why: str = ""
    fit: dict | None = None
    drift: tuple[float, float, float] | None = None

    @property
    def warning(self) -> bool:
        return self.used != "fit"

    def line(self) -> str:
        cx, cy, r = self.lens_circle
        if self.used == "fit":
            d = ("" if self.drift is None else
                 f"; {self.drift[0]:+.1f},{self.drift[1]:+.1f},{self.drift[2]:+.1f} px "
                 f"from fallback_lens_circle")
            return (f"lens {self.camera}: cx={cx:.1f} cy={cy:.1f} r={r:.1f} "
                    f"(resid {self.fit['resid_p95']:.1f} px, "
                    f"{self.fit['inlier_frac'] * 100:.0f}% of the boundary){d}; "
                    f"square {list(self.crop)}, side {self.crop[2] - self.crop[0]}")
        if self.used == "placeholder":
            return (f"WARNING lens {self.camera}: no circle found and no fallback_lens_circle; "
                    f"using the centred square {list(self.crop)} until a fit succeeds "
                    f"(retried every {REFIT_RETRY_S:.0f} s) — {self.why}")
        return (f"WARNING lens {self.camera}: using the profile's fallback_lens_circle "
                f"({cx:.1f}, {cy:.1f}, r {r:.1f}), square {list(self.crop)} — {self.why}")

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in dataclasses.asdict(self).items()}


def decide(camera: str, frames, fallback_lens_circle=None) -> CropDecision:
    """The lens square for one fisheye camera, from `frames` (grayscale, full sensor).

    A good fit is used as is. A fit that fails, or lands more than `MAX_DRIFT_PX` from
    `fallback_lens_circle`, falls back to `fallback_lens_circle` with a WARNING; with no
    fallback either, to the centred sensor-height square (`placeholder_crop`), which P2
    replaces as soon as a later fit succeeds (`REFIT_RETRY_S`).
    """
    reference = (None if fallback_lens_circle is None
                 else tuple(float(c) for c in fallback_lens_circle))

    def fallback(why, **kw):
        if reference is None:
            h, w = frames[0].shape[:2] if len(frames) else (0, 0)
            box = placeholder_crop(h, w)
            half = (box[2] - box[0]) / 2
            return CropDecision(camera, "placeholder", box,
                                (box[0] + half, half, half - CROP_MARGIN_PX), why, **kw)
        return CropDecision(camera, "fallback_lens_circle", crop_for(*reference), reference,
                            why, **kw)

    if not frames:
        return fallback("no frames were captured for the fit")
    try:
        fit = fit_circle(frames)
    except Exception as exc:  # noqa: BLE001
        return fallback(f"the fit raised {type(exc).__name__}: {exc}")
    if not fit.ok:
        return fallback(f"the fit was refused: {fit.why}", fit=fit.as_dict())

    drift = None
    if reference is not None:
        drift = (fit.cx - reference[0], fit.cy - reference[1], fit.r - reference[2])
        if max(abs(d) for d in drift) > MAX_DRIFT_PX:
            return fallback(
                f"the fit is {max(abs(d) for d in drift):.0f} px from fallback_lens_circle, "
                f"past the {MAX_DRIFT_PX:.0f} px this trusts. That is not a camera "
                f"that drifted — check the lens is not covered, and that this is the camera "
                f"the profile means",
                fit=fit.as_dict(), drift=drift)
    circle = (float(fit.cx), float(fit.cy), float(fit.r))
    return CropDecision(camera, "fit", crop_for(*circle), circle,
                        fit=fit.as_dict(), drift=drift)


__all__ = [
    "CROP_MARGIN_PX",
    "MAX_DRIFT_PX",
    "REFIT_FRAMES",
    "REFIT_RETRY_S",
    "SETTLE_S",
    "CircleFit",
    "CropDecision",
    "crop_for",
    "cut",
    "decide",
    "fit_circle",
    "full_frame",
    "luma",
    "placeholder_crop",
]

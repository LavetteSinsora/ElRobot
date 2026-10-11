"""The fisheye lens square, fitted at every launch.

A `fisheye180` camera's lens circle `(cx, cy, r)` is fitted from this launch's
own frames, and the square around it (`crop_for`: the circle plus 2 px of air)
is what the arena holds for that camera: P2 cuts every frame to it (`cut`), so
a policy reads a square-cropped disc and applies no crop of its own. Nothing
about the square comes from the robot file; its optional `fallback_lens_circle` is
only a fallback for a fit that fails, and a sanity reference a fit must land
near.

WHEN
----

At INIT, before the arena exists, because the square's size is the camera
ring's shape. P0 reads the camera itself (`capture_source`, the same binding P2
uses), lets it run `SETTLE_S` so the board's exposure servo has converged, pools
`REFIT_FRAMES` frames and fits. The decision goes into the config every process
is handed (`CameraSpec.crop`), into the log, and into every manifest.

WHY FIT EVERY LAUNCH
--------------------

A stored fit goes **stale**, silently: one measured three weeks after it was
taken agreed in radius to 0.8 px while the CENTRE was 53 px out — a camera that
had moved in its mount. Every rollout in between was served a crop with the
disc clipped at the top, and nothing detected it.

* **Repeatable.** Three captures minutes apart agreed to 0.06 px.
* **Exposure-independent.** `photometry.fit_circle` finds the rim by its
  steepest brightness DROP, not by a luma threshold, so glare softens the edge
  without moving it.

Rectilinear cameras (`kind = "normal"`) have no lens circle and are never
touched here; their `crop` stays None, meaning "the whole frame".

For a policy author: `docs/deploying_a_policy/policy_contract.md`, "Cameras";
the robot-file keys: `docs/setup/robot_toml.md`, "`[cameras.<name>]`".
"""

from __future__ import annotations

import dataclasses

import numpy as np

from g1d_deploy.config_loading.errors import ConfigError

#: Air between the fitted circle and the square, in sensor pixels.
CROP_MARGIN_PX = 2

#: Refuse a fit that disagrees with `fallback_lens_circle` by more than this, in any
#: of cx, cy, r — when the robot file gives one. Not a tolerance on correctness
#: (the stored geometry has been wrong by 53 px); a guard against a fit that
#: found something that is not this lens at all: a reflection, a covered lens,
#: a swapped cable.
MAX_DRIFT_PX = 150.0

#: Frames per camera for the fit. It takes a per-pixel maximum, so more frames
#: fill in more of what a dark object happens to cover; 24 gave 0.06 px
#: repeatability on the live rig and costs under a second at 30 fps.
REFIT_FRAMES = 24

#: Seconds of frames discarded before the fit: the board's exposure servo
#: primes on open and converges within a few hundred ms.
SETTLE_S = 1.0


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
    """The square assumed when no circle is known and none was fitted
    (check-policy, a robot file without `fallback_lens_circle`): the sensor-height
    square in the middle of the frame. Only its SHAPE matters there."""
    x0 = (width - height) // 2
    return (x0, 0, x0 + height, height)


def cut(frame: np.ndarray, x0: int, y0: int, side: int,
        out: np.ndarray | None = None) -> np.ndarray:
    """`frame[y0:y0+side, x0:x0+side]`, with whatever falls off the sensor
    black — the square a fisheye view holds. Into `out` when given (P2 cuts
    straight into a ring slot); a new array otherwise (the replay)."""
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
    """What the launch decided for one camera: the square it uses, and why.
    Written to the log and to every rollout manifest."""

    camera: str
    used: str                                   # "fit" or "fallback_lens_circle"
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
        return (f"WARNING lens {self.camera}: using the robot file's fallback_lens_circle "
                f"({cx:.1f}, {cy:.1f}, r {r:.1f}), square {list(self.crop)} — {self.why}")

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in dataclasses.asdict(self).items()}


def decide(camera: str, spec, frames) -> CropDecision:
    """The lens square for one `fisheye180` camera, from `frames` (grayscale,
    full sensor).

    A good fit is used as is. A fit that fails, or lands more than
    `MAX_DRIFT_PX` from `fallback_lens_circle`, falls back to `fallback_lens_circle` with a
    WARNING; with no `fallback_lens_circle` to fall back to, the launch is refused
    (`ConfigError`), because there is no square to serve.
    """
    from g1d_deploy.runtime.drivers.board.fisheye_image_server import photometry

    reference = (None if getattr(spec, "fallback_lens_circle", None) is None
                 else tuple(float(c) for c in spec.fallback_lens_circle))

    def fallback(why, **kw):
        if reference is None:
            raise ConfigError(
                f"camera {camera!r}: could not find its lens circle ({why}), and the "
                f"robot file gives no `fallback_lens_circle` to fall back to. Check the "
                f"lens is uncovered and lit, or add fallback_lens_circle = {{ cx = .., "
                f"cy = .., r = .. }} to [cameras.{camera}].")
        return CropDecision(camera, "fallback_lens_circle", crop_for(*reference), reference,
                            why, **kw)

    if not frames:
        return fallback("no frames were captured for the fit")
    try:
        fit = photometry.fit_circle(frames)
    except Exception as exc:                    # noqa: BLE001 - a fallback, not a crash
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
                f"that drifted — check the lens is not covered and that the cables "
                f"are in the sockets the board's cameras.yaml names",
                fit=fit.as_dict(), drift=drift)
    circle = (float(fit.cx), float(fit.cy), float(fit.r))
    return CropDecision(camera, "fit", crop_for(*circle), circle,
                        fit=fit.as_dict(), drift=drift)


def capture_source(source, shape, n: int = REFIT_FRAMES, *, settle_s: float = SETTLE_S,
                   timeout_s: float = 5.0) -> list[np.ndarray]:
    """`n` full-sensor luma frames read straight from a camera binding (what P2
    would open: `p2_camera.make_source`), after `settle_s` of frames thrown
    away. Returns what it got by `timeout_s` past the settling, possibly none:
    a source that blocks is abandoned, never waited on forever."""
    import threading
    import time

    out: list[np.ndarray] = []
    done = threading.Event()

    def read():
        buf = np.empty(shape, np.uint8)
        try:
            source.connect()
            t_settled = time.monotonic() + settle_s
            while not done.is_set() and len(out) < n:
                if not source.read_into(buf):
                    return
                if time.monotonic() >= t_settled:
                    out.append(luma(buf).copy())
        except Exception:                        # noqa: BLE001 - the fit reports "no frames"
            return

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(settle_s + timeout_s)
    done.set()
    try:
        source.close()
    except Exception:                            # noqa: BLE001
        pass
    return list(out)


__all__ = ["CROP_MARGIN_PX", "CropDecision", "MAX_DRIFT_PX", "REFIT_FRAMES", "SETTLE_S",
           "capture_source", "crop_for", "cut", "decide", "luma", "placeholder_crop"]

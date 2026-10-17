"""P2: one camera's process. One per `[cameras.<name>]`; the one writer of that camera's ring.

    start   open the camera at its own mode (`cameras.sources`), open its control channel if
            it has one (`cameras.uvc`), put exposure where the profile's mode says
            (`cameras.exposure`, "THE MODES": by default the camera's own settings), let it
            settle, then find the lens circle from its own first frames (`cameras.lens`)
    loop    block for a frame, decode it straight into the ring's spare slot, publish it with
            its capture time, the square to cut and the exposure in effect; every third frame
            is metered inside the lens circle and handed to the exposure controller, whose
            writes go to the camera at once
    status  a few times a second, JSON in the ring's status block: the device and mode, the
            lens decision, the exposure state (who owns it, what it measures, whether and why
            it took over), the frame rate. The dashboard and P0 read it.

The camera is opened ONCE, here: a USB camera's stream can have only one reader, so the lens
fit is this process's own rather than P0's (g1d fitted at INIT, from a board's image server
any number of processes could subscribe to). The ring therefore holds whole frames and every
frame carries its square: the ring's shape does not wait for the fit, and a fit that failed at
start (a covered lens) is retried and takes effect from the next frame.

`CameraRunner` is the process minus the arena: `elrobot_deploy camera test` drives the same
code on its own and prints what it sees.
"""

from __future__ import annotations

import json
import math
from collections import deque

import numpy as np

from elrobot_deploy.runtime.cameras import CameraError, exposure, lens, make_source
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.timing import now

#: Meter one frame in this many (10 Hz at 30 fps): g1d measured the plant settling in two
#: frames, so a 100 ms sample measures a settled response (`exposure.py`, "WHY 10 Hz").
METER_EVERY = 3
#: Meter at 1/this scale (g1d: biases p99.5 by -0.16 levels at 1/2, -0.98 at 1/8).
METER_SCALE = 2
#: How often the status block is rewritten, s.
STATUS_EVERY_S = 0.5
#: While the camera's own loop owns exposure, read back what it chose this often (frames), so
#: every frame records it.
READBACK_EVERY = 15


class CameraRunner:
    """One camera, from open to close, with no arena."""

    def __init__(self, cam, *, source=None, log=print):
        self.cam = cam
        self.source = source if source is not None else make_source(cam)
        self.log = log
        self.ctl = None
        self.controls_note = ""
        self.controller: exposure.Controller | None = None
        self.decision: lens.CropDecision | None = None
        self.crop = lens.full_frame(cam.height, cam.width)
        self._mask = None
        self.frames = 0
        self.control_faults = 0
        self.readback = (-1, -1)
        self.error = ""
        self.phase = "opening"
        self._times: deque = deque(maxlen=64)
        self._refit_at = math.inf
        self._refit_frames: list | None = None

    def say(self, text: str) -> None:
        self.log(f"[P2 {self.cam.name}] {text}")

    # --- start ---------------------------------------------------------------------------

    def start(self, pixels: np.ndarray) -> None:
        cam = self.cam
        self.source.connect()
        self.say(f"streaming {self.source.describe()}")
        try:
            self.ctl = self.source.controls()
            if self.ctl is None:
                self.controls_note = "this kind of camera has no control channel here"
        except Exception as exc:  # noqa: BLE001
            self.ctl, self.controls_note = None, f"{type(exc).__name__}: {exc}"
        if self.ctl is not None:
            self.controls_note = "ok"
        elif cam.exposure not in ("camera", "camera_first"):
            raise CameraError(f"exposure = {cam.exposure!r} needs the camera's control channel, "
                              f"which is unavailable ({self.controls_note}). Use "
                              f"camera_first or camera, or fix the channel "
                              f"(`elrobot_deploy camera probe`).")
        else:
            self.say(f"no control channel ({self.controls_note}): exposure stays the "
                     f"camera's own")
        self.controller = exposure.Controller(
            cam.exposure, can_control=self.ctl is not None,
            patience=exposure.Patience(takeover_clip=cam.takeover_clip_pct / 100.0),
            pinned=(cam.pinned_exposure, cam.pinned_gain),
            white_balance=cam.white_balance or None)
        self._apply(self.controller.begin(now()))

        self.phase = "settling"
        t_end = now() + lens.SETTLE_S
        while now() < t_end:
            self._read(pixels)
        if cam.lens == "fisheye":
            self.phase = "fitting the lens"
            frames = [lens.luma(pixels).copy() for _ in range(lens.REFIT_FRAMES)
                      if self._read(pixels)]
            self._use(lens.decide(cam.name, frames, cam.fallback_lens_circle))
        else:
            self._set_mask(None)
        self.phase = "running"

    def _read(self, pixels) -> bool:
        if not self.source.read_into(pixels):
            raise CameraError(f"{self.source.describe()}: the stream stopped")
        self.frames += 1
        self._times.append(self.source.t_capture)
        return True

    def _use(self, decision: lens.CropDecision) -> None:
        self.decision = decision
        self.crop = decision.crop
        self._set_mask(decision.lens_circle)
        self.say(decision.line())
        self._refit_at = math.inf if decision.used == "fit" else now() + lens.REFIT_RETRY_S

    def _set_mask(self, circle) -> None:
        h, w = self.cam.height // METER_SCALE, self.cam.width // METER_SCALE
        if circle is None:
            self._mask = np.ones((h, w), bool)
        else:
            self._mask = exposure.circle_mask((h, w), *circle, scale=METER_SCALE)

    # --- the loop ------------------------------------------------------------------------

    def frame(self, pixels: np.ndarray) -> bool:
        """The next frame into `pixels`, metered and acted on. False = the stream ended."""
        if not self.source.read_into(pixels):
            return False
        self.frames += 1
        t = now()
        self._times.append(self.source.t_capture)
        if self.frames % METER_EVERY == 0:
            self._meter(pixels, t)
        if self.ctl is not None and self.controller.owner == "camera" and \
                self.frames % READBACK_EVERY == 0:
            self.readback = self._current() or (-1, -1)
        if t >= self._refit_at:
            self._refit(pixels, t)
        return True

    def _meter(self, pixels, t) -> None:
        import cv2
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self._mask.shape[1], self._mask.shape[0]),
                           interpolation=cv2.INTER_AREA)
        meas = exposure.measure(small, self._mask)
        self._apply(self.controller.update(meas, t, current=self._current))
        for line in self.controller.events:
            self.say(line)
        self.controller.events.clear()

    def _refit(self, pixels, t) -> None:
        if self._refit_frames is None:
            self._refit_frames = []
        self._refit_frames.append(lens.luma(pixels).copy())
        if len(self._refit_frames) < lens.REFIT_FRAMES:
            return
        frames, self._refit_frames = self._refit_frames, None
        decision = lens.decide(self.cam.name, frames, self.cam.fallback_lens_circle)
        if decision.used == "fit":
            self._use(decision)
        else:
            self._refit_at = t + lens.REFIT_RETRY_S

    def _current(self):
        try:
            return int(self.ctl.get("exposure")), int(self.ctl.get("gain"))
        except Exception:  # noqa: BLE001
            return None

    def _apply(self, writes) -> None:
        if self.ctl is None:
            return
        for name, value in writes:
            try:
                self.ctl.write(name, value)
            except Exception as exc:  # noqa: BLE001
                self.control_faults += 1
                if self.control_faults in (1, 10) or self.control_faults % 200 == 0:
                    self.say(f"control write {name}={value} failed (#{self.control_faults}): "
                             f"{exc}")

    # --- what is published ---------------------------------------------------------------

    def exposure_in_effect(self) -> tuple[int, int, int]:
        """(owner, exposure, gain) for the frame just read."""
        c = self.controller
        if c is not None and c.owner == "ours":
            return schema.EXPOSURE_OURS, int(c.state.exposure), int(c.state.gain)
        return schema.EXPOSURE_CAMERA, int(self.readback[0]), int(self.readback[1])

    def fps(self) -> float:
        ts = [t for t in self._times if t > 0]
        if len(ts) < 2 or ts[-1] <= ts[0]:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    def status(self) -> dict:
        t = now()
        d = self.decision
        out = {
            "name": self.cam.name, "phase": self.phase, "error": self.error,
            "device": getattr(self.source, "describe", lambda: "")() if self.phase != "opening"
            else self.cam.device,
            "fps": round(self.fps(), 1), "frames": self.frames,
            "dropped": int(getattr(self.source, "dropped", 0)),
            "lens": self.cam.lens, "crop": list(self.crop),
            "lens_used": d.used if d else "", "lens_circle": list(d.lens_circle) if d else None,
            "lens_line": d.line() if d else "",
            "controls": self.controls_note, "control_faults": self.control_faults,
            "readback": list(self.readback),
        }
        if self.controller is not None:
            out["exposure"] = self.controller.status(t)
        return out

    def close(self) -> None:
        """Hand the camera back as we found it: if we ever owned its exposure, its own
        settings again (they persist in the camera until it is unplugged)."""
        try:
            if self.ctl is not None and self.controller is not None and \
                    self.controller.owner == "ours":
                self._apply(exposure.HAND_BACK)
                self.say("exposure handed back to the camera")
        finally:
            try:
                if self.ctl is not None:
                    self.ctl.close()
            finally:
                self.source.close()


def run(shm_name: str, cfg, *, camera: str, ready=None, source=None) -> None:
    """P2 as P0 starts it. `ready` is set once the lens is decided and frames are flowing;
    a camera that cannot start writes why into its status and exits (P0 reports it)."""
    from elrobot_deploy.runtime.shared_memory import Arena, layout_for, open_all

    arena = Arena.attach(shm_name, layout_for(cfg))
    try:
        rings = open_all(arena)
        control, ring = rings["control"], rings["cameras"][camera]
        runner = CameraRunner(cfg.camera(camera), source=source,
                              log=lambda s: print(s, flush=True))

        def publish_status():
            ring.set_status(json.dumps(runner.status()))

        try:
            publish_status()
            runner.start(ring.spare()["pixels"])
            publish_status()
            if ready is not None:
                ready.set()
            t_status = now()
            while control.session < schema.STOPPED:
                slot = ring.spare()
                if not runner.frame(slot["pixels"]):
                    raise CameraError(f"{runner.source.describe()}: no frame for 2 s "
                                      f"(unplugged?)")
                owner, exp, gain = runner.exposure_in_effect()
                ring.publish(runner.source.t_capture, runner.source.t_recv, runner.crop,
                             owner=owner, exposure=exp, gain=gain)
                if now() >= t_status:
                    publish_status()
                    t_status = now() + STATUS_EVERY_S
        except Exception as exc:  # noqa: BLE001
            runner.error, runner.phase = f"{type(exc).__name__}: {exc}", "stopped"
            runner.say(f"stopped: {runner.error}")
            publish_status()
        finally:
            try:
                runner.close()
            except Exception as exc:  # noqa: BLE001
                runner.say(f"closing: {exc}")
            if not runner.error:
                runner.phase = "stopped"
                publish_status()
    finally:
        arena.close()


__all__ = ["CameraRunner", "run"]

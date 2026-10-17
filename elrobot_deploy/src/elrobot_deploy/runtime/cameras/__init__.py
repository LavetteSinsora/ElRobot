"""Cameras: where frames come from (`sources`), the fisheye lens square (`lens`), exposure
(`exposure`) and a USB camera's controls (`uvc`). The process that runs one camera is
`runtime/processes/p2_camera.py`; the profile's `[cameras.<name>]` tables declare them.

A camera is a profile table, not code: a second camera (an exocentric view, say) is another
`[cameras.<name>]` with its own source, mode and lens, and gets its own P2, its own arena ring
and its own pane on the dashboard. A new KIND of camera is one class with the source contract
(`sources.py`, top) added to `SOURCES` below.
"""

from __future__ import annotations

from elrobot_deploy.runtime.cameras.sources import (AvfCamera, CameraError, FakeCamera,
                                                    RealSenseCamera, V4l2Camera,
                                                    default_usb_source)


def _avf(cam):
    return AvfCamera(cam.device, cam.shape, fps=cam.fps)


def _v4l2(cam):
    return V4l2Camera(cam.device, cam.shape, fps=cam.fps)


def _realsense(cam):
    return RealSenseCamera(cam.shape, serial=cam.device, fps=cam.fps)


def _fake(cam):
    circle = cam.fallback_lens_circle if cam.lens == "fisheye" else None
    return FakeCamera(cam.shape, fps=cam.fps, circle=circle)


#: `source` -> a factory taking the camera's `CameraConfig`. "usb" means this machine's own
#: USB camera binding (`avfoundation` on a Mac, `v4l2` on Linux).
SOURCES = {"avfoundation": _avf, "v4l2": _v4l2, "realsense": _realsense, "fake": _fake}


def make_source(cam):
    """The binding for one `CameraConfig`."""
    kind = default_usb_source() if cam.source == "usb" else cam.source
    try:
        factory = SOURCES[kind]
    except KeyError:
        raise CameraError(f"camera {cam.name!r}: unknown source {cam.source!r} "
                          f"(have usb, {', '.join(SOURCES)})") from None
    return factory(cam)


__all__ = ["SOURCES", "CameraError", "make_source"]

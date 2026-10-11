"""P2… — the camera readers.

Each instance is the one writer of its camera's ring(s); the sources are
`runtime/drivers/cameras.py`. Cameras as the robot file declares them:
`docs/setup/robot_toml.md`, "`[cameras.<name>]`".

One module, one process per declared camera. There is nothing per-camera in the
loop: which cameras exist is `cfg.cameras`, the resolution is that camera's
region, and where the frames come from is that camera's `CameraSpec.source` —
the robot board's image server over ZMQ, or a camera on the deploy PC's own
USB. A separate module per camera would be copies of four lines that must
not drift apart, and the source classes keep that true across rigs.

The loop is: block for a frame, stamp its arrival, decode it **into the spare
slot** of the camera's ring, publish. Decoding into the slot means a ~1 MB frame
is written once, in place — the slot's sequence counter is odd for the whole
decode, which is what tells a reader not to trust it (`shared_memory.rings.CameraRing`).
Nothing here resizes a frame: any resize belongs to the consumer (the policy's
own preprocessing).

**The one crop: a `fisheye180` camera** is decoded into a private scratch frame
and the square around its lens circle — fitted at INIT, before this process
started (`CameraView.x0/y0/width`) — is cut into the slot
(`fisheye_circle_fit.cut`, black off the sensor). The JPEG is the whole wire
frame, as the robot sent it.

**The one split: a `side_by_side` camera** (the G1-D head, 480x1280, two eyes).
It is decoded once into a private scratch frame and its two halves are copied
into two rings, `<name>_left` (columns 0..W/2, the left eye) and `<name>_right`
(`resolved.SPLIT_HALVES` says how the eye was checked). One extra ~1.8 MB of
memcpy per head frame (~0.2 ms), paid here so that no policy, reader or anchor
ever sees a binocular frame. Both halves carry the same `t_recv` and
`source_seq`, and the left one also carries the wire JPEG — the whole
binocular frame, which is what the recorder stores (`CameraView.carries_jpeg`).

`ring.publish()` is what stamps the header, so the metadata cannot be separated
from the pixels it describes — it lives inside the same slot.
"""

from __future__ import annotations

import numpy as np

from g1d_deploy.runtime import cpu_pinning as affinity
from g1d_deploy.runtime.shared_memory import Arena, build, open_all, schema
from g1d_deploy.config_loading.resolved import DeployConfig, Hardware, views_of
from g1d_deploy.config_loading.derive import derive
from g1d_deploy.runtime.drivers import fisheye_circle_fit
from g1d_deploy.runtime.drivers.cameras import RealSenseCamera, V4l2Camera, ZmqCamera

# Which of P2/P3/P4 a camera is in the CPU pin table — the pin, and nothing
# else, depends on it. An unlisted camera name simply finds no core in
# the machine file and runs unpinned rather than refusing to start.
#
# **P4 is the head's, and shares with nothing.** Two decode loops on one core
# is not a refusal anywhere; it is one camera's frames arriving late, which
# reads as a rig fault.
#
# A NEW CAMERA NEEDS NO ENTRY HERE. `run` pins `PROCESS.get(camera, camera)`,
# so a camera absent from this table is pinned by its own name — give `exo` a
# core under the machine file's `[cpu.pin]` and it is pinned, with no change to
# this file. The three entries below exist only to keep the P2/P3/P4 names
# of `cpu_pinning.EFFICIENCY`, not because the mechanism needs them.
PROCESS = {"rwrist": "p2_camera", "lwrist": "p3_camera", "head": "p4_camera"}


def make_source(cfg: DeployConfig, hw: Hardware, camera: str):
    """The binding for one camera, chosen by its `CameraSpec.source`.

    Here rather than in `runtime/drivers/cameras.py` because it is the only place that needs
    both halves of the config — the task's `CameraSpec` and the machine's ports —
    and because "which source" is P2's decision, while `runtime/drivers/` holds bindings
    that know nothing about either config object.
    """
    spec = cfg.cameras[camera]
    shape = (spec.height, spec.width, 3)
    if spec.source == "realsense":
        return RealSenseCamera(shape, serial=spec.device)
    if spec.source == "v4l2":
        return V4l2Camera(spec.device, shape)
    # The spec's own host when the robot file names one; the machine file's
    # robot IP otherwise, which is every camera on a G1-D board.
    return ZmqCamera(spec.host or hw.camera_host, hw.camera_ports[camera], shape)


def _loop_split(control, rings: list, views: list, source) -> None:
    """A side_by_side camera: decode once, publish each half to its own ring.

    The scratch frame is private, so the decode can take as long as it takes
    without holding either ring's slot odd; each slot is odd only for its own
    ~0.9 MB copy. Published left then right, back to back: a reader of both
    eyes asking for the frame nearest an instant gets the same frame from each.
    """
    spec_h, spec_w = views[0].source_height, views[0].source_width
    scratch = np.empty((spec_h, spec_w, 3), np.uint8)
    while control.session < schema.STOPPED:
        if not source.read_into(scratch):
            return
        t_recv = source.t_recv
        jpeg = getattr(source, "jpeg", None)
        seq = getattr(source, "seq", None)
        for ring, view in zip(rings, views):
            ring.spare()["pixels"][...] = scratch[:, view.x0:view.x0 + view.width]
            ring.publish(t_recv, jpeg if view.carries_jpeg else None, seq)


def _loop_crop(control, ring, view, source) -> None:
    """A fisheye camera: decode the sensor, publish the square around its lens."""
    scratch = np.empty((view.source_height, view.source_width, 3), np.uint8)
    while control.session < schema.STOPPED:
        if not source.read_into(scratch):
            return
        fisheye_circle_fit.cut(scratch, view.x0, view.y0, view.width,
                               out=ring.spare()["pixels"])
        ring.publish(source.t_recv, getattr(source, "jpeg", None),
                     getattr(source, "seq", None))


def _loop(control, region, source) -> None:
    while control.session < schema.STOPPED:
        if not source.read_into(region.spare()["pixels"]):
            return
        # The compressed source rides in the SAME slot as the pixels it decoded
        # to, so the two are one frame by construction. A source
        # that has none, or a frame over the slot's capacity, records as
        # "not captured" rather than failing the run.
        #
        # `seq` rides there too, for the same reason and with the same
        # tolerance: a source without a counter publishes None and the slot
        # records -1. Only the fisheye image server has one; the vendor's bare JPEG
        # and the fakes do not.
        region.publish(source.t_recv, getattr(source, "jpeg", None),
                       getattr(source, "seq", None))


def run(shm_name: str, cfg: DeployConfig, hw: Hardware, *, camera: str,
        source=None) -> None:
    """Entrypoint. `camera` names one of `cfg.cameras` — the robot file's, a
    wire stream — and this process fills every view cut from it (one ring, or
    two for a side_by_side camera). The source is injectable so the loop can be
    driven from `drivers/cameras.py::FakeCamera` with no board."""
    layout = build(cfg, derive(cfg, hw))
    arena = Arena.attach(shm_name, layout)
    r = open_all(arena, cfg)
    affinity.pin(hw, PROCESS.get(camera, camera))

    views = views_of(cfg.views, camera)
    rings = [r["cameras"][v.name] for v in views]
    if source is None:
        source = make_source(cfg, hw, camera)

    try:
        source.connect()
        if len(views) == 1 and views[0].cropped:
            _loop_crop(r["control"], rings[0], views[0], source)
        elif len(views) == 1 and views[0].width == views[0].source_width:
            _loop(r["control"], rings[0], source)
        else:
            _loop_split(r["control"], rings, views, source)
    finally:
        source.close()
        arena.close()

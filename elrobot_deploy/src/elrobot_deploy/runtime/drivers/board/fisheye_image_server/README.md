# The fisheye image server

**Read this before touching cameras on the G1-D.**

This directory is the image server elrobot_deploy runs on the robot's board for a
rig with fisheye wrist cameras. It is shipped in the package and copied to the
robot file's `[image_server] dir` on the board (default
`/home/unitree/fish_eye_image_server/`), where it runs under the robot file's
conda environment (default `tv`). The board's address and login are the robot
file's `[board]`; both tables are in `docs/setup/robot_toml.md`.

- **Code:** `publisher.py` (capture and publish, the wire header),
  `photometry.py` (the exposure servo and the lens-circle fit), `cameras.yaml`
  (the board's cameras); started and kept in sync by
  `src/elrobot_deploy/runtime/drivers/board/image_server_launch.py`; read on the
  deploy box by `src/elrobot_deploy/runtime/drivers/cameras.py` (`ZmqCamera`).

> ### Which server should I be running?
>
> | rig | robot file `[image_server] type` | server | ports |
> |---|---|---|---|
> | fisheye wrists (1280×720) | `fisheye` | **this one** | 55565 / 55566 / 55567 |
> | stock cameras, the vendor's teleop | `teleimager` | the vendor's `teleimager` | 55555 / 55556 / 55557 |
>
> The stock server lives at
> `/home/unitree/unitree_eai_environment/service/teleimager/`, is a systemd unit
> (`teleimager.service`), and is started with `sudo systemctl start teleimager`.
> **It is the board's default.** This server is used only with a fisheye rig.

**`cameras.yaml` holds the reference robot's values**: its USB sockets, serials
and lens circles. Another unit needs its own; see "Bring-up, in order".

## ⚠️ The two servers cannot run at the same time

Not a preference — a hard constraint. The vendor's `image_server.py` contains:

```python
def reload_uvc_driver():
    subprocess.run("sudo modprobe -r uvcvideo", shell=True, check=True)
    subprocess.run("sudo modprobe uvcvideo debug=0", shell=True, check=True)
```

It removes the `uvcvideo` kernel module **globally** at startup, and its unit is
`Restart=always`. Any restart of `teleimager` rips the module out from under any
other capture process on the board. Exactly one server owns the cameras.

Both also bind their own ports, so running both gives you a bind error at best
and a silent 480×640 stream where you expected 720p at worst. That is why this
server deliberately uses a different port range.

## Why this exists, given teleimager already works

The vendor's ZMQ path is already pure MJPEG passthrough — for `type: uvc` it
writes `frame.jpeg_buffer` (the sensor's own JPEG) straight to the socket, no
decode and no re-encode. There is nothing to improve about the transport, and
this server does the same thing.

Two things the vendor cannot do:

1. **Camera controls.** `cam_config_server.yaml` has no exposure, gain or
   white-balance field. Serving systematically brighter or dimmer images than
   a policy was made for is a skew nothing reports. `cameras.yaml` carries the
   controls, sets them, **reads them back, and refuses to start on a
   mismatch.**
2. **A sequence number.** ZMQ PUB drops silently under back-pressure — by design,
   since both ends use a high-water mark of 1 so the newest frame wins. With
   the vendor's server a dropped frame is invisible; the header below makes it
   countable.

Everything else is deliberately absent: no WebRTC, no crop, no resize, no colour
conversion. The crop is the deploy box's
(`src/elrobot_deploy/runtime/drivers/fisheye_circle_fit.py`, fitted at every
launch) and any resize is the policy's.

## Photometry: why the wrists do not use the camera's own auto-exposure

**`photometry: servo` is the default for the wrists** (`photometry.py`).

A fisheye projects a circle onto a rectangular sensor, so **~60% of the frame
the camera meters is the black surround around the lens** — pixels no exposure
setting can brighten. The camera's auto-exposure averages the whole frame, finds
it dark, and raises exposure and gain until the part that is a picture goes
through the ceiling. Measured on both wrists of the reference robot:

| | full-frame mean | disk mean | disk clipped ≥250 |
|---|---|---|---|
| `auto` | 67.7 / 67.9 | 150.4 / 163.4 | **26.3% / 26.3%** |
| `servo` | — | 141.7 / 158.7 | **0.00% / 0.11%** |
| well-exposed reference recordings (tuned manual controls) | — | 149 / 155 | 1.5% / 4.5% |

The head camera is the control: same room, same light, same `Auto Exposure
Mode: 8`, but rectilinear with no black surround — 0.16% clipped. **The room is
fine; the metering window is the fault.** The head therefore stays on `auto`,
and any camera with no `circle:` in `cameras.yaml` falls back to it.

`servo` meters **only inside the lens circle** and holds its 99.5th percentile
at 245, just under the clipping knee (measured across 1798 recorded frames:
below 250 clipping is ~0%, at 250–254 it is 3.6%, past 254 it is 27%). It
chooses among three exposure bands — 10/20/30 ms, whole multiples of the 10 ms
mains flicker period, all below the 33 ms frame-rate cliff — and trims with
gain.

The camera cannot be asked to do this itself: probed live, it exposes **no
metering-region control of any kind**.

Cost, measured on the board: one frame in three gets a half-scale grayscale
decode (2.30 ms against a 33 ms frame period), after the frame has already been
sent. The pass-through path is untouched, and the two servo threads measure
2.0% and 0.7% of a core. `ServoRunner.offer` swallows its own exceptions — a
photometry fault must cost a tick, never an image server.

To go back: set `photometry: auto` (or `pinned`) on the wrists in
`cameras.yaml`. Both blocks are there and work.

## Wire format

One ZMQ PUB socket per camera. One message per frame: a 16-byte header, then the
camera's own JPEG verbatim.

| offset | size | field | value |
|---|---|---|---|
| 0 | 4 | magic | `b"G1DF"` |
| 4 | 1 | version | `1` |
| 5 | 1 | flags | `0`, reserved |
| 6 | 2 | pad | `0` |
| 8 | 8 | seq | `uint64` little-endian |
| 16 | — | jpeg | verbatim from the sensor |

**No timestamp**, deliberately: a board timestamp only means something against a
board↔box clock mapping, and the two clocks are unsynchronised free-running
monotonics. `version` and `flags` leave room to add one without a flag day.

**`seq`** counts frames *captured and assigned*, from 0, never reset while the
process lives. It is not rolled back when the socket drops a frame at its
high-water mark — so a gap seen by the subscriber means exactly "a frame existed
that you did not receive", whether it was dropped by ZMQ or lost on the network.
It does **not** count frames the sensor dropped before capture.

Clients must tolerate both formats. A bare JPEG starts `FF D8`, never `G1DF`, so
the two are distinguishable with no ambiguity — `publisher.parse_header()` is the
reference implementation.

## What on the deploy box depends on this

**The board is only half of it.** Listed so the coupling is visible when any of
it changes.

| # | where | what it does |
|---|---|---|
| 1 | `src/elrobot_deploy/runtime/drivers/cameras.py` | `ZmqCamera` strips the 16-byte header and exposes `seq`, auto-detecting on the magic and falling back to bare JPEG so the vendor server keeps working. It imports `parse_header` from `publisher.py`, so the wire format has one definition. |
| 2 | the robot file (`src/elrobot_deploy/robot/profiles/g1d_fisheye_dex1.toml`) | `[cameras.*] port` = 55565 / 55566 / 55567 and each wrist's `shape` and `fallback_lens_circle`. They must agree with `cameras.yaml`. |
| 3 | `src/elrobot_deploy/runtime/shared_memory/schema.py`, `rings.py` | `camera_dtype` carries `source_seq` beside the frame it numbers (`-1` = source has no counter); `CameraRing.peek_counters()` reads `(frame_id, source_seq)` without copying pixels. |
| 4 | `src/elrobot_deploy/runtime/launcher/startup_checks.py` | CHECK reports transit drops per camera — `source_seq` counts what the server sent, `frame_id` what P2 received, so the difference is what ZMQ discarded. Reported, never refused. |
| 5 | `src/elrobot_deploy/runtime/drivers/board/image_server_launch.py` | SSH autostart at CHECK: stops `teleimager`, syncs this directory, starts this server, confirms the process. `ELROBOT_NO_BOARD_AUTOSTART=1` skips it. Only runs when a camera declares `source = "zmq"` and the robot file says `[image_server] type = "fisheye"`. |

Things that specifically do **not** depend on it:

- **The arena.** `jpeg_capacity` is derived as `h*w*3/4` = 691 KB at 720p,
  against measured frames of 50–150 KB.
- **The lens square and a policy's image transform.** The box-side geometry is
  untouched by where the bytes came from.

### The head is served, recorded, and observed only if a policy asks

This server publishes the head so one server covers every camera on the board.
A robot file that declares it gives it an arena ring, its own reader process, a
pane on the live dashboard, and a stream in every rollout record. Whether a
policy *observes* it is the policy's (`uses_cameras`,
`docs/deploying_a_policy/policy_contract.md`, "Cameras"): a camera that is
declared and not observed never enters the anchor's wait and cannot delay or
hang a run that does not use it.

**Both eyes are recorded.** The whole 480x1280 binocular frame goes to disk,
because the recorder stores the wire JPEG verbatim and there is no such thing as
half a JPEG. In the arena it is two views, `head_left` and `head_right`.

## Installing on the board

Nothing to do by hand: at every CHECK, P0 hashes this directory's `.py`/`.yaml`
files against the robot file's `[image_server] dir`, copies any that differ, and
restarts a publisher that was not started from exactly this code
(`image_server_launch.ensure_running`). So an edit to those files here reaches
the board at the next launch, and a hand edit on the board is overwritten. To
install without a run:

```bash
# from the elrobot_deploy checkout; the target is the robot file's [image_server] dir
rsync -av src/elrobot_deploy/runtime/drivers/board/fisheye_image_server/ \
      unitree@192.168.123.164:/home/unitree/fish_eye_image_server/
```

No dependencies to install on the reference board. Its `tv` env already has
everything: Python 3.10.19, pyuvc 1.0.0b7, pyzmq 27.1.0, cv2 4.11.0, numpy
1.26.4, pyyaml. (The system `python3.8` has cv2 but **no** pyzmq — always use
the conda env.)

## Bring-up, in order

```bash
ssh unitree@192.168.123.164
source /home/unitree/miniconda3/etc/profile.d/conda.sh && conda activate tv
cd /home/unitree/fish_eye_image_server
```

**1 — find the cameras.** Safe while `teleimager` is running; claims nothing.

```bash
python publisher.py --list-devices
```

Copy each camera's `physical_path` into `cameras.yaml` (in the checkout, so the
sync does not undo it). Use `physical_path`, not `serial_number`: fisheye pairs
have been seen reporting **duplicate serials**, and pyuvc's `uid` is
`bus:device_address`, which changes on every replug.

**2 — stop the vendor server.** Everything past here needs the cameras free.

```bash
sudo systemctl stop teleimager
```

**3 — probe modes and controls.**

```bash
python publisher.py --probe
```

Confirms each unit offers `1280x720 @ 30 MJPG` (marked `<= configured`), and
dumps every control's exact name, current value, range and step. This is also
how a reference rig's settings get read off and written into `cameras.yaml`.

**4 — run it.**

```bash
python publisher.py
```

It logs per-camera rate, `seq` and HWM drops every 10 s. Expect ~30 Hz and
`hwm_dropped` near zero with a subscriber attached.

In a normal rollout you do not do this by hand: elrobot_deploy's CHECK phase stops
`teleimager` and starts this over SSH itself. Set `ELROBOT_NO_BOARD_AUTOSTART=1` to
leave the board alone and drive it manually.

**4a — the lens squares** need no step here: elrobot_deploy fits each fisheye's
lens circle from live frames at every launch, at INIT, and falls back to the
robot file's `fallback_lens_circle` only if the fit fails. Keep the workspace
lit at launch — the fit works from the per-pixel maximum over frames. Put the
fitted circle into `cameras.yaml`'s `circle:` too: it is what the exposure
servo meters.

**5 — verify from the deploy box**: `elrobot_deploy check` — ≥29.5 Hz per wrist,
no transit drops. Then the real gate for a learned policy — one captured frame
through the policy's image transform against a frame it was made from through
the same function: same shape, same dtype, lens circle in the same place.

## Switching the rig back

**To the stock 480×640 cameras**, whichever server you use, the resolution is
declared in two places and both must move together:

1. board — in `cameras.yaml`, delete the two 720×1280 wrist entries (or set
   `enable_zmq: false` on them), then uncomment the stock block at the bottom:

   ```bash
   sed -i '/BEGIN STOCK/,/END STOCK/s/^# //' cameras.yaml
   ```

   The block is contiguous and holds no prose, so this is mechanical. Three
   details make the command safe, and all three are load-bearing:

   - each marker string appears exactly **once** in `cameras.yaml` — this
     command lives here rather than in the file so the range cannot match its
     own instructions;
   - the markers are written `## ---`, not `# ---`, so stripping one `# ` leaves
     them as comments. Uncommented, `--- BEGIN STOCK ---` would be a YAML
     **document separator** and the file would parse as two documents;
   - `s/^# //` rather than `s/^# \?//`, because `\?` is a GNU extension. The
     board is Ubuntu so GNU sed is what runs there, but the portable form works
     on both and every line in the block is `# `-prefixed.

2. deploy — a robot file whose wrist cameras are `kind = "normal"`,
   `shape = [480, 640]` (no lens circle; with `kind = "fisheye180"` left in
   place the fisheye crop would be applied to a rectilinear frame and nothing
   would raise).

**To the vendor server entirely:**

```bash
sudo systemctl start teleimager     # restores 55555/6/7 at the stock geometry
```

and use a robot file with `[image_server] type = "teleimager"` and the vendor's
ports (`src/elrobot_deploy/robot/profiles/g1d_normal_dex1.toml`). Nothing on the
board needs undoing — this server never edits the vendor's config and never
touches the `uvcvideo` module.

## Board facts worth not rediscovering

Surveyed read-only on the reference robot's board, 2026-08-28.

- Jetson, `5.10.104-tegra`, aarch64, Ubuntu 20.04, 8 cores, 15 GB.
- **There is no `/dev/video*` on this board, even as root.** The cameras are
  claimed from userspace through `usbfs` by pyuvc, which detaches the kernel
  driver. `uvcvideo` is loaded with use-count 0. Any plan built on
  `cv2.VideoCapture` and `/dev/v4l/by-path/` is wrong here.
- Stock cameras, as surveyed:

  | sysfs | uid | USB id | serial | role |
  |---|---|---|---|---|
  | `1-2.1` | `1:4` | `1bcf:2d4f` | `01.00.00` | head, 480×1280 binocular |
  | `1-3.2` | `1:7` | `0001:0001` | `JR0001` | left wrist |
  | `1-3.1` | `1:5` | `0002:0002` | `JR0002` | right wrist |

- Unitree's published `cam_config_client.yaml` claims wrist serials
  `200901010001/2`. The board says `JR0001`/`JR0002`. **Trust the board
  (`--list-devices`), not a vendored config.**

# Robot files: what the robot is

A robot file describes one physical rig: the G1-D's joints, gains, limits,
model and rates, plus what is bolted onto this one — which cameras exist,
which hand, who serves the board's cameras. A deploy file picks one by name or
path (`robot = ...`).

- **Code:** `src/g1d_deploy/config_loading/robot_file.py` (loading, `extends`
  merge, refusals), `src/g1d_deploy/config_loading/resolved.py`
  (`load_hardware` → `Hardware`, `CameraSpec`), `src/g1d_deploy/robot/end_effectors.py`.
- **Profiles:** `src/g1d_deploy/robot/profiles/*.toml`.
- **Related:** `examples/robot_custom.toml`, `docs/setup/machine_toml.md` (the box,
  not the robot), `docs/deploying_a_policy/policy_contract.md` ("Cameras").

## Shipped profiles and `extends`

`g1d_base.toml` is the G1-D itself (no hand, no cameras). Each rig profile
says `extends = "g1d_base"` and adds what exists:

| profile | cameras | image server | hand |
|---|---|---|---|
| `g1d_fisheye_dex1` | `lwrist`, `rwrist` (720×1280 fisheye), `head` (480×1280, split into `head_left`/`head_right`) | ours, started over SSH | Dex-1 |
| `g1d_fisheye_dex1_exo` | + `exo` (a RealSense on this box; needs `scripts/install.sh --realsense`) | ours | Dex-1 |
| `g1d_normal_dex1` | `head` only | the vendor's teleimager | Dex-1 |
| `g1d_normal_brainco` | `head` only | teleimager | BrainCo (no policy contract yet: a policy is refused) |

**Your own rig** is a file that extends one and changes only what differs:

```toml
extends = "g1d_fisheye_dex1"        # a shipped name, or a path relative to this file

[cameras.table]                     # add a camera
source = "v4l2"
device = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.0-video-index0"
shape = [720, 1280]
```

Tables merge key by key, this file winning; arrays are replaced whole.
Unknown keys are refused. `g1d_deploy check --robot path/to/file.toml`
verifies it against the real robot, read-only.

**A robot file lists the cameras that exist.** CHECK waits for every one, so
list only what is plugged in. There are no camera on/off flags: a rig with
one more camera is one more file.

## `[end_effector]`

```toml
[end_effector]
type = "dex1"                                     # dex1 | brainco
tcp = [0.184759, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]    # optional: flange -> TCP, [x y z qw qx qy qz]
```

- **`type`** — everything else about a hand (its DDS topics, travel, which end
  is closed, whether a policy can drive it) is built in
  (`end_effectors.py`). Dex-1: `rt/dex1/{left,right}/{cmd,state}`, one jaw
  motor per hand, policy units gear radians 0 (closed) … 5.4 (open). BrainCo:
  wire implemented, policy contract not decided, so a policy is refused on it.
- **`tcp`** — the tool-centre point in the flange (wrist-yaw link) frame, the
  same for both arms; policies read it as `robot.tcp`. The Dex-1's is the
  midpoint of the jaw tips, 0.184759 m along +x. It does not change what a
  policy commands (flange poses); **it makes every IK solver drive the TCP**
  (`docs/infra_logic/ik_solvers.md`). A file that changes `type` without its own `tcp`
  inherits none.

## `[cameras.<name>]`

| key | meaning |
|---|---|
| `source` | `zmq` (the robot board's image server), `v4l2` (this box's USB), `realsense` (this box; needs the `--realsense` install) |
| `port`, `host` | zmq: the port; `host` defaults to `[board] ip` |
| `device` | v4l2: a `/dev/v4l/by-path/...` node (the fisheye units share serials, and `/dev/videoN` is not stable across reboots) |
| `serial` | realsense: the device serial |
| `shape` | `[height, width]` as the camera sends it |
| `hz` | frame rate; default 30. The anchor waits at most half a period for a fresher frame, so a wrong value costs latency |
| `kind` | `normal` or `fisheye180`. A fisheye has a lens circle, fitted from live frames at every launch; the arena holds only the square around it (the circle plus 2 px). Nothing about the square is configured |
| `fallback_lens_circle` | fisheye, optional: a known lens circle `{ center_x, center_y, radius }` (full-sensor pixels). The launch falls back to it, with a warning, when the fit fails, and refuses a fit landing more than 150 px from it as "not this lens". Without it a failed fit refuses the launch |
| `layout` | `"side_by_side"`: one frame holds two eyes; the runtime splits it into `<name>_left` / `<name>_right` views (left eye = left columns). Recorded whole |
| `intrinsics`, `extrinsics` | optional calibration, passed to the policy untouched (`Policy.cameras`); the infra reads neither |

Camera names become the names a policy uses (`obs.frames["rwrist"]`), except
split cameras, which are used by their view names.

**Check the wrists by eye.** Two identical cameras cannot be told apart in
software, so which port is the left wrist depends on how this unit is cabled.
On the first launch, look at the live dashboard: the `lwrist` pane must show
the left arm's view. If the two are crossed, extend the profile and exchange
the ports:

```toml
extends = "g1d_fisheye_dex1"

[cameras.lwrist]
port = 55567        # the profile's rwrist port
[cameras.rwrist]
port = 55566        # the profile's lwrist port
```

## `[board]`

The robot's onboard computer. `g1d_base` carries the G1-D's factory values, so
a rig file states this only for a unit that differs.

```toml
[board]
ip = "192.168.123.164"        # where the board's cameras are read from
ssh_user = "unitree"          # the login P0 uses to start the image server
# ssh_password = "..."        # default: the factory password
```

- **What the login is for:** only to start (and keep in sync) the fisheye
  image server on the board, below. With `[image_server] type = "teleimager"`
  or `"none"` it is never used.
- **One shell's override:** `G1D_ROBOT_IP`, `G1D_BOARD_SSH_USER`,
  `G1D_BOARD_SSH_PASSWORD`. Prefer the variable for a changed password: a
  robot file is copied verbatim into every rollout's manifest.

## `[image_server]`

```toml
[image_server]
type = "fisheye"                             # fisheye | teleimager | none
dir = "/home/unitree/fish_eye_image_server"  # fisheye: where it is on the board
conda = "/home/unitree/miniconda3"           # fisheye: the board's conda
conda_env = "tv"                             # fisheye: env with pyuvc, pyzmq, pyyaml
```

- **`fisheye`** is ours
  (`src/g1d_deploy/runtime/drivers/board/fisheye_image_server/`): at CHECK,
  P0 SSHes to the board with the `[board]` login, stops the
  vendor's teleimager (the two cannot run at once) and starts ours from `dir`
  in `conda_env`. Before starting it, P0 hashes the server's `.py`/`.yaml`
  files on both sides, copies any that differ into `dir` (creating it if
  needed), and restarts a publisher that was not started from exactly this
  code (`/tmp/fish_eye_image_server.stamp` on the board). So the board always
  runs the package's copy; a hand edit there is overwritten at the next CHECK.
  `G1D_NO_BOARD_AUTOSTART=1` skips all of it.
- **`teleimager`** is Unitree's service; P0 never touches the board.
- **absent / `none`**: no board cameras.

## `g1d_base.toml`: the robot itself

Rarely edited; every number cites its source in the file.

| section | holds |
|---|---|
| `[rates]` | `body_hz` 500 (P7), `lowstate_hz` 1042 (measured), `hand_hz` 200 (P8), `pico_poll_hz` |
| `[dds]` | domain and the `rt/lowcmd` / `rt/lowstate` topics (the network card is the machine file's) |
| `[joints]` | wire indices: `arm` 15–28, `waist` 12–14, `leg` 0–11 |
| `[gains]` | kp/kd per group, the damp gains, `waist_tau_offset` |
| `[limits]` | `qdot_max` 2π rad/s, the slew every clamp uses |
| `[model]` | `urdf` (relative to the package's `robot/` folder), `[model.wrist_payload]` |
| `[servo_model]` | the arm's identified inertia, viscous and Coulomb friction per joint, for the torque feedforward (`docs/infra_logic/500hz_execution.md`). Identified on the reference G1-D; inertia transfers between units with the same hand, friction only approximately. Without it, only `torque_ff = 0` runs |

## Records carry their robot

Every rollout's `manifest.json` carries the robot file chain verbatim. The
inspection dashboard rebuilds the robot from that text and this package's
model, so a record opens on any machine with g1d_deploy installed — unless its
robot names a custom URDF, which is reported as unsupported there.

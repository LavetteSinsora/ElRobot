"""The resolved settings: the two objects every process consumes.

  `DeployConfig`  the TASK   — resolved from a deploy file (`deploy.toml`,
                              `deploy_file.resolve`): policy entry, checkpoint,
                              prompt, home/exit pose, rollout settings
  `Hardware`      the ROBOT and the BOX — `load_hardware` builds it from a
                              robot file (`robot/profiles/*.toml`,
                              `robot_file.py`) and the machine file
                              (`machine_file.py`)

Plus the camera specs and views both are built from. What is computed from them
is `derive.py`; the startup consistency checks are `validate.py`.

Built once by P0 at resolution and handed to every process (pickled, or as
JSON to P5: `json_transport.py`); nothing mutates it after launch except the
per-rollout settings (`rollout_settings.py`). Each field's comment is its
specification; the user-facing account is
`docs/deploying_a_policy/deploy_toml.md`, `docs/setup/robot_toml.md` and
`docs/setup/machine_toml.md`.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from elrobot_deploy.runtime import cpu_pinning as affinity, ik as ik_solvers
from elrobot_deploy.robot import end_effectors
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.config_loading.errors import ConfigError

try:                                    # Python 3.11+
    import tomllib
except ModuleNotFoundError:             # 3.10 and older — the robot PC is 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise ModuleNotFoundError(
            "elrobot_deploy needs a TOML parser. Python 3.11+ has `tomllib` in "
            "the standard library; on 3.10 install the backport it is derived "
            "from:\n\n    pip install tomli\n\n"
            "This is the only 3.11-ism in the package — everything else uses "
            "`from __future__ import annotations`.") from exc


def next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1)).bit_length()


# --- the task ---------------------------------------------------------------


#: Where a camera's frames come from. `zmq` is the robot board's image server;
#: the other two are cameras plugged into the deploy PC itself — e.g. fisheye
#: wrists on USB and a RealSense for an exocentric view, none of them reachable
#: through the robot's image server.
#:
#: The source decides what `device` means and whether a robot-file `port` is
#: required; see `validate`.
CAMERA_SOURCES = ("zmq", "v4l2", "realsense")

#: Camera sources whose driver is an OPTIONAL extra of elrobot_deploy, not a base
#: dependency: source -> (import name, extra). Imported only when a camera of
#: that source is opened; `validate` refuses a declared camera whose driver is
#: missing, before anything starts.
OPTIONAL_CAMERA_DRIVERS = {"realsense": ("pyrealsense2", "realsense")}


def _importable(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_camera_driver(name: str, source: str) -> str | None:
    """Why camera `name` cannot be opened on this interpreter, or None."""
    if source not in OPTIONAL_CAMERA_DRIVERS:
        return None
    module, extra = OPTIONAL_CAMERA_DRIVERS[source]
    if _importable(module):
        return None
    return (f"camera {name!r} has source {source!r}, but {module} is not "
            f"installed. It is an optional extra: `uv sync --extra runtime --extra {extra}` "
            f"(or `scripts/install.sh --{extra}`), or use a robot profile "
            f"without that camera.")

#: The IK MODES — which joints a solver may move (`ik_solvers.MODES`). A run
#: does not set its mode: it names a solver (`DeployConfig.ik`), and the solver
#: carries its mode (`ik_solvers.mode_of`).
IK_MODES = ik_solvers.MODES

#: The solvers elrobot_deploy ships (`ik_solvers.SOLVERS`), in the dashboard's
#: order. `DeployConfig.ik` is one of these, or a research `module:factory`.
IK_SOLVERS = tuple(ik_solvers.SOLVERS)

#: Which end effector is on the arms — a robot file's `[end_effector] type`.
#: Each type's topics, travel and direction are built in
#: (`end_effectors.END_EFFECTORS`), not written in the robot file.
HANDS = end_effectors.TYPES

#: The Dex-1's travel, rad. The intervention's grip knobs are written in
#: these units, and another hand gets the same fraction of ITS travel.
DEX1_TRAVEL = end_effectors.DEX1_TRAVEL

@dataclass(frozen=True)
class CameraSpec:
    """Native resolution, because decode happens at native resolution and any
    resize belongs to the policy's own preprocessing, where it can match what the
    policy expects.

    `device` is the source's address, and what it means depends on `source`:

    | source | `device` |
    |---|---|
    | `zmq` | unused — the port is `port`, and P2 reads it from `Hardware.camera_ports` |
    | `v4l2` | **a `/dev/v4l/by-path/...` path**, see below |
    | `realsense` | a serial number, or None for "the only device" |

    **`by-path`, not `by-id`, and this is not a style preference.** The two MSK
    R4231 fisheye units report the *same* serial number (`200901010001`, on both
    left and right), so `by-id`
    symlinks collide and `/dev/videoN` numbering is not stable across reboots.
    The only stable, distinguishing address is which physical USB port the
    camera is plugged into. Getting it backwards swaps the two wrist views and
    nothing raises: the policy runs, mirrored.
    """

    height: int
    width: int
    port: int | None = None
    source: str = "zmq"
    device: str | None = None

    #: The camera's frame rate, and the only thing `acquire_anchor` needs from
    #: this spec. Half a period is the threshold at which the frame already
    #: in hand is provably nearer to the trigger instant than any future one, so
    #: this decides when the anchor may stop waiting.
    #:
    #: A wrong value costs latency, never correctness: too low and the wait
    #: behaves as it did before this knob existed, too high and the anchor can
    #: settle for a frame staler than it should. `_verify_rates` prints every
    #: camera's measured rate at CHECK, which is where a disagreement shows.
    #:
    #: 30 because that is what the board's image server publishes and what the
    #: wrists measure (29.99–30.00 Hz across three rollouts). A camera that runs at
    #: another rate says so here rather than being discovered by its latency.
    hz: float = 30.0

    #: True when one frame carries TWO eyes side by side, as the G1-D's head
    #: module does: a single 480x1280 image whose left 640 columns are the left
    #: eye. Declared rather than inferred from the aspect ratio, because
    #: "unusually wide" is a guess and this is a fact the sensor already states
    #: — `runtime/drivers/board/fisheye_image_server/cameras.yaml` carries the
    #: same flag for the same camera, so the board and the deploy config use
    #: one vocabulary.
    #:
    #: **The arena splits it; the record does not.** P2 cuts the decoded frame
    #: into two views, `<name>_left` / `<name>_right`, one ring each
    #: (`camera_views`), so a policy never sees a binocular frame.
    #: The recorder still stores the wire JPEG verbatim — the whole binocular
    #: frame, under the robot file's name — and the inspector splits it for
    #: display. Recording both eyes costs ~56 MB per rollout over recording one,
    #: and buys back the only thing that cannot be recovered later: the eye
    #: nobody asked for yet.
    binocular: bool = False

    # --- from the robot file -------------------------------------------------
    #
    # Everything below is set by `robot_file.load_robot` from `[cameras.<name>]`
    # (`docs/setup/robot_toml.md`, "`[cameras.<name>]`").
    # Defaults are "a plain camera with nothing declared", so a spec built by
    # hand (every test fixture) means what it always meant.

    #: The board to read a `zmq` camera from. None = the machine file's
    #: `robot_ip` (`Hardware.camera_host`), which is every G1-D camera today.
    host: str | None = None

    #: `fisheye180` or `normal` (`robot_file.CAMERA_KINDS`). A `fisheye180` camera
    #: has a lens circle, fitted at every launch (`runtime/drivers/fisheye_circle_fit.py`).
    kind: str = "normal"

    #: `side_by_side` for the binocular head, else None. `binocular` above is
    #: the older spelling of the same fact and is kept equal to it.
    layout: str | None = None

    #: Optional: a known lens circle `(center_x, center_y, radius)` in
    #: full-sensor pixels (robot file `fallback_lens_circle`). The fit falls back to it
    #: (with a warning) when it fails, and refuses a fit that lands too far
    #: from it as "not this lens". Without it a failed fit refuses the launch.
    fallback_lens_circle: tuple[float, float, float] | None = None

    #: The lens square THIS launch uses, `(x0, y0, x1, y1)` in full-sensor
    #: pixels (end exclusive, may run off the sensor), and the circle it was
    #: cut around: the launch-time fit's, or `fallback_lens_circle`'s when the fit
    #: could not be trusted. None until P0 has fitted (`Launcher._fit_lenses`,
    #: before the arena is created, so every process is handed it).
    crop: tuple[int, int, int, int] | None = None
    #: `(center_x, center_y, radius)`, full-sensor pixels.
    lens_circle: tuple[float, float, float] | None = None

    #: Optional calibration, passed through to the policy untouched and never
    #: read by the infra: `fx, fy, cx, cy`, a distortion model and its coeffs;
    #: `parent`, `xyz`, `wxyz`. The URDF has no camera links, so these can only
    #: come from calibration or from the device itself.
    intrinsics: dict | None = None
    extrinsics: dict | None = None

    @property
    def frame_period(self) -> float:
        return 1.0 / float(self.hz)


# --- camera views: what the arena and the policy call a camera ---------------

#: The two names a `layout = "side_by_side"` camera becomes, in column order:
#: `<name>_left` is columns `[0, W/2)`, `<name>_right` is `[W/2, W)`.
#:
#: **Columns 0..W/2 are the LEFT eye**, checked on a recorded G1-D head frame
#: (480x1280, reference robot, 2026-09-17) rather than taken from a comment:
#: every object sits further right in the left half than in the right half —
#: the gripper jaws by ~90 px, the plate on the far table by ~70 px — which is
#: the sign a left camera's disparity has (`x_left - x_right > 0`), and the left
#: half shows more of the scene's left edge (the operator standing beside the
#: table) while the right half shows more of its right (the window). The robot
#: file's own note, "left eye in the left 640 columns", agrees.
SPLIT_HALVES = ("left", "right")

#: The minimum slots per camera ring, and what a camera with no declared
#: history gets: one slot the writer is filling, one holding the newest frame,
#: one holding the frame before it — a triple buffer, the least a lock-free
#: single-writer camera region can be.
CAMERA_MIN_SLOTS = 3

#: Headroom on the frame count a declared history needs: a 30 Hz camera that
#: runs a few percent fast, or delivers a burst after a stall, must not lap the
#: oldest frame a policy asked for. 25% is ~7 frames per second of history.
CAMERA_SLOT_MARGIN = 1.25


def camera_slots(history_s: float, hz: float) -> int:
    """`N = ceil(history_s * hz * 1.25) + 3`. 3 when no history."""
    if history_s <= 0:
        return CAMERA_MIN_SLOTS
    return int(np.ceil(history_s * hz * CAMERA_SLOT_MARGIN)) + CAMERA_MIN_SLOTS


@dataclass(frozen=True)
class CameraView:
    """One camera as the ARENA and the POLICY name it — one ring each.

    A robot-file camera (`CameraSpec`) is a wire stream: one socket or device,
    one P2 process. Almost always it is also one view, under its own name. The
    exception is a `layout = "side_by_side"` camera, whose one frame carries two
    eyes: P2 decodes it once and cuts it into two views, `<name>_left` and
    `<name>_right` (`SPLIT_HALVES` says which is which and how that was
    checked), each with its own ring, so a policy asks for either eye, or both,
    by name. The wire name itself (`head`) is then not a view at all: there is
    no ring called `head`, and a policy naming it is refused with the two names
    it meant.

    **A `fisheye180` camera is one view, cut square around its lens circle**
    (`cropped`): P2 decodes the whole sensor frame and publishes only the
    square, so every reader — the policy's `obs.frames`, the reader, the
    anchor — sees the lens disc and never the sensor around it. The square is
    `lens_square(spec)`: the launch-time fit's (`CameraSpec.crop`), which is
    decided before the arena is sized. Off-sensor parts of it are black.

    `x0` / `width` are the columns of the wire frame this view is. `carries_jpeg`
    marks the one view per wire camera whose slots also hold the compressed
    source — the WHOLE wire frame, stored once, because the two halves of one
    frame share one JPEG and the recorder and the dashboard want it verbatim
    rather than re-encoded. For an unsplit camera that is the camera itself;
    for a split one it is the left half, and the right half's slots reserve no
    JPEG bytes at all.
    """

    name: str
    source: str            # the robot-file camera (wire stream) this is cut from
    height: int
    width: int
    x0: int                # first column in the wire frame
    hz: float
    carries_jpeg: bool
    #: The wire frame's shape, which is what the JPEG decodes to.
    source_height: int
    source_width: int
    #: `left` / `right` for half of a split camera, None for a whole one.
    half: str | None = None
    #: First row in the wire frame: the lens square's top for a `cropped` view
    #: (it may be negative: the square may run off the sensor), else 0.
    y0: int = 0
    #: A `fisheye180` view: the square around the lens circle.
    cropped: bool = False
    #: Seconds of history the policy declared for this view (`Policy.history`,
    #: via `DeployConfig.camera_history_s`); 0 = none. Sizes the ring.
    history_s: float = 0.0

    @property
    def frame_period(self) -> float:
        return 1.0 / float(self.hz)

    @property
    def slots(self) -> int:
        return camera_slots(self.history_s, self.hz)

    @property
    def jpeg_capacity(self) -> int:
        return (schema.jpeg_capacity(self.source_height, self.source_width)
                if self.carries_jpeg else 0)


def split_names(name: str) -> tuple[str, str]:
    """`head` -> `("head_left", "head_right")`."""
    return tuple(f"{name}_{half}" for half in SPLIT_HALVES)


def camera_views(cameras: dict, history: dict | None = None) -> dict[str, CameraView]:
    """Robot-file cameras -> the views the arena holds, sorted by name.

    Pure, and the ONE place the split happens, so the layout, P2, the anchor,
    the reader, the recorder and the dashboard cannot disagree about which
    view is which columns. `history` maps view names to seconds (entries for
    names that are not views are ignored here; `validate` refuses them).
    """
    history = history or {}
    views: dict[str, CameraView] = {}
    for name in sorted(cameras):
        spec = cameras[name]
        if spec.layout == "side_by_side":
            if spec.width % 2:
                raise ConfigError(
                    f"camera {name!r} is side_by_side but {spec.width} px wide; "
                    f"two eyes of equal width need an even width")
            w = spec.width // 2
            for i, (view, half) in enumerate(zip(split_names(name), SPLIT_HALVES)):
                if view in cameras:
                    raise ConfigError(
                        f"camera {name!r} is side_by_side, so it is split into "
                        f"{split_names(name)} — and the robot file also declares "
                        f"a camera {view!r}. Rename one of them.")
                views[view] = CameraView(
                    name=view, source=name, height=spec.height, width=w,
                    x0=i * w, hz=float(spec.hz), carries_jpeg=(i == 0),
                    source_height=spec.height, source_width=spec.width,
                    half=half, history_s=float(history.get(view, 0.0)))
        elif spec.kind == "fisheye180":
            (x0, y0, x1, _), _ = lens_square(spec)
            views[name] = CameraView(
                name=name, source=name, height=x1 - x0, width=x1 - x0,
                x0=x0, y0=y0, cropped=True, hz=float(spec.hz), carries_jpeg=True,
                source_height=spec.height, source_width=spec.width,
                history_s=float(history.get(name, 0.0)))
        else:
            views[name] = CameraView(
                name=name, source=name, height=spec.height, width=spec.width,
                x0=0, hz=float(spec.hz), carries_jpeg=True,
                source_height=spec.height, source_width=spec.width,
                history_s=float(history.get(name, 0.0)))
    return dict(sorted(views.items()))


def lens_square(spec) -> tuple[tuple[int, int, int, int], str]:
    """A `fisheye180` camera's square and where it came from: `"fitted"` (this
    launch's fit, `spec.crop`), `"fallback_lens_circle"` (no fit ran, e.g.
    check-policy: the robot file's circle), or `"placeholder"` (neither: the
    sensor-height square in the middle, right in shape only)."""
    from elrobot_deploy.runtime.drivers.fisheye_circle_fit import crop_for, placeholder_crop

    if spec.crop is not None:
        return tuple(int(c) for c in spec.crop), "fitted"
    if spec.fallback_lens_circle is not None:
        return crop_for(*spec.fallback_lens_circle), "fallback_lens_circle"
    return placeholder_crop(spec.height, spec.width), "placeholder"


def views_of(views: dict[str, CameraView], source: str) -> list[CameraView]:
    """The views cut from robot-file camera `source`, in column order."""
    return sorted((v for v in views.values() if v.source == source),
                  key=lambda v: v.x0)


def not_a_view_hint(name: str, cameras: dict) -> str:
    """The sentence that turns "no camera `head`" into the fix."""
    spec = cameras.get(name)
    if spec is not None and spec.layout == "side_by_side":
        return (f" {name!r} is a side_by_side camera and is split into "
                f"{list(split_names(name))}; name one (or both) of those.")
    return ""


@dataclass(frozen=True)
class ServoShaping:
    """How P7 turns the 30 Hz target line into 500 Hz commands (`runtime/execution/servo_filter.py`).

    The TASK's choice; the torque model it uses is the ROBOT's
    (`Hardware.servo_model`, robot file `[servo_model]`).

    **Default: smoothed, with torque feedforward.** The arm renders its
    command's roughness faithfully, and a straight-line command kicks it at
    every 30 Hz target. Measured on the reference robot (replays of two
    recorded rollouts, 2026-09) the 4 Hz filter halved 4-40 Hz judder
    (122 -> 67 and 138 -> 61 mrad/s) with no loss of hand accuracy, and 50 %
    feedforward halved lag (12 -> 6 ms). `linear()` is the unshaped servo.
    Design: `docs/infra_logic/500hz_execution.md`.
    """

    #: Cutoff of the critically damped command filter on all 17 body joints, Hz.
    #: 0 = no filter: the straight-line command.
    filter_hz: float = 4.0

    #: Read the target line ahead by the filter's own lag (2/w: 80 ms at 4 Hz)
    #: so smoothing costs no delay. Capped at the last written target.
    filter_lead: bool = True

    #: Scale on the published velocity feedforward `dq`.
    velocity_ff: float = 1.0

    #: Scale on the model torque feedforward (inertia + friction), added to
    #: gravity comp on the arm joints. 0 = off.
    torque_ff: float = 0.5

    #: Scale on the ARM joints' kp from the robot file; kd by its square root,
    #: which keeps each joint's damping ratio.
    arm_gain_scale: float = 1.0

    @classmethod
    def linear(cls) -> "ServoShaping":
        """The unshaped servo: straight segments, slope as dq, gravity only."""
        return cls(filter_hz=0.0, torque_ff=0.0)

    @property
    def lead_s(self) -> float:
        if self.filter_hz <= 0 or not self.filter_lead:
            return 0.0
        return 2.0 / (2.0 * np.pi * float(self.filter_hz))

    def describe(self) -> str:
        parts = [f"filter {self.filter_hz:g} Hz, lead {self.lead_s * 1e3:.0f} ms"
                 if self.filter_hz > 0 else "no filter (straight-line command)"]
        parts.append(f"torque ff x{self.torque_ff:g}" if self.torque_ff else "no torque ff")
        if self.velocity_ff != 1.0:
            parts.append(f"dq ff x{self.velocity_ff:g}")
        if self.arm_gain_scale != 1.0:
            parts.append(f"arm kp x{self.arm_gain_scale:g}")
        return ", ".join(parts)


@dataclass(frozen=True)
class ServoModel:
    """Per arm joint dynamics for the torque feedforward. Robot file `[servo_model]`.

    (14,) arrays in body order: inertia kg m^2, viscous N m s/rad, Coulomb N m.
    """

    inertia: tuple[float, ...]
    viscous: tuple[float, ...]
    coulomb: tuple[float, ...]
    eps_v: float = 0.02         # rad/s: Coulomb term is Fc tanh(v / eps_v)
    tau_max: float = 2.0        # N m: bound on the feedforward per joint


@dataclass(frozen=True)
class DeployConfig:
    """Everything the task owns — resolved from a deploy file by
    `deploy_file.resolve`, layered: infra defaults < the policy class's
    declared rates (`trained_*`) < deploy.toml < command line."""

    prompt: str
    #: The resolved checkpoint path (`checkpoints.resolve`): a step directory,
    #: or "" for a policy that loads nothing.
    checkpoint: str

    sync: bool
    replan_period_ticks: int          # P
    action_horizon: int               # H
    control_hz: float                 # the ACTION rate

    cameras: dict[str, CameraSpec]
    primary_camera: str

    # The two postures the run starts and ends at. Both are full: arm, waist and
    # end effector, so "where the robot goes when the run ends" is one place in
    # one file rather than three defaults and an implicit hold.
    home_q: np.ndarray                # (14,) arm
    home_waist: np.ndarray            # (3,)
    home_hand: np.ndarray             # (2, 6)
    exit_q: np.ndarray                # (14,)
    exit_waist: np.ndarray            # (3,)
    exit_hand: np.ndarray             # (2, 6)

    #: Which of `cameras` the POLICY observes. Empty means "all of them".
    #:
    #: **`cameras` is what EXISTS; this is what is USED, and the gap between
    #: them is the point.** A camera in `cameras` gets an arena region, a reader
    #: process, a place in every rollout record and a pane on the live
    #: dashboard. A camera named here additionally gates `acquire_anchor` and
    #: reaches the policy. So a camera declared but not observed is recorded and
    #: nothing else — e.g. a head camera the board already publishes and the
    #: policy does not use.
    #:
    #: The distinction is not bookkeeping. `acquire_anchor` waits for EVERY
    #: camera it is handed to settle, and a camera that has produced no frame at
    #: all is deliberately not covered by that wait's cap — so a record-only
    #: camera left in the anchor set could hang a run that does not use it, and
    #: would inflate the `image latency` figure P5 prints for the cameras that
    #: do matter. The recorder needs none of that: P9 sweeps the arena directly
    #: at 500 Hz and never consults an anchor.
    observed_cameras: tuple[str, ...] = ()

    #: Seconds of history per camera VIEW: the policy class's
    #: `history` minus `"state"`, copied here by `deploy_file.resolve` because the
    #: class is known at resolve time and every process must size the arena
    #: identically without importing it. Empty = no camera history = every ring
    #: at `CAMERA_MIN_SLOTS`.
    #:
    #: Fixed per launch: it is part of the arena layout, and the arena is never
    #: resized.
    camera_history_s: dict[str, float] = field(default_factory=dict)

    # --- human intervention (`runtime/teleop/intervention.py`) -------------
    #
    # Eight values, every one of them read by exactly one process, and none of
    # them shared state. A run without `--intervene` never looks at any of them.

    #: How far ahead of the operator's own instant a target is written, seconds.
    #: **The one real knob**, and it buys exactly two things that are the same
    #: number: the jitter margin (how late a PICO sample may arrive and still be
    #: executed at all) and the perceived lag (the arm trails the hand by
    #: `lead + transport + solve`). Responsiveness and robustness trade off one
    #: for one, so this wants to be SET from a measurement rather than chosen.
    #:
    #: Zero is not a fast intervention but a dead one. Anchor the device-to-local
    #: clock map on the first sample with no lead and the margin on every later
    #: target is `L_1 - L_n` — the first sample's total latency minus this one's,
    #: which is a coin flip — so about half of all targets are published with a
    #: deadline already in the past and P7, which selects only on
    #: `deadline > now`, discards them. See `intervention.Pace`.
    #:
    #: 35 ms: sample age on the deploy box has a p1..p99 spread of ~30 ms with
    #: P11 on an efficiency core, and the IK is p99 ~2 ms.
    intervene_lead_s: float = 0.035

    #: How long the PICO stream may go silent before a running intervention is
    #: handed back to the policy, seconds. Until then the arm simply HOLDS: P11
    #: writes no rows, P6 has nothing new to solve, P7 stalls and republishes the
    #: last command frozen — the correct response, reached with no code.
    #:
    #: The handback is deliberately loud, because "the robot started moving on
    #: its own" must never be something an operator has to infer.
    intervene_stale_budget_s: float = 0.5

    #: Metres of arm per metre of hand. 1.0 is the honest default; below 1 trades
    #: reach for precision, which is worth having for a fine correction. Applied
    #: to the TRANSLATION only — scaling the rotation as well would make a full
    #: turn of the wrist a partial turn of the gripper, and the two would then
    #: disagree about which way is up (`intervention.Anchor.follow`).
    intervene_scale: float = 1.0

    #: (The ROBOT's side of the mapping — flange -> the virtual TCP, where a
    #: rotation pivots — is not a deploy setting: it is the end effector's
    #: `tcp` in the robot file, `Hardware.tcp`.)

    #: Controller -> the OPERATOR's virtual TCP, **per hand**, `(2, 7)` in
    #: `[x y z qw qx qy qz]`, left then right.
    #:
    #: **Per hand, and its ROTATION is the only calibration the mapping has.**
    #: Both halves of that sentence are measured facts rather than caution. The
    #: rotation is 162.2 degrees from identity, and the two hands differ from
    #: each other by 12.5 degrees — so a single shared value would be wrong for
    #: at least one hand by that much, and identity would be wrong for both by
    #: almost a half-turn. This is not a refinement; it is the difference between
    #: an arm that follows the operator and one that does not.
    #:
    #: The default was calibrated for PICO controllers held as a handheld
    #: gripper's handles (one operator rig, 2026-08); a different controller or
    #: grip needs its own (`[intervene] hand_tcp` in the deploy file).
    #:
    #: `intervention.Anchor` assumes the operator's tool axes and the robot's
    #: TCP axes are the same convention, so no fixed 3x3 sits between them, and
    #: `Anchor.follow` composes increments in the operator's TCP frame at the
    #: moment of takeover.
    intervene_hand_tcp: tuple[tuple[float, ...], ...] = (
        # left
        (0.00440398426517279, -0.1824239566616859, -0.022715495135066984,
         -0.15442163755283994, 0.7167830094782406, -0.6497363212120365,
         -0.200546224289228),
        # right
        (-0.007934722248033496, -0.18267240227738427, -0.018012588844547275,
         -0.19002149384707825, 0.6546588150711479, -0.715762704485474,
         -0.15164899799670792),
    )

    #: How fast a held thumbstick drives the jaw, rad/s. Travel is 5.4 rad, so
    #: 3.0 is a 1.8 s full sweep. A feel knob, not a safety one — the bound on
    #: force is `intervene_squeeze_margin_rad`.
    intervene_gripper_rate: float = 3.0

    #: **The squeeze bound**, radians below the MEASURED jaw that the command may
    #: sit. This is the one intervention parameter that is about force.
    #:
    #: elrobot_deploy publishes jaw position straight to the wire at
    #: `runtime/drivers/gripper.py::GRIPPER_KP` = 10.0 with no leash of its
    #: own. A held stick is an integrator, so with no bound an operator can
    #: wind the command to 0 against an object stopped at 1.4 rad and ask for
    #: 14 N·m.
    #:
    #: 0.6 rad is 6 N·m — a firm grasp, far below the runaway. Set it to `inf`
    #: to remove the bound entirely.
    intervene_squeeze_margin_rad: float = 0.6

    #: Which way the thumbstick closes the jaw. `True`: pushing the stick up
    #: closes.
    intervene_up_closes: bool = True

    state_history_s: float = 1.0
    home_min_s: float = 3.0
    home_velocity_margin: float = 0.5  # ramp inside qdot_max by this factor

    #: How far the MEASURED posture may sit from `home_q` / `home_waist` once
    #: homing has finished. Two numbers, not one, because the two errors
    #: are different in kind.
    #:
    #: An arm error is honest. P6 seeds from the written target and never from
    #: the measurement, so it does not accumulate, and the observation
    #: carries the measured joints, so the policy sees where the arm really is.
    #: What is left is steady-state droop under `kp`, and it is real: measured
    #: 0.041, 0.086 and 0.091 rad on three rollouts of the reference robot
    #: (2026-08-20), always on the more heavily loaded arm; the other stayed
    #: inside 0.014. The default clears the worst of those with room and
    #: still catches an arm that never arrived.
    #:
    #: A waist error is NOT honest, and that is why its tolerance is tighter.
    #: The waist is the frame every downstream pose is expressed against, and two
    #: things assume it is exactly what was commanded: P6's IK pins it at the
    #: commanded angles and `PinocchioComp` locks it at zero. A trunk
    #: that is not where those two believe puts the observed flange and the
    #: solved flange in different geometries — `kinematics.place()`'s docstring calls
    #: this out as a silent bias in metres that nothing reports.
    #:
    #: 0.05 rather than 0.01, because 0.01 refused every run the reference
    #: robot can produce. The G1-D's second waist axis is a BOW, not a
    #: roll (see `kinematics.WAIST_JOINT_NAMES`), and it carries the whole upper body's
    #: forward moment: measured 0.039 rad on every rollout, moving less than
    #: 0.005 rad across an entire 80-second run while commanded to zero the whole
    #: time. At kp=300 that is ~11.7 Nm of gravity the joint does not have the
    #: authority to close. It is structural, not a fault, and a tolerance that
    #: refuses it refuses the machine.
    #:
    #: What the number still catches is a waist that moved, which is the failure
    #: worth catching: 0.05 is under a third of the arm's tolerance and a small
    #: fraction of the bow's real travel (−0.04 to +2.36 rad, 135° of it), so a
    #: trunk actually displaced — knocked, mis-homed, or commanded by a
    #: policy that moves the waist — still refuses. Raise it further only
    #: with a measurement in hand.
    #:
    #: That travel is the G1-D's own (its URDF's), not the G1 humanoid's ±0.52
    #: rad waist, which is a quarter of what this machine has.
    home_tol_arm: float = 0.15
    home_tol_waist: float = 0.05

    #: Integration steps the policy's sampler takes to produce one chunk.
    #:
    #: elrobot_deploy never integrates anything itself -- `Policy.infer` is the
    #: policy's own, so what this number means (Euler steps on a velocity
    #: field, a diffusion schedule, nothing at all for a direct regressor or
    #: a scripted controller) is the policy's business; it reaches the policy
    #: through `Policy.settings` / `on_settings`. It lives here for the two
    #: reasons a task-owned number does: it is the main latency/quality dial
    #: an operator wants to turn per rollout (`--sample-steps`), and being in
    #: the config it lands in the rollout record, so two runs can never
    #: silently differ by it. A policy with no such notion simply ignores it.
    sample_steps: int = 10

    #: How async's overlapping chunks become one action per tick
    #: (`chunk_blend.get`): "temporal_ensemble" (ACT's), "newest", or a
    #: module / file path. Inert in sync (one chunk per tick).
    blend: str = "temporal_ensemble"
    #: `temporal_ensemble`'s `m`, per tick of a prediction's age (ACT: 0.01).
    temporal_ensemble_m: float = 0.01

    #: `[control.servo] smoother` / `feedforward`: a deploy file's own
    #: `"package.module:factory"` replacing the shipped critically damped
    #: filter / model torque feedforward (`servo_filter.build`). Per launch;
    #: `servo.filter_hz` / `servo.torque_ff` still switch them on and off.
    servo_smoother: str = ""
    servo_feedforward: str = ""

    ik_posture_cost: float = 0.02
    ik_collision_min_dist: float = 0.005
    ik_early_stop: float = 1e-4
    ik_max_iters: int = 20

    # --- the IK solver (elrobot_deploy.runtime.ik) ---------------------------------

    #: WHICH IK SOLVER P6 RUNS, by name — and with it which joints the IK may
    #: move, since every solver carries its mode (`ik_solvers.SOLVERS`):
    #:
    #:   ``arm_planner``   arm mode, the streamed branch planner. **The
    #:                     default.** The trunk is read, never commanded.
    #:   ``arm_qp``        arm mode, `arm_qp.DualArmSolver` — one QP tick per
    #:                     loop iteration from the last answer. The only
    #:                     solver `--intervene` runs.
    #:   ``waist_planner`` waist mode (+ yaw and bow), the arm planner on a
    #:                     slewed, lag-aware trunk plan.
    #:   ``waist_qp``      waist mode, `waist_qp.WholeBodySolver`'s waist tier.
    #:   ``module:factory`` a research solver whose factory carries `.mode`.
    #:
    #: ``"arm"`` and ``"waist"`` are accepted as shorthands for the mode's
    #: default (`ik_solvers.DEFAULT`) and normalized to it in `__post_init__`,
    #: so a manifest always names the solver that actually ran. **Changing the
    #: default changes what a deploy file that says `ik = "arm"` (or nothing)
    #: runs.**
    #:
    #: Waist mode is SYNC ONLY. Sync gives exactly one contributor per tick; in
    #: async two chunks overlap by design and a trunk plan would be made
    #: against a blend that changes underneath it. `validate` refuses the
    #: combination rather than silently falling back. `waist_qp.py`'s wider
    #: tiers (`waist_column`, `full`) are not solvers: nothing publishes
    #: `rt/cmd_hispeed`, so nothing here commands the column or the base.
    #:
    #: Switchable between rollouts among the four shipped names; a
    #: `module:factory` is fixed per launch. `docs/infra_logic/ik_solvers.md`.
    ik: str = "arm_planner"

    #: How many ticks ahead of the servo P6 may solve. 0 = no bound.
    #:
    #: P6 solves a tick in ~4 ms against P7's 33 ms period, so unbounded it
    #: solves a whole chunk within ~110 ms of it landing and then idles for the
    #: rest. Every tick is then solved against one observation of the trunk,
    #: taken before any of them ran — which is fine while the trunk is frozen
    #: and is not fine once a waist-mode solver moves it, because the arm is solved in a
    #: frame that is up to a chunk out of date (`p6_target.Target.solve_frame`).
    #:
    #: Bounding the lead keeps that measurement fresh: the solve for tick k
    #: happens ~`solve_lead_ticks` ticks before k executes, so the frame is a
    #: short extrapolation rather than a chunk-long prediction.
    #:
    #: **The floor is set by the solve time, not by accuracy.** P6 must publish
    #: a tick before its deadline or it becomes a hole; 5 ticks is ~165 ms of
    #: slack against a ~4 ms solve. Going much below 2 leaves a single solve of
    #: margin. There is no ceiling — a large value simply reverts to racing.
    solve_lead_ticks: int = 5

    #: How many ticks PAST the replan period P5 publishes for the trunk planner
    #: to look at. 0 — the default —
    #: publishes exactly the `P` actions that will run, and nothing here is
    #: reachable.
    #:
    #: **These rows are never executed.** They go in `n_plan`, not `n_actions`,
    #: so coverage — which is what makes a tick executable, what P5's window
    #: scan reads, and what partitions the tick line for the recorder — is
    #: unchanged. The one reader is `p6_target._plan_trunk`, which appends them
    #: to the trajectory it optimizes and then solves the same `P` ticks it
    #: always did.
    #:
    #: WHY. `StreamPlannerSolver` picks a branch per chunk and cannot leave it:
    #: a chunk whose targets are all reachable from the shoulder-roll stop is
    #: solved there at ~0 mm error, and the next chunk, which needed the other
    #: side, then cannot be reached at all. Measured on one recorded rollout
    #: (reference robot, 2026-09-16): the arm
    #: pins at Lsr = +2.25 rad and mean TCP error goes to 18.8 mm. Showing the
    #: planner a few ticks past the boundary is enough to make the trap visible
    #: while the branch is still free, and it takes the other one.
    #:
    #: WHY BOUNDED, and why this is not simply `H`. Rows far past the period are
    #: the policy's least certain: the median prediction error is flat across
    #: the tail but the p95 grows (95 mm over rows 0-8, 143 mm over rows 16-25),
    #: and one bad far row can re-trap the very branch this is meant to free.
    #: Over three sampler draws of the policy flown that day, spans of 8-16 ticks
    #: never trapped; the full 25 trapped one draw in three. **10** is the pick:
    #: tied-best mean (7.9 mm vs 18.8 at 0) and mid-plateau, so neither edge of
    #: the safe band is load bearing. Costs ~0.9 ms of arm-tick p95.
    #:
    #: Sync only, and `P + this <= H`, both refused at startup.
    plan_lookahead_ticks: int = 0


    #: Environment variables P5 sets **before it imports the adapter** (the
    #: deploy file's `[policy.env]`). The task's, not the machine's, because
    #: what belongs on the accelerator is a property of the policy: e.g. a JAX
    #: policy that shares the card wants XLA preallocation off, one that puts
    #: nothing else on the GPU pays for that with a `cudaMalloc` per
    #: intermediate. Setting it here rather than in the machine file keeps two
    #: runs comparable: the allocator regime is recorded with the config that
    #: produced the rollout.
    #:
    #: Applied by `p5_policy.apply_policy_env` with `os.environ`, before the
    #: adapter's module is imported, so it suits flags a framework reads at
    #: import or backend-init time. Nothing else in the
    #: system reads it, and an empty dict means "whatever the shell exported".
    policy_env: dict[str, str] = field(default_factory=dict)

    #: How P7 shapes its 500 Hz command (`ServoShaping`).
    servo: ServoShaping = field(default_factory=ServoShaping)

    #: Free-form labels copied into every rollout record and read by nothing in
    #: the control path — e.g. which A/B arm a run was, or which recording a
    #: replay replays. Strings only, so a record never needs this code to read.
    labels: dict[str, str] = field(default_factory=dict)

    #: The robot file this run was launched against — a shipped profile name or
    #: a path (`robot_file.load_robot`). Informational here: the robot's facts are
    #: already in `Hardware` and `cameras`; the manifest copies the file itself.
    robot: str = ""

    #: The policy class, as the deploy file's `[policy] entry` names it —
    #: `"pkg.module:Class"` or `"/abs/path/file.py:Class"` (relative paths are
    #: made absolute at resolution, so every spawned process agrees). Loaded
    #: with `policy_contract.loading.load_entry`. Empty = no policy named (the EchoPolicy dry run
    #: and every test fixture).
    policy_entry: str = ""

    #: The interpreter P5 runs under — the deploy file's `[policy] python`,
    #: absolute — so the policy's own stack (torch, jax, a numpy 2) lives in
    #: its own venv, which needs only elrobot_deploy's minimal core
    #: (`docs/deploying_a_policy/policy_contract.md`, "The adapter's own Python"). Empty = the
    #: runtime's own interpreter.
    policy_python: str = ""

    #: The policy class's `action_mode` ("eef" | "joint"), read at resolution
    #: (`policy_contract/describe.py`) so processes that never see the class — the banner,
    #: P10's settings panel — can say that in joint mode the IK is not used.
    #: Empty when no deploy file resolved it (tests). P5 still goes by the class.
    policy_action_mode: str = ""

    @property
    def observed(self) -> tuple[str, ...]:
        """`observed_cameras`, resolved — every camera when it is empty.

        Sorted so the anchor's camera set has one order everywhere it is
        printed or compared.
        """
        return self.observed_cameras or tuple(sorted(self.views))

    @property
    def views(self) -> dict[str, CameraView]:
        """The cameras as the arena and the policy name them (`camera_views`):
        `cameras` with every side_by_side camera split into its two eyes, each
        carrying its declared history. `cameras` stays the robot file's — one
        entry per wire stream, one P2 process each; every policy-facing name
        (`primary_camera`, `observed_cameras`, `uses_cameras`, `history`) is a
        view name."""
        return camera_views(self.cameras, self.camera_history_s)

    @property
    def dt(self) -> float:
        return 1.0 / float(self.control_hz)

    @property
    def P(self) -> int:
        return int(self.replan_period_ticks)

    @property
    def H(self) -> int:
        return int(self.action_horizon)

    @property
    def wbik(self) -> bool:
        """Does this run plan the trunk with the arms (a waist-mode solver;
        `docs/infra_logic/ik_solvers.md`)?

        The ONE place the question is answered, so P6's solver choice and its
        per-chunk planning cannot drift apart. `ik` alone is not enough:
        planning a chunk's trunk assumes its ticks are all known and belong to
        one epoch, which only sync guarantees (one chunk of exactly `P` actions
        per epoch). In async two chunks
        overlap by design and the plan would be made against a blend that
        changes underneath it, so async is arm-only whatever `ik` says — and
        `validate` refuses that combination rather than letting this quietly
        answer False. A waist-mode solver in sync is the only True.
        """
        return self.ik_mode == "waist" and bool(self.sync)

    @property
    def ik_mode(self) -> str:
        """"arm" or "waist": the mode of the solver `ik` names
        (`ik_solvers.mode_of`; imports a `module:factory` to read it)."""
        return ik_solvers.mode_of(self.ik)

    def __post_init__(self):
        # "arm" / "waist" are shorthands for the mode's default solver. Stored
        # normalized, so every reader — P6's solver choice, the settings record,
        # the manifest — sees the one name that ran.
        object.__setattr__(self, "ik", ik_solvers.normalize(str(self.ik)))


# --- the robot --------------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    """Everything the robot and the box own, shared by every task.

    Built by `load_hardware` from two files: the robot file (rates, DDS topics,
    joints, gains, limits, model, end effector, camera ports) and the machine
    file (the NIC, the robot's IP, the board login, the CPU table)."""

    body_hz: float
    hand_hz: float
    lowstate_hz: float
    #: How often P11 polls the XRoboToolkit SDK, Hz. Deliberately
    #: about twice the ~90 Hz the SDK republishes at, because a poll that lands
    #: between two publishes is free and a poll that lands late is not: the age
    #: of a sample when it is used is what `intervene_lead_s` has to cover, and
    #: therefore what the operator feels as lag. Sized here rather than in
    #: `DeployConfig` because it is a property of the headset and the machine,
    #: which no task gets an opinion about.
    pico_poll_hz: float

    cpu: dict[str, int]
    cpu_reserved: list[int]
    rt_priority: int | None

    dds_iface: str
    dds_domain: int
    topic_lowcmd: str
    topic_lowstate: str
    #: One topic per hand, in `[left, right]` order — the Dex-1 gives each
    #: gripper its own topic carrying its own single motor.
    topic_hand_cmd: tuple[str, ...]
    topic_hand_state: tuple[str, ...]

    camera_host: str
    camera_ports: dict[str, int]

    joints_arm: list[int]
    joints_waist: list[int]
    joints_leg: list[int]

    kp: np.ndarray                    # (29,)
    kd: np.ndarray                    # (29,)
    qdot_max: np.ndarray              # (29,)

    #: The end effector's travel, in its own units. The arm has three things
    #: bounding it — the QP's `ConfigurationLimit`, P6's velocity clamp and P7's
    #: slew clamp — and the hand had none: `validate_chunk` bounds the waist
    #: against the URDF and nothing bounded the gripper at all — a policy has
    #: been seen commanding 5.4304 rad against a top of travel at 5.4.
    hand_min: float
    hand_max: float

    damp_kp: float
    damp_kd: float

    #: The robot's model — `robot/assets/unitree_g1d/g1_d.urdf`, and the only one.
    #: `elrobot_deploy.robot.g1d_model` builds both the MuJoCo model P1 and P6 need and the
    #: pinocchio inertia model P7 needs out of this single file.
    urdf: str

    #: Hardware bolted to each wrist that no URDF has ever modelled — the wrist
    #: camera and its mount. Mass in kg and CoM along the wrist-yaw frame's own
    #: +x, in metres; applied to BOTH wrists by
    #: `runtime/drivers/arm_command.py::_real_end_effector`.
    #:
    #: 0.0 means "nobody has weighed it", not "there is nothing there". The
    #: rollouts say there IS something: holding home, the arm's measured sag
    #: implies a distal load heavier than the model even after the Dex-1
    #: correction, and this is the only place that can go. It stays 0 until it
    #: is measured, because a fitted value would be indistinguishable from
    #: joint friction, which produces a sag of its own.
    wrist_payload_mass: float = 0.0
    wrist_payload_com_x: float = 0.0

    #: A constant added to each waist joint's feedforward torque, N.m, in
    #: `kinematics.WAIST_JOINT_NAMES` slot order (yaw, bow, absent).
    #:
    #: MEASURED, not fitted (on the reference robot; the robot profile holds
    #: the value). The bow settles short of its command by the same
    #: ~25 mrad whichever direction it travelled from, which at `kp = 300` is
    #: 7.6 N.m the feedforward does not supply and the position error has been
    #: making up. It is a CONSTANT, and that is what identifies it: over the
    #: bow's range the gravity model's own torque goes from -5.7 to -25.9 N.m,
    #: so a modelling error would have grown with lean and this does not. Nor
    #: is it stiction — stepping either side of a bias came out short one way
    #: and long the other, both errors in the same direction, which friction
    #: cannot do.
    #:
    #: Confirmed by adding it: with +7.0 N.m the mean settled error fell from
    #: 25.2 to 9.3 mrad, and a +0.05 rad step landed on +0.0525 instead of
    #: +0.0310.
    #:
    #: What it is NOT: a tuning knob. It is a fact about one robot's trunk —
    #: a spring, a harness, or a driver offset — sitting beside the mechanical
    #: stop as another thing the URDF does not describe. A second robot needs
    #: its own measurement: command small bow steps from both directions and
    #: read the settled error (N.m = kp x error).
    #:
    #: The residual after it is direction-dependent (~7 mrad) and no constant
    #: can remove that; it is the friction the direction test ruled out as the
    #: MAIN cause but did not rule out entirely.
    waist_tau_offset: tuple[float, ...] = (0.0, 0.0, 0.0)

    #: The arm's dynamics for the servo's torque feedforward (`ServoModel`),
    #: or None when this robot has no `[servo_model]`; `check_servo` then
    #: refuses `servo.torque_ff > 0`.
    servo_model: ServoModel | None = None

    #: Where `cpu`/`cpu_reserved` came from: "file" if the machine file stated
    #: a `[cpu.pin]` table, "derived" if this machine's own topology was read
    #: and `cpu_pinning.derive`'s rules applied to it. Recorded in every rollout manifest,
    #: because a table in a record cannot otherwise say whether a human chose it.
    cpu_source: str = "file"
    #: Anything the derivation had to say about this machine — e.g. that it has
    #: no efficiency cores. Empty when the table was stated.
    cpu_notes: tuple[str, ...] = ()
    #: The robot file's `[end_effector] type` (`HANDS`). Its built-in topics
    #: and travel (`end_effectors.END_EFFECTORS`) are already in the fields
    #: above (`for_hand`); this says which way that travel runs.
    hand: str = "dex1"

    #: Flange -> the end effector's **virtual TCP**, `(7,)` float64
    #: `[x y z qw qx qy qz]` in the flange (wrist-yaw link) frame, the same for
    #: both arms. The robot file's `[end_effector] tcp` (optional); None when
    #: the robot file states none.
    #:
    #: A property of the HAND, not of a policy and not of a deploy: the
    #: Dex-1's is the midpoint of the two jaw tips at their furthest extent,
    #: `robot/g1d_model.py::FLANGE_TO_TCP_X` = 0.184759 m along +X with identity
    #: rotation (0.041500 wrist_yaw -> dex1_base_link + 0.143259 base -> tip
    #: midpoint, measured off a MuJoCo model of the gripper at three arm poses;
    #: both arms agree). Symmetric jaws, so it does not move as they open.
    #:
    #: Who reads it: the intervention, where it is **where a rotation pivots**
    #: (`intervention.Anchor`) — measured: with the flange instead, an operator who
    #: turns their wrist 90 degrees across the tool without moving their hand
    #: sweeps the fingertips 33.20 cm rather than 7.07 cm, the extra 26.13 cm
    #: being exactly the chord of a quarter turn on an 18.5 cm arm; and the
    #: dashboards, which draw it. The policy contract stays at the FLANGE:
    #: a policy's own TCP is its own definition, and this is only the
    #: robot's. It reaches the record as the robot file's text in the manifest.
    tcp: np.ndarray | None = None

    # --- the board (machine file [board], robot file [image_server]) -------

    #: Who serves the `zmq` cameras on the board — the robot file's
    #: `[image_server] type` (`robot_file.IMAGE_SERVERS`). Only `fisheye` makes P0
    #: SSH in and start it, from these three (all None otherwise):
    image_server: str = "none"
    #: `[image_server] dir`: the server's directory on the board.
    image_server_dir: str | None = None
    #: `[image_server] conda`: the board's conda installation.
    image_server_conda: str | None = None
    #: `[image_server] conda_env`: the env the server runs in.
    image_server_conda_env: str | None = None
    #: The login P0 uses for that SSH (`image_server_launch.ensure_running`): the robot
    #: file's `[board]` (or `ELROBOT_BOARD_SSH_USER` / `ELROBOT_BOARD_SSH_PASSWORD`).
    board_ssh_user: str = "unitree"
    board_ssh_password: str = field(default="123", repr=False)
    #: Where P10's dashboard listens (machine file `[dashboard] port`,
    #: `ELROBOT_DASHBOARD_PORT`). 0 = any free port.
    dashboard_port: int = 8080

    @property
    def hand_close_is_max(self) -> bool:
        """True when a LARGER hand value is more closed: the BrainCo (0 =
        open). The Dex-1 runs the other way (0 = closed)."""
        return end_effectors.get(self.hand).close_is_max

    @property
    def hand_open(self) -> float:
        """The fully open hand, in the hand's own units."""
        return self.hand_min if self.hand_close_is_max else self.hand_max

    @property
    def qdot_max_body(self) -> np.ndarray:
        """(17,) in body-target order: 14 arm joints then 3 waist joints."""
        idx = list(self.joints_arm) + list(self.joints_waist)
        return self.qdot_max[idx]


#: A relative model path in a robot file (`urdf = "assets/..."`) is relative to
#: the package's `robot/` directory, beside the shipped profiles.
MODEL_ROOT = Path(__file__).resolve().parents[1] / "robot"


def _resolve(p: str) -> str:
    """Model paths are written relative to the package root, not the cwd.

    A relative path here would mean the run works when launched from the checkout
    and silently loads nothing when launched from anywhere else — and mujoco's
    failure for a missing model arrives deep inside the backend, far from the
    line that chose it.
    """
    q = Path(p)
    q = q if q.is_absolute() else MODEL_ROOT / q
    if not q.exists():
        raise ConfigError(f"robot file [model] path does not exist: {q}")
    return str(q)


def _rt_priority(value) -> int | None:
    """`[cpu] rt_priority`, or None for "do not ask for real-time at all".

    Range-checked here rather than at the syscall: 99 is where the kernel's own
    migration and watchdog threads live, and a control loop that outranks them
    can starve the machine into a hard lockup.
    """
    if value is None:
        return None
    prio = int(value)
    if not 1 <= prio <= 98:
        raise ConfigError(
            f"machine file [cpu] rt_priority = {prio}; SCHED_FIFO priorities "
            f"run 1-99, and 99 is reserved for the kernel's own threads.")
    return prio


def parse_tcp(value, key: str = "[end_effector] tcp") -> np.ndarray | None:
    """`[end_effector] tcp`: `[x, y, z, qw, qx, qy, qz]`, flange frame.

    The quaternion must already be a unit one (to 1e-3) and is renormalized;
    a value far from unit is a typo or a different convention (xyzw), and
    either would rotate every intervention about the wrong axis."""
    if value is None:
        return None
    try:
        tcp = np.asarray([float(v) for v in value], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"robot file {key} must be 7 numbers "
                          f"[x, y, z, qw, qx, qy, qz], got {value!r}") from exc
    if tcp.shape != (schema.POSE_DIM,) or not np.all(np.isfinite(tcp)):
        raise ConfigError(f"robot file {key} must be 7 finite numbers "
                          f"[x, y, z, qw, qx, qy, qz], got {value!r}")
    n = float(np.linalg.norm(tcp[3:]))
    if abs(n - 1.0) > 1e-3:
        raise ConfigError(f"robot file {key}: the quaternion (qw, qx, qy, qz) "
                          f"has norm {n:.6f}, must be a unit quaternion, w first")
    tcp[3:] /= n
    return tcp


def _cpu(raw: dict) -> dict:
    """`[cpu.pin]`, or this machine's own topology if the table is absent.

    Stating the table wins: it is a deliberate allocation, and `validate_pinning`
    then refuses if the machine does not match it, which is what catches a table
    carried over from a different box.

    Leaving it out is any other machine's answer. `cpu_pinning.derive`
    applies the same rules to whatever CPU this actually is, and refuses
    loudly rather than degrading:

      * off Linux there is no `/sys` topology to read, so there is nothing to
        derive from and nothing to deploy to either -- it raises rather than
        handing back an empty table that would look like "no pinning wanted";
      * a CPU too small to give p7_body and p8_hand a physical core each, or to
        give the latency-path processes a performance thread each, is a machine
        this system should not start on.

    That is why tests/machine.toml states a table: a laptop runs the whole test
    suite through this function and cannot derive one.
    """
    if "pin" in raw:
        return dict(
            cpu={k: (list(v) if isinstance(v, list) else int(v))
                 for k, v in raw["pin"].items()},
            cpu_reserved=[int(v) for v in raw.get("reserved", [])],
            cpu_source="file",
        )

    derived = affinity.derive(error=ConfigError)
    return dict(
        cpu=derived["pin"],
        cpu_reserved=derived["reserved"],
        cpu_source="derived",
        cpu_notes=tuple(derived.get("notes", ())),
    )


def _env_or(name: str, value: str) -> str:
    """`$name` when set (one shell's override of a robot-file value), else `value`."""
    return os.environ.get(name, "").strip() or value


def load_hardware(robot=None, machine=None) -> Hardware:
    """`Hardware` for one robot file on one machine.

    `robot` is a `robot_file.Robot`, a shipped profile name or a path (default
    `robot_file.DEFAULT_ROBOT`); `machine` a `machine_file.Machine` or a path to a
    machine file (default: `machine_file.machine_path()`'s search). Everything a robot or
    machine file can be refused for is raised here as `ConfigError`, before an
    arena exists.
    """
    from elrobot_deploy.config_loading import machine_file as _machine
    from elrobot_deploy.config_loading import robot_file as _robot

    if not isinstance(robot, _robot.Robot):
        robot = _robot.load_robot(robot or _robot.DEFAULT_ROBOT)
    if not isinstance(machine, _machine.Machine):
        machine = _machine.load_machine(machine)
    raw = robot.raw

    j = raw["joints"]
    g = raw["gains"]
    n = schema.N_JOINTS

    def _expand(spec: dict, default: float) -> np.ndarray:
        """Gains are written per joint GROUP; the wire wants one value per
        joint. Expanding here means the toml stays readable and nothing
        downstream has to know which index is a wrist."""
        out = np.full(n, float(default))
        for name, idx in (("arm", j["arm"]), ("waist", j["waist"]),
                          ("leg", j["leg"])):
            if name in spec:
                out[list(idx)] = float(spec[name])
        for key, value in spec.items():
            if key.startswith("arm_"):
                out[list(j[key])] = float(value)
        return out

    ee_type, tcp = _robot.end_effector_of(raw["end_effector"])
    dex1 = end_effectors.get("dex1")
    cpu_raw = machine.cpu
    hw = Hardware(
        dds_iface=machine.iface,
        qdot_max=np.full(n, float(raw["limits"]["qdot_max"])),
        body_hz=float(raw["rates"]["body_hz"]),
        hand_hz=float(raw["rates"]["hand_hz"]),
        lowstate_hz=float(raw["rates"]["lowstate_hz"]),
        pico_poll_hz=float(raw["rates"].get("pico_poll_hz", 180.0)),
        **_cpu(cpu_raw),
        rt_priority=_rt_priority(cpu_raw.get("rt_priority")),
        dds_domain=int(raw["dds"]["domain"]),
        topic_lowcmd=raw["dds"]["topic_lowcmd"],
        topic_lowstate=raw["dds"]["topic_lowstate"],
        topic_hand_cmd=dex1.topic_hand_cmd,
        topic_hand_state=dex1.topic_hand_state,
        hand_min=dex1.hand_min,
        hand_max=dex1.hand_max,
        camera_host=_env_or("ELROBOT_ROBOT_IP", robot.board.ip),
        # The ONE port table every camera consumer reads (P2, CHECK, the
        # preflight probes). A robot file decides which port is which.
        camera_ports={name: int(spec.port) for name, spec in robot.cameras.items()
                      if spec.source == "zmq"},
        joints_arm=list(j["arm"]),
        joints_waist=list(j["waist"]),
        joints_leg=list(j["leg"]),
        kp=_expand(g["kp"], 0.0),
        kd=_expand(g["kd"], 0.0),
        damp_kp=float(g["damp"]["kp"]),
        damp_kd=float(g["damp"]["kd"]),
        urdf=_resolve(raw["model"]["urdf"]),
        wrist_payload_mass=float(
            raw["model"].get("wrist_payload", {}).get("mass", 0.0)),
        wrist_payload_com_x=float(
            raw["model"].get("wrist_payload", {}).get("com_x", 0.0)),
        waist_tau_offset=tuple(
            float(v) for v in g.get("waist_tau_offset", (0.0, 0.0, 0.0))),
        servo_model=_servo_model(raw.get("servo_model")),
        image_server=robot.image_server_spec.type,
        image_server_dir=robot.image_server_spec.dir,
        image_server_conda=robot.image_server_spec.conda,
        image_server_conda_env=robot.image_server_spec.conda_env,
        board_ssh_user=_env_or("ELROBOT_BOARD_SSH_USER", robot.board.ssh_user),
        board_ssh_password=_env_or("ELROBOT_BOARD_SSH_PASSWORD", robot.board.ssh_password),
        dashboard_port=machine.dashboard_port,
    )
    return for_hand(hw, ee_type, tcp=parse_tcp(tcp))


def _servo_model(raw: dict | None) -> ServoModel | None:
    if raw is None:
        return None
    arrays = {}
    for key in ("inertia", "viscous", "coulomb"):
        values = tuple(float(v) for v in raw[key])
        if len(values) != schema.ARM_DOF:
            raise ConfigError(f"[servo_model] {key} must list {schema.ARM_DOF} "
                              f"values (left arm then right), got {len(values)}")
        arrays[key] = values
    return ServoModel(**arrays, eps_v=float(raw.get("eps_v", 0.02)),
                      tau_max=float(raw.get("tau_max", 2.0)))


def for_hand(hw: Hardware, hand: str, *, tcp: np.ndarray | None = None) -> Hardware:
    """`hw` with end effector `hand`'s built-in topics and travel
    (`end_effectors.END_EFFECTORS`) and the given `tcp` swapped in, once, so
    P1, P8, P11 and CHECK cannot disagree about which hand they are talking to.
    The TCP belongs to the end effector type, so it is always set with it."""
    if hand not in HANDS:
        raise ConfigError(f"end effector {hand!r} is not one of {list(HANDS)}")
    ee = end_effectors.get(hand)
    return replace(hw, hand=hand, topic_hand_cmd=ee.topic_hand_cmd,
                   topic_hand_state=ee.topic_hand_state, hand_min=ee.hand_min,
                   hand_max=ee.hand_max, tcp=tcp)


__all__ = ["CAMERA_MIN_SLOTS", "CAMERA_SOURCES",
           "CameraSpec", "CameraView", "SPLIT_HALVES", "ServoModel",
           "camera_slots", "camera_views", "not_a_view_hint", "split_names",
           "views_of", "lens_square",
           "ServoShaping",
           "DEX1_TRAVEL", "DeployConfig", "HANDS",
           "Hardware", "IK_MODES", "IK_SOLVERS", "for_hand",
           "load_hardware", "next_pow2", "replace"]

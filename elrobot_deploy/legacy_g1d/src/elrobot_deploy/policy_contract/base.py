"""The policy class, and how the infra finds, checks and binds one.

Runs in P5 (the adapter's own interpreter); the launcher and the inspection
dashboard use the checks. Design: `docs/deploying_a_policy/policy_contract.md`
("Declarations", "Methods", "Lifecycle").

A policy is a class. It declares the facts the infra cannot guess —
`action_mode`, `end_effector` and `anchor_camera` — and implements one method,
`infer`. The infra owns everything else: the anchor (`anchor.py`), the
observation and its RGB conversion (`observation.py`), the deadline shift
(`deadline_shift.py`), sync truncation, validation and the ring write
(`runtime/processes/p5_policy.py`). What is left for the policy is whatever
turns an observation into actions — for a learned policy the preprocessing,
normalization, model call and inverse transform; for a scripted one, its rule.

LIFECYCLE (P5, once per process unless noted)

1. `loading.load_entry` imports the class named by `[policy] entry`;
   `validate_policy_class` checks its declarations — before anything is
   constructed, so a typo in `action_mode` costs no slow load.
2. `check_against(cls, cfg)` checks them against this run's cameras and ring
   depth.
3. `cls(checkpoint, prompt, robot)` — the CHEAP load (for a network: config,
   tokenizer, norm stats, no GPU). The inspection dashboard's prompt check
   constructs the class and stops here (`count_prompt_tokens`,
   `check_prompt_length`).
4. `bind_runtime` hands the policy the infra-built dummy observation and the
   reader; `policy.warmup()` runs — anything slow, such as a weight restore
   and a compile (default: one `infer` on the dummy). Explicit, never lazy in
   `infer`: lazy loading makes a 40 s compile indistinguishable from a hang.
5. P5 times further dummy `infer` calls itself until latency settles —
   scheduling needs a measured latency, and only P5 can measure it.
6. Per rollout, per replan: `infer(obs, reader)`.

`robot` is the `Hardware` the robot file resolves to
(`config_loading/resolved.py`, alias `RobotSpec`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from elrobot_deploy.robot import end_effectors
from elrobot_deploy.config_loading.resolved import Hardware, not_a_view_hint
from elrobot_deploy.policy_contract.actions import ACTION_MODES, EEF, EefActions
from elrobot_deploy.policy_contract.observation import Observation, Reader

#: What `Policy.__init__` receives as `robot`: the `Hardware` built from the
#: robot file (`docs/setup/robot_toml.md`).
RobotSpec = Hardware


class Policy:
    """Subclass this. See `docs/deploying_a_policy/policy_contract.md` for the whole contract."""

    # --- declarations (class attributes) ----------------------------------

    #: REQUIRED. `"eef"` (flange poses, P6 runs IK) or `"joint"` (joint angles,
    #: P6 skips only the IK). Checked at load.
    action_mode: str | None = None

    #: REQUIRED. The end effector the policy's `hand` actions speak —
    #: `"dex1"` (jaw gear radians, 0 = closed). Checked at load against the
    #: robot file's `[end_effector] type`: a policy written for one hand is
    #: never run on another. `"brainco"` is reserved (not in the contract yet).
    end_effector: str | None = None

    #: REQUIRED. The camera whose frame time is "now": its frame's `t_recv` is
    #: `obs.t_anchor`, and the state is matched to it (`anchor.py`). For a wrist-mounted
    #: acting camera this must be that camera, or image and flange describe
    #: different instants — measured at half a frame period: p99 5.4 mm.
    anchor_camera: str | None = None

    #: Optional. The cameras this policy reads. Unset: every camera the run
    #: observes (`DeployConfig.observed`). Set: exactly these are anchored and
    #: delivered, the dashboard marks them, the manifest records them.
    uses_cameras: tuple[str, ...] | None = None

    #: Optional. Seconds of history per stream, e.g.
    #: `{"state": 2.0, "head_left": 4.0}`. A stream is `"state"` or a camera
    #: view name (a side_by_side head is `head_left` / `head_right`). Only the
    #: declared streams can be read through `reader`, and each camera's ring is
    #: sized from this at launch (`resolved.camera_slots`): `ceil(s x fps x 1.25) + 3`
    #: slots, ~0.9 MB each for a 640x480 view.
    history: dict[str, float] | None = None

    #: Optional. The prompt length the adapter's forward pass is BUILT for, in
    #: the policy's own tokens: the fixed shape it pads the prompt to and
    #: compiles at in `warmup`. Not a training limit — a model trained on
    #: shorter prompts may cope with longer ones — only what fits the compiled
    #: input. Only the policy owns a tokenizer, so it also supplies the count
    #: (`count_prompt_tokens`); `check_prompt_length` compares the two.
    max_prompt_tokens: int | None = None

    #: Optional. Re-time the chunk onto its real deadlines (`deadline_shift.py`). On by
    #: default because off is the version with a known error in it; a field so
    #: the two can be compared on the robot.
    deadline_shift: bool = True

    #: Optional. Where, in the FLANGE frame, eef poses are interpolated by the
    #: deadline shift — the policy's own TCP if its actions were labelled at
    #: one (`deadline_shift.py`, "THE INTERPOLATION POINT"). Metres.
    deadline_shift_point: tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: Optional. The control rate (Hz) and the action horizon (actions per
    #: chunk) the policy was made for — for a learned policy, what it was
    #: trained at — or None when unknown or not applicable. Read from the
    #: CLASS at resolution, before anything is constructed
    #: (`config_loading/deploy_file.py::resolve`): they sit between the infra
    #: defaults and the deploy file. A horizon above this one is refused; a
    #: different control rate is a warning (a deliberate speed A/B).
    trained_control_hz: float | None = None
    trained_action_horizon: int | None = None

    # --- construction -----------------------------------------------------

    def __init__(self, checkpoint: Path | None, prompt: str, robot: RobotSpec):
        """Load what is cheap here (for a network: config, tokenizer, norm
        stats) and leave anything slow — a weight restore, a compile, opening
        a connection — to `warmup()`: the dashboard's prompt check constructs
        the class and never warms it up. `checkpoint` is None for a policy
        that loads nothing from disk (a scripted controller; the hold and toy
        examples)."""
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.prompt = prompt
        self.robot = robot

    # --- the one required method -------------------------------------------

    def infer(self, obs: Observation, reader: Reader):
        """`obs` -> `EefActions` | `JointActions`, timed from `obs.t_anchor`.

        Return the whole chunk: P5 cuts it to `H`, and in sync to `P` (after
        the deadline shift). In sync at least `P`. `reader.arena` offers
        read-only access to the rings; a policy never writes one.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement infer()")

    # --- optional ---------------------------------------------------------

    def warmup(self) -> None:
        """Do the slow part of loading and warm up. Default: one `infer` on
        the infra's dummy observation, whose camera shapes and state layout
        are this run's; a policy with weights restores them first, then calls
        this."""
        self.infer(self.dummy_observation(), self.reader())

    def count_prompt_tokens(self, text: str) -> int | None:
        """How many tokens `text` becomes as this policy consumes it —
        including any template the policy wraps around the task text. None
        (the default) = the policy does not count (or has no tokenizer), and
        no prompt is refused for its length."""
        return None

    def check_prompt_length(self, text: str) -> None:
        """Raise `PromptTooLong` if `text` exceeds `max_prompt_tokens`.

        Called when the operator SUBMITS a prompt on the dashboard (P5 judges
        it between inferences) and again before `set_prompt` stores one. A
        policy that sets neither `max_prompt_tokens` nor `count_prompt_tokens`
        accepts every string."""
        if not isinstance(text, str):
            raise ValueError(f"prompt must be a str, got {type(text).__name__}")
        if self.max_prompt_tokens is None:
            return
        n = self.count_prompt_tokens(text)
        if n is not None and n > self.max_prompt_tokens:
            raise PromptTooLong(n, self.max_prompt_tokens)

    def set_prompt(self, prompt: str) -> None:
        """A new prompt between rollouts. Checked first."""
        self.check_prompt_length(prompt)
        self.prompt = prompt

    def on_settings(self, settings) -> None:
        """New rollout settings (`config_loading/rollout_settings.py`), e.g.
        `sample_steps`. Default: ignore."""

    @classmethod
    def resolve_checkpoint(cls, run: Path, step: str) -> Path | None:
        """`[checkpoint] run` + `step` -> the path to load, for a policy
        whose layout is its own (aliases like "best"). Return None (the
        default) for the infra's rule: numbered step subdirectories, "latest"
        = the highest (`checkpoints.resolve`). Never called when the deploy
        file names no checkpoint."""
        return None

    # --- what the infra binds ----------------------------------------------

    def dummy_observation(self) -> Observation:
        rt = getattr(self, "_g1d_runtime", None)
        if rt is None:
            raise RuntimeError(
                "no dummy observation bound: the infra calls bind_runtime() "
                "before warmup(). Did a subclass override __init__ and forget "
                "to call super().__init__?")
        return rt[0]

    def reader(self) -> Reader:
        rt = getattr(self, "_g1d_runtime", None)
        return Reader(None, dict(self.history or {})) if rt is None else rt[1]

    @property
    def settings(self):
        """The current rollout's settings (`RolloutSettings`: `sync`,
        `replan_period`, `control_hz`, `ik`, `sample_steps`, `servo`): the
        launch's until a Start changes them, then that rollout's. Bound by the
        infra before `warmup()`; None inside `__init__`. `on_settings` is the
        hook for reacting to a change; this is for reading them any time."""
        return getattr(self, "_g1d_settings", None)

    @property
    def cameras(self) -> dict:
        """View name -> `CameraInfo`: size, rate, lens kind, where this
        run's fisheye square was cut, calibration. Bound by the infra before `warmup()`;
        empty inside `__init__`."""
        return getattr(self, "_g1d_cameras", {})


@dataclass(frozen=True)
class CameraInfo:
    """What the infra knows about one camera VIEW (`Policy.cameras[name]`).

    Frames reach `obs` and the reader at native resolution, RGB. A
    `fisheye180` view is ALREADY the square around its lens circle — the infra
    cut it, where the launch's fit found the lens — so a policy applies no crop
    of its own; a normal view is the whole frame (or one eye of the head).
    Any resize the policy needs is the policy's.
    """

    name: str                       # the view: what obs.frames / reader use
    source: str                     # the robot-file camera it is cut from
    height: int                     # this view's frame, pixels (a fisheye's
    width: int                      #   square side, both)
    x0: int                         # its first column in the wire frame
    hz: float                       # the frame rate the robot file states
    kind: str                       # "normal" | "fisheye180"
    #: For information: where a `fisheye180` view's square sits on the
    #: sensor, `(x0, y0, x1, y1)` (it may run off it; that part is black).
    #: The pixels are already cut; nothing needs this to use them. None for a
    #: normal lens.
    crop: tuple | None = None
    #: Where `crop` came from (`resolved.lens_square`): "fitted" (this
    #: launch's fit, or the one a record carries), "fallback_lens_circle" (no fit
    #: ran: the robot file's circle), "placeholder" (neither: shape only).
    crop_source: str | None = None
    #: The lens circle `(center_x, center_y, radius)` in full-sensor pixels
    #: the square was cut around, when known.
    lens_circle: tuple | None = None
    #: The robot file's optional calibration, untouched; the infra reads neither.
    intrinsics: dict | None = None
    extrinsics: dict | None = None


def camera_info(cfg) -> dict[str, CameraInfo]:
    """Every view of `cfg`'s robot -> its `CameraInfo`."""
    out = {}
    for name, v in cfg.views.items():
        spec = cfg.cameras[v.source]
        crop = how = circle = None
        if v.cropped:
            from elrobot_deploy.config_loading.resolved import lens_square
            crop, how = lens_square(spec)
            circle = spec.lens_circle if spec.crop is not None else spec.fallback_lens_circle
        out[name] = CameraInfo(
            name=name, source=v.source, height=int(v.height), width=int(v.width),
            x0=int(v.x0), hz=float(v.hz), kind=str(getattr(spec, "kind", "normal")),
            crop=None if crop is None else tuple(int(c) for c in crop),
            crop_source=None if crop is None else how,
            lens_circle=None if circle is None else tuple(float(c) for c in circle),
            intrinsics=getattr(spec, "intrinsics", None),
            extrinsics=getattr(spec, "extrinsics", None))
    return out


def bind_runtime(policy: Policy, dummy: Observation, reader: Reader,
                 cameras: dict | None = None, settings=None) -> None:
    """Give the policy what its default `warmup` needs, `cameras`
    (`camera_info(cfg)`) and the launch's rollout `settings`
    (`RolloutSettings.of(cfg)`). Infra-only."""
    policy._g1d_runtime = (dummy, reader)
    policy._g1d_cameras = dict(cameras or {})
    policy._g1d_settings = settings


def deliver_settings(policy: Policy, settings) -> None:
    """The settings a rollout will run with become `policy.settings`.
    Infra-only: called at every accepted Start, after `on_settings` (if it
    ran) succeeded."""
    policy._g1d_settings = settings


# --- checking a class --------------------------------------------------------


class PolicyError(TypeError):
    """A policy class whose declarations break the contract."""


class PromptTooLong(ValueError):
    """A prompt longer than the policy's `max_prompt_tokens`. Its message is
    what the dashboard shows the operator, verbatim."""

    def __init__(self, n_tokens: int, max_tokens: int):
        self.n_tokens, self.max_tokens = int(n_tokens), int(max_tokens)
        super().__init__(
            f"Prompt exceeds the policy's maximum token length "
            f"({self.n_tokens} tokens > {self.max_tokens}).")


def validate_policy_class(cls) -> None:
    """The declarations, checked at load — before anything is constructed."""
    name = getattr(cls, "__name__", repr(cls))
    if not (isinstance(cls, type) and issubclass(cls, Policy)):
        raise PolicyError(f"{name} is not a subclass of elrobot_deploy.policy_contract.Policy")
    if cls.action_mode not in ACTION_MODES:
        raise PolicyError(
            f"{name}.action_mode = {cls.action_mode!r}; it is REQUIRED and must "
            f"be one of {ACTION_MODES}")
    supported = [t for t, e in end_effectors.END_EFFECTORS.items() if e.contract]
    if cls.end_effector not in end_effectors.TYPES:
        raise PolicyError(
            f"{name}.end_effector = {cls.end_effector!r}; it is REQUIRED — the "
            f"end effector its `hand` actions speak, one of {supported}")
    if not end_effectors.get(cls.end_effector).contract:
        raise PolicyError(
            f"{name}.end_effector = {cls.end_effector!r}: the {cls.end_effector} "
            f"is not supported by the policy action contract yet (only "
            f"{supported}). Its wire exists, but what `hand` means for it has "
            f"not been decided.")
    if not isinstance(cls.anchor_camera, str) or not cls.anchor_camera:
        raise PolicyError(
            f"{name}.anchor_camera = {cls.anchor_camera!r}; it is REQUIRED — the "
            f"camera whose frame time is obs.t_anchor")
    if cls.uses_cameras is not None:
        cams = cls.uses_cameras
        if isinstance(cams, str) or not all(isinstance(c, str) for c in cams) \
                or not cams:
            raise PolicyError(
                f"{name}.uses_cameras must be a non-empty tuple of camera "
                f"names, got {cams!r}")
        if cls.anchor_camera not in cams:
            raise PolicyError(
                f"{name}.anchor_camera {cls.anchor_camera!r} is not in "
                f"uses_cameras {tuple(cams)}")
    if cls.history is not None:
        if not isinstance(cls.history, dict):
            raise PolicyError(f"{name}.history must be a dict, got {cls.history!r}")
        for stream, seconds in cls.history.items():
            if not isinstance(stream, str) or not (
                    isinstance(seconds, (int, float)) and math.isfinite(seconds)
                    and seconds > 0):
                raise PolicyError(
                    f"{name}.history[{stream!r}] = {seconds!r}; want positive "
                    f"seconds per stream name")
    point = np.asarray(cls.deadline_shift_point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise PolicyError(f"{name}.deadline_shift_point must be 3 finite metres")
    hz, h = cls.trained_control_hz, cls.trained_action_horizon
    if hz is not None and not (isinstance(hz, (int, float)) and not isinstance(hz, bool)
                               and math.isfinite(hz) and hz > 0):
        raise PolicyError(f"{name}.trained_control_hz = {hz!r}; want a positive "
                          f"number of Hz, or None")
    if h is not None and not (isinstance(h, int) and not isinstance(h, bool) and h > 0):
        raise PolicyError(f"{name}.trained_action_horizon = {h!r}; want a "
                          f"positive int, or None")
    if cls.infer is Policy.infer:
        raise PolicyError(f"{name} does not implement infer()")


def observation_cameras(cls, cfg) -> tuple[str, ...]:
    """Which cameras are anchored and delivered in `obs.frames`: the resolved
    `cfg.observed_cameras` (the class's `uses_cameras`, or the deploy file's
    override of it), else the class's `uses_cameras` (a config built by hand),
    else every camera. The anchor camera is always among them."""
    if cfg.observed_cameras:
        return tuple(cfg.observed_cameras)
    return tuple(cls.uses_cameras) if cls.uses_cameras is not None else cfg.observed


def anchor_camera(cls, cfg) -> str:
    """Whose frame time is `t_anchor`: the resolved `cfg.primary_camera` (the
    class's `anchor_camera`, or the deploy file's override) when it is one of
    `observation_cameras`, else the class's own."""
    if cfg.primary_camera and cfg.primary_camera in observation_cameras(cls, cfg):
        return cfg.primary_camera
    return cls.anchor_camera


def class_name(cls) -> str:
    """`cls.__name__` — for a class, or for a `describe.PolicyInfo` standing
    in for one that lives in another interpreter."""
    return getattr(cls, "__name__", None) or getattr(cls, "name", repr(cls))


def check_end_effector(cls, hw) -> None:
    """The class's `end_effector` against the robot's `[end_effector] type`.
    Raises `PolicyError` naming both, and for a type the contract does not
    speak (the BrainCo) says so rather than the mismatch. `cls` may be a
    `describe.PolicyInfo`."""
    name = class_name(cls)
    robot = getattr(hw, "hand", None)
    if robot is None:
        return
    if not end_effectors.get(robot).contract:
        raise PolicyError(
            f"this robot's end effector is {robot!r}, which the policy action "
            f"contract does not support yet: no policy can run on it "
            f"(robot/end_effectors.py). `elrobot_deploy check` still works on this rig.")
    if cls.end_effector != robot:
        raise PolicyError(
            f"{name}.end_effector = {cls.end_effector!r} but the robot file's "
            f"[end_effector] type is {robot!r}: the policy's `hand` actions "
            f"would be read in the wrong units")


def check_against(cls, cfg, hw=None) -> None:
    """The declarations against THIS run: its cameras, its ring depth, and —
    given the robot — its end effector. `cls` may be a `describe.PolicyInfo`."""
    name = class_name(cls)
    if hw is not None:
        check_end_effector(cls, hw)
    cams = observation_cameras(cls, cfg)
    views = cfg.views
    unknown = sorted(set(cams) - set(views))
    if unknown:
        raise PolicyError(f"{name} uses cameras {unknown} that this run does "
                          f"not have; it has {sorted(views)}."
                          + "".join(not_a_view_hint(c, cfg.cameras) for c in unknown))
    if cfg.primary_camera and cfg.primary_camera != cls.anchor_camera \
            and cfg.primary_camera not in cams:
        # A deploy-file override that names a camera not delivered would
        # otherwise fall back to the class's anchor without a word.
        raise PolicyError(
            f"[policy] anchor_camera {cfg.primary_camera!r} is not among the "
            f"observed cameras {sorted(cams)}; add it to uses_cameras")
    if anchor_camera(cls, cfg) not in cams:
        raise PolicyError(
            f"{name}.anchor_camera {cls.anchor_camera!r} is not among the "
            f"observed cameras {sorted(cams)}; set uses_cameras, or the run's "
            f"observed_cameras, to include it")
    hist = cls.history or {}
    if hist.get("state", 0.0) > cfg.state_history_s:
        raise PolicyError(
            f"{name}.history['state'] = {hist['state']} s exceeds this run's "
            f"state ring ({cfg.state_history_s} s, DeployConfig.state_history_s)")
    stray = sorted(set(hist) - {"state"} - set(views))
    if stray:
        raise PolicyError(f"{name}.history names unknown streams {stray}; the "
                          f"streams are 'state' and the cameras {sorted(views)}."
                          + "".join(not_a_view_hint(c, cfg.cameras) for c in stray))
    # The camera rings were sized at launch from the history the RESOLVED
    # config carries (`DeployConfig.camera_history_s`, filled by
    # `deploy_file.resolve` from this very class). A class asking for more than
    # its ring holds would get a window silently cut short, so it is refused.
    short = sorted(c for c, s in hist.items()
                   if c != "state" and s > views[c].history_s + 1e-9)
    if short:
        raise PolicyError(
            f"{name}.history asks for "
            + ", ".join(f"{c} {hist[c]} s" for c in short)
            + " but this run's camera rings hold "
            + ", ".join(f"{c} {views[c].history_s} s" for c in short)
            + ". The rings are sized at launch from the resolved config's "
            f"camera_history_s (deploy_file.resolve fills it from the policy "
            f"class); launch through a deploy file, or set it.")


# --- the built-in policy -----------------------------------------------------


class EchoPolicy(Policy):
    """Holds position: every action is the flange pose the anchor measured.

    Exists so the whole system is runnable with no policy named. It exercises
    the anchor, the timing, the rings, the IK and the wire; the only thing not
    under test is a policy. `anchor_camera` and `horizon` are bound per run by
    `make_policy`, since it belongs to no rig.
    """

    action_mode = EEF
    end_effector = "dex1"
    anchor_camera = "unbound"
    horizon = 1

    def infer(self, obs, reader) -> EefActions:
        n = self.horizon
        lo, hi = (self.robot.hand_min, self.robot.hand_max) \
            if self.robot is not None else (0.0, 5.4)
        return EefActions(
            pose=np.repeat(obs.state.flange[None], n, axis=0),
            hand=np.repeat(np.clip(obs.state.hand, lo, hi)[None], n, axis=0))


def make_policy(cfg, hw) -> Policy:
    """P5's construction step: find, check, construct.

    The class is `cfg.policy_entry` (`loading.load_entry`, the one loader); a
    config that names none runs `EchoPolicy`."""
    if getattr(cfg, "policy_entry", ""):
        from elrobot_deploy.policy_contract.loading import load_entry
        cls = load_entry(cfg.policy_entry)
    else:
        cls = type("EchoPolicy", (EchoPolicy,),
                   {"anchor_camera": cfg.primary_camera, "horizon": cfg.H})
    validate_policy_class(cls)
    check_against(cls, cfg, hw)
    policy = cls(Path(cfg.checkpoint) if cfg.checkpoint else None, cfg.prompt, hw)
    h, hz = cls.trained_action_horizon, cls.trained_control_hz
    if (h is not None and h != cfg.H) or (
            hz is not None and abs(hz - cfg.control_hz) > 1e-9):
        print(f"[P5] WARNING: {cls.__name__} was trained at H={h}, "
              f"{hz} Hz; this run uses H={cfg.H}, {cfg.control_hz:g} Hz",
              flush=True)
    return policy


__all__ = ["EchoPolicy", "Policy", "PolicyError", "RobotSpec", "bind_runtime",
           "check_against", "class_name", "check_end_effector", "make_policy", "observation_cameras",
           "validate_policy_class"]

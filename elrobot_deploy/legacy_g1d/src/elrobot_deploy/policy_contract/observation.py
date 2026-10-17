"""What a policy sees: the observation (`obs`) and the history reader.

Built by P5 for every `infer` call (`runtime/processes/p5_policy.py`), and by
the inspection replay from a record. Reads the arena; writes nothing. Design:
`docs/deploying_a_policy/policy_contract.md`, "The observation (`obs`)",
"Cameras" and "History: `reader`".

The infra builds an `Observation` out of the `Anchor` the anchor rule chose
(`anchor.py`), and a policy needs nothing else to run: no ring, no seqlock, no
sequence counter, no dtype. That is what lets the arena's layout change without
a single policy changing, and what keeps every reading rule — copy-and-verify,
the nearest-frame rule, the state match — in one place that can be tested once.

For a policy that needs what `obs` and the reader's windows do not give —
where execution is now (`k_exec`), the chunks it published and when each tick
is due (real-time chunking), P6's targets, P7's commands — `reader.arena` is
every region, read-only (`ArenaRings`). Those reads follow the arena's own
rules and are versioned with it (`docs/infra_logic/arena_layout_and_reads.md`).

**RGB, converted once, here.** Every wire source publishes BGR on purpose
(`runtime/drivers/cameras.py`: one convention across ZMQ, V4L2 and RealSense), and
image pipelines almost universally work in RGB (`PIL .convert("RGB")`). Getting it
wrong produces no error and no warning — the policy simply behaves as though it
had never seen the scene — so it is done by the infra, for every policy, and no
policy gets the chance to forget.

**One clock.** Every `t_recv` here is box-monotonic seconds, stamped on
receipt. The camera-to-box transport delay is invisible to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from elrobot_deploy.runtime.shared_memory import schema

#: Arena cameras whose frames arrive BGR. Every current source does
#: (`runtime/drivers/cameras.py`), so the conversion is unconditional; this names the
#: assumption so a future RGB source is one line here instead of a silent swap.
WIRE_IS_BGR = True


@dataclass(frozen=True)
class Frame:
    """One camera frame. `pixels` is uint8 `(H, W, 3)` **RGB**, a private copy.

    `frame_id` is the camera ring's own counter for this frame (1, 2, 3, ... per
    camera view, per session) — what `reader.frames_by_id` takes, and how a policy
    tells which of the ids it asked for came back. 0 on the warmup dummy.
    """

    pixels: np.ndarray
    t_recv: float
    frame_id: int = 0


@dataclass(frozen=True)
class State:
    """The robot state sample nearest `t_anchor` (the anchor rule's state match).

    `q`, `dq`       (29,)   G1_29 wire order, radians / rad/s — the arm is
                            indices 15..28, the waist 12..14
                            (`docs/deploying_a_policy/policy_contract.md`,
                            "Joint order and limits")
    `flange`        (2, 7)  measured, FK of this `q` (xyz + wxyz, base frame)
    `hand`          (2,)    measured hand, the end effector's own quantity
                            (Dex-1: jaw gear radians, 0 = closed) — the same
                            units `EefActions.hand` / `JointActions.hand` take
    """

    t_recv: float
    q: np.ndarray
    dq: np.ndarray
    flange: np.ndarray
    hand: np.ndarray


@dataclass(frozen=True)
class Observation:
    """The aligned present. Built by the infra, read-only by contract."""

    t_anchor: float
    frames: dict[str, Frame]
    state: State
    prompt: str
    #: The instant P5 scheduled this inference for. A fact about the SCHEDULE
    #: where `t_anchor` is a fact about the WORLD; the infra already corrects
    #: for their difference (the deadline shift), so a policy should not need
    #: it — it is here for logging.
    t_trigger: float = float("nan")


class StateWindow(NamedTuple):
    """Every state sample captured in a window, ascending `t_recv`. `(m, ...)`."""

    t_recv: np.ndarray      # (m,)
    q: np.ndarray           # (m, 29)
    dq: np.ndarray          # (m, 29)
    flange: np.ndarray      # (m, 2, 7)
    hand: np.ndarray        # (m, 2)


def _state(rec) -> State:
    """A `schema.STATE` record -> `State`. Copies, so nothing aliases the ring."""
    hand = np.array(rec["grip_meas"], dtype=np.float64).reshape(
        schema.N_HANDS, schema.HAND_DIM)
    return State(t_recv=float(rec["t_recv"]),
                 q=np.array(rec["q"], dtype=np.float64),
                 dq=np.array(rec["dq"], dtype=np.float64),
                 flange=np.array(rec["flange"], dtype=np.float64),
                 hand=hand[:, 0].copy())


def to_rgb(pixels: np.ndarray) -> np.ndarray:
    """The wire's BGR -> RGB, as a fresh contiguous array."""
    img = np.asarray(pixels)
    return np.ascontiguousarray(img[..., ::-1] if WIRE_IS_BGR else img)


def observe(anchor, prompt: str) -> Observation:
    """Build the policy-facing observation from the instant the anchor chose."""
    frames = {name: Frame(pixels=to_rgb(rec["pixels"]),
                          t_recv=float(rec["t_recv"]),
                          frame_id=int(rec["frame_id"]))
              for name, rec in anchor.frames.items()}
    return Observation(t_anchor=float(anchor.t_anchor), frames=frames,
                       state=_state(anchor.state), prompt=prompt,
                       t_trigger=float(anchor.t_trigger))


def dummy_observation(cameras: dict, prompt: str) -> Observation:
    """A correctly-SHAPED observation for warmup, built from the camera views.

    Shapes are what matter: a differently-shaped dummy compiles a different
    graph and the real path pays the compile anyway. So every camera the
    policy will see is here at its native `(height, width, 3)` — one eye's shape
    for a split head — and the state has the ring's layout. The values are
    neutral rather than plausible: zero joints, identity flange orientations, a
    closed hand. `t_anchor = 0`, and `DummyReader` builds its history
    backwards from that instant.
    """
    frames = {name: Frame(np.zeros((spec.height, spec.width, 3), np.uint8), 0.0)
              for name, spec in cameras.items()}
    return Observation(t_anchor=0.0, frames=frames, state=_neutral_state(0.0),
                       prompt=prompt, t_trigger=0.0)


def _neutral_state(t: float) -> State:
    flange = np.zeros((schema.N_HANDS, schema.POSE_DIM))
    flange[:, 3] = 1.0
    hand = np.zeros((schema.N_HANDS, schema.HAND_DIM))
    return State(t_recv=t, q=np.zeros(schema.N_JOINTS),
                 dq=np.zeros(schema.N_JOINTS), flange=flange,
                 hand=hand[:, 0].copy())


class FrameTimes(NamedTuple):
    """`reader.frame_ids_between`: which frames exist in a window. Metadata only.

    `ids` are the frames' `frame_id`s — what `reader.frames_by_id` takes — and are
    ascending, as are the times. `(m,)` each; `m = 0` for an empty window.
    """

    ids: np.ndarray         # (m,) int64
    t_recv: np.ndarray      # (m,) float64


_EMPTY_TIMES = FrameTimes(np.empty(0, np.int64), np.empty(0, np.float64))


@dataclass
class Reader:
    """History. Everything captured in a window, each item
    with its own timestamp; alignment and subsampling are the policy's.

    **Copy-and-verify is the only read path**: every item is
    copied out of the arena and its slot's sequence counter checked across the
    copy; an item overwritten mid-copy is DROPPED, never returned torn and never
    retried (it is gone — the writer only returns to a slot a ring's length
    later). So a window can be one item short at its leading edge, and a
    `frames(ids)` call can return fewer frames than ids — check `frame_id`.
    Nothing returned aliases shared memory; a policy may keep it as long as it
    likes.

    Windows are closed, `t0 <= t_recv <= t1`, box-monotonic seconds. Early in a
    rollout, or for a window reaching past what the ring holds, the reader
    returns what exists — never padding, never an error.

    `history` is the policy's declaration (`Policy.history`), and it is
    enforced: a stream the policy did not declare cannot be read, because what
    was declared is what the arena was sized for and an undeclared
    read would work today by accident and fail on the next layout.

    Frames are **RGB**, like `obs.frames`: the conversion happens in the copy.
    """

    state_ring: object = None
    history: dict = field(default_factory=dict)
    #: View name -> `shared_memory.rings.CameraRing`. Only the declared ones are needed.
    cameras: dict = field(default_factory=dict)
    #: Every arena region, read-only (`shared_memory.ArenaRings`), for reads
    #: the windows above do not offer. Not bound by `history`: what it holds
    #: is whatever each ring still has. **None** outside a live rollout — in
    #: `warmup` (the dummy reader) and in `check-policy` — so a policy that
    #: uses it must have a path for None, and must not change its compiled
    #: graph on it.
    arena: object = None

    # --- state ---------------------------------------------------------------

    def states_between(self, t0: float, t1: float) -> StateWindow:
        """Every state sample with `t0 <= t_recv <= t1`, ascending (~1 kHz)."""
        self._declared("state", "states")
        _check_window(t0, t1)
        recs = (np.empty(0, schema.STATE) if self.state_ring is None
                else self.state_ring.window(t0, t1))
        return _state_window(recs)

    # --- frames --------------------------------------------------------------

    def frame_ids_between(self, cam: str, t0: float, t1: float) -> FrameTimes:
        """Which frames of `cam` have `t0 <= t_recv <= t1`: ids and times,
        ascending. No pixels are copied — pick ids, then call `frames`."""
        ring = self._ring(cam)
        _check_window(t0, t1)
        if ring is None:
            return _EMPTY_TIMES
        ids, t = ring.window(t0, t1)
        return FrameTimes(ids, t)

    def frames_by_id(self, cam: str, ids) -> list[Frame]:
        """Exactly the frames `ids` (from `frame_ids_between`), RGB private copies,
        in the order asked. A frame no longer in the ring, or overwritten while
        it was being copied, is left out — so compare `frame_id`s when every
        one matters."""
        ring = self._ring(cam)
        if ring is None:
            return []
        out = []
        for fid in np.asarray(ids, dtype=np.int64).reshape(-1):
            got = ring.read_frame(int(fid), flip_channels=WIRE_IS_BGR)
            if got is not None:
                out.append(Frame(pixels=got[0], t_recv=got[1], frame_id=int(fid)))
        return out

    def frames_between(self, cam: str, t0: float, t1: float) -> list[Frame]:
        """Every frame in the window: `frames_by_id(cam, frame_ids_between(cam, t0, t1).ids)`.
        One ~0.9 MB copy per frame — a 1 fps policy should pick ids instead."""
        return self.frames_by_id(cam, self.frame_ids_between(cam, t0, t1).ids)

    # --- checks --------------------------------------------------------------

    def _declared(self, stream: str, call: str) -> None:
        if stream not in self.history:
            raise LookupError(
                f"reader.{call}() needs `history = {{{stream!r}: seconds, ...}}` "
                f"declared on the policy class")

    def _ring(self, cam: str):
        self._declared(cam, "frame_ids_between/frames_by_id")
        if cam not in self.cameras:
            if self.state_ring is None and not self.cameras:
                return None             # an unbound reader: nothing captured yet
            raise LookupError(
                f"no camera {cam!r} in this run; its cameras are "
                f"{sorted(self.cameras)}")
        return self.cameras[cam]


class DummyReader(Reader):
    """The warmup reader: every declared stream FULL, at its real rate.

    The point of warmup is to compile the graph the run will execute, and a
    policy that stacks 4 s of frames compiles a different graph for 120 frames
    than for the 0 an empty arena would give it. So this reader invents the
    whole declared history, ending at the dummy observation's `t_anchor = 0`:
    frames at `0, -1/hz, -2/hz, ...` back to `-history`, state samples at the
    robot's lowstate rate likewise. Pixels are black and state is the neutral
    dummy state; ids and times are consistent, so `frame_ids_between` -> `frames`,
    `nearest` and `every` all behave as they will live.
    """

    def __init__(self, cameras: dict, history: dict, *, state_hz: float = 1000.0):
        super().__init__(None, dict(history), {})
        #: View name -> anything with `height`, `width`, `hz`.
        self.specs = dict(cameras)
        self.state_hz = float(state_hz)

    @staticmethod
    def _timeline(seconds: float, hz: float) -> np.ndarray:
        """Ascending instants `-(K/hz) .. 0`, `K = ceil(seconds * hz)`."""
        k = int(np.ceil(seconds * hz - 1e-9))
        return -np.arange(k, -1, -1, dtype=np.float64) / hz

    def states_between(self, t0: float, t1: float) -> StateWindow:
        self._declared("state", "states")
        _check_window(t0, t1)
        t = self._timeline(float(self.history["state"]), self.state_hz)
        t = t[(t >= t0) & (t <= t1)]
        recs = np.zeros(len(t), schema.STATE)
        recs["t_recv"] = t
        recs["flange"][:, :, 3] = 1.0
        return _state_window(recs)

    def _ring(self, cam: str):
        self._declared(cam, "frame_ids_between/frames_by_id")
        if cam not in self.specs:
            raise LookupError(f"no camera {cam!r} in this run; its cameras are "
                              f"{sorted(self.specs)}")
        return self.specs[cam]

    def _times(self, cam: str) -> np.ndarray:
        spec = self._ring(cam)
        return self._timeline(float(self.history[cam]), float(spec.hz))

    def frame_ids_between(self, cam: str, t0: float, t1: float) -> FrameTimes:
        _check_window(t0, t1)
        t = self._times(cam)
        ids = np.arange(1, len(t) + 1, dtype=np.int64)
        keep = (t >= t0) & (t <= t1)
        return FrameTimes(ids[keep], t[keep])

    def frames_by_id(self, cam: str, ids) -> list[Frame]:
        spec = self._ring(cam)
        t = self._times(cam)
        out = []
        for fid in np.asarray(ids, dtype=np.int64).reshape(-1):
            if 1 <= fid <= len(t):
                out.append(Frame(np.zeros((spec.height, spec.width, 3), np.uint8),
                                 float(t[fid - 1]), int(fid)))
        return out


def _check_window(t0: float, t1: float) -> None:
    if t1 < t0:
        raise ValueError(f"empty window: t1 {t1} < t0 {t0}")


def _state_window(recs) -> StateWindow:
    hand = np.asarray(recs["grip_meas"], np.float64).reshape(
        -1, schema.N_HANDS, schema.HAND_DIM)
    return StateWindow(t_recv=np.asarray(recs["t_recv"], np.float64),
                       q=np.asarray(recs["q"], np.float64),
                       dq=np.asarray(recs["dq"], np.float64),
                       flange=np.asarray(recs["flange"], np.float64),
                       hand=hand[:, :, 0].copy())


# --- pure helpers over what the reader returned ------------------------------


def _times_of(window) -> np.ndarray:
    if hasattr(window, "t_recv"):
        return np.asarray(window.t_recv, np.float64)
    return np.asarray([f.t_recv for f in window], np.float64)


def _take(window, idx: np.ndarray):
    """`window` restricted to positions `idx`, as the same kind of thing."""
    if hasattr(window, "_fields"):                  # FrameTimes, StateWindow
        return type(window)._make(np.asarray(f)[idx] for f in window)
    return [window[int(i)] for i in idx]            # list[Frame]


def nearest_to_times(window, times):
    """For each instant in `times`, the item of `window` nearest it.

    `window` is anything the reader returned — `FrameTimes`, `StateWindow`, or
    a `list[Frame]` — and so is the result, one item per instant, in the order
    of `times` (an item can repeat when two instants share a nearest one). Ties
    go to the earlier item. An empty window gives an empty result.

        ft = reader.frame_ids_between("head_left", t - 4.0, t)
        picks = nearest_to_times(ft, t - np.array([3.0, 2.0, 1.0, 0.0]))
        frames = reader.frames_by_id("head_left", picks.ids)
    """
    t = _times_of(window)
    q = np.atleast_1d(np.asarray(times, np.float64))
    if len(t) == 0:
        return _take(window, np.empty(0, np.intp))
    hi = np.clip(np.searchsorted(t, q), 1, len(t) - 1) if len(t) > 1 \
        else np.zeros(len(q), np.intp)
    lo = np.maximum(hi - 1, 0)
    idx = np.where(np.abs(t[hi] - q) < np.abs(t[lo] - q), hi, lo)
    return _take(window, idx)


def subsample_every(window, dt: float):
    """`window` subsampled to one item per `dt` seconds, ascending.

    The grid is anchored at the NEWEST item — the present is always kept, which
    is what a policy stacking "now, now-dt, now-2dt, ..." wants — and runs back
    through the window; each grid instant takes its nearest item, and an item
    two instants share is kept once.
    """
    if dt <= 0:
        raise ValueError(f"dt = {dt}; want > 0")
    t = _times_of(window)
    if len(t) == 0:
        return _take(window, np.empty(0, np.intp))
    n = int(np.floor((t[-1] - t[0]) / dt + 0.5)) + 1
    grid = t[-1] - dt * np.arange(n - 1, -1, -1)
    sel = nearest_to_times(FrameTimes(np.arange(len(t)), t), grid).ids
    return _take(window, np.unique(sel))


__all__ = ["DummyReader", "Frame", "FrameTimes", "Observation", "Reader",
           "State", "StateWindow", "WIRE_IS_BGR", "dummy_observation", "subsample_every",
           "nearest_to_times", "observe", "to_rgb"]

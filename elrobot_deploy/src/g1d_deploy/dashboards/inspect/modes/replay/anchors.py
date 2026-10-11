"""Rebuild the observations the live policy saw, from what the record kept.

P5 builds an observation with the anchor rule (`policy_contract/anchor.py`): every camera
takes the frame whose `t_recv` is NEAREST `t_trigger`, the anchor camera's
choice becomes `t_anchor`, and the joint state is the sample nearest `t_anchor`.
To re-run a policy on the same observations, that choice is reproduced from
`*.idx.bin` and `state.bin`.

**The anchor camera is exact and needs no rule at all.** `chunks.t_anchor` IS
the anchor frame's `t_recv`, written down at the time, so that frame is looked
up by equality and the reconstruction rests on a recorded number rather than
on an emulation.

**The other cameras need one, and "nearest to `t_trigger`" is not it.**
Replaying that rule misses where the nearer frame had arrived on the wire but
P2 had not finished decoding and PUBLISHING it when P5 looked. The record
stores `t_recv` (arrival) and `t_obs_received` (when P5 was done acquiring),
but never the publish instant, so the rule is emulated with one latency:

    candidates = frames with  t_recv + lambda <= t_obs_received
    choice     = the candidate nearest t_trigger

`lambda` is CALIBRATED on the anchor camera, whose answer is known for every
chunk: the value that reproduces the most anchor choices is used for the
others. Where a second frame sits within `_AMBIGUOUS_S` of the visibility edge
the choice is flagged per chunk, so a replay never quietly claims to have seen
exactly what the robot saw when it might not have.

**The state needs no rule either, but it can be STALE**: P5 takes the sample
nearest `t_anchor`, and so does this — the same row. A gap past
`_STALE_STATE_S` means the telemetry stalled across the anchor; the live policy
acted on that stale state too, so it is flagged, not refused.

**Cameras are the record's, views are the policy's.** P9 records each
robot-file camera whole (the G1-D head is one binocular JPEG); a policy names
VIEWS (`head_left`, `head_right`), cut from it by columns. Both eyes of a frame
share its `t_recv`, so the anchor rule runs per SOURCE camera and a view is
its source's row plus a column window (`views_of`).

Used by `run.py` (the replay producer) and `observe.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Grid searched for the publish latency, seconds. Brackets the 5.3-6.7 ms of
#: JPEG decode measured on the G1-D rig.
_LAMBDA_GRID = tuple(np.arange(0.0, 0.01201, 0.0005))

#: A second frame this close to the visibility edge means the emulation could
#: have gone either way.
_AMBIGUOUS_S = 0.0015

#: The state stream runs at ~1 kHz; past this the telemetry stalled.
_STALE_STATE_S = 0.005


@dataclass
class View:
    """One policy-facing camera: which recorded camera, which columns — or,
    for a fisheye view, which square of it."""

    name: str
    source: str
    x0: int = 0
    width: int | None = None        # None = the whole frame
    #: A fisheye view's square, `(x0, y0, side)` on the recorded frame: the
    #: one the live run fitted and cut. None = columns only.
    square: tuple[int, int, int] | None = None


def views_of(manifest: dict, names) -> dict[str, View]:
    """`names` (view names) -> `View`, from the manifest's `camera_views`.

    A record written before views existed names cameras directly, and a name
    that is not a view is taken as a camera of its own.
    """
    table = manifest.get("camera_views") or {}
    out = {}
    for name in names:
        v = table.get(name)
        if v and v.get("cropped"):
            # The square the live run's P2 cut, as its manifest states it.
            out[name] = View(name, str(v.get("source", name)),
                             square=(int(v["x0"]), int(v["y0"]), int(v["width"])))
        elif v:
            out[name] = View(name, str(v.get("source", name)), int(v.get("x0", 0)),
                             int(v["width"]) if v.get("width") else None)
        else:
            out[name] = View(name, name)
    return out


@dataclass
class AnchorSpec:
    """Which rows of which streams make up one observation. No pixels."""

    index: int                              # position in the replay, 0-based
    chunk_id: int
    k_first_action: int
    n_actions: int
    t_trigger: float
    t_anchor: float
    t_obs_received: float
    #: recorded camera name -> row of that camera's index array.
    frames: dict[str, int]
    state_row: int
    primary_exact: bool = True
    ambiguous: tuple[str, ...] = ()
    state_gap_s: float = 0.0


@dataclass
class Report:
    """What the reconstruction had to assume, for the manifest and the log."""

    n: int = 0
    primary: str = ""
    primary_exact: int = 0
    lambda_s: float = 0.0
    lambda_reproduced: int = 0
    ambiguous_chunks: list[int] = field(default_factory=list)
    max_state_gap_s: float = 0.0
    stale_state_chunks: list[int] = field(default_factory=list)
    cameras: tuple[str, ...] = ()

    def as_json(self) -> dict:
        return {
            "rule": "anchor camera: recorded t_anchor; others: nearest t_trigger "
                    "among frames with t_recv + lambda <= t_obs_received",
            "n": self.n, "primary": self.primary,
            "primary_exact": self.primary_exact,
            "lambda_s": float(self.lambda_s),
            "lambda_reproduced_primary": self.lambda_reproduced,
            "ambiguous_chunks": list(self.ambiguous_chunks),
            "max_state_gap_s": float(self.max_state_gap_s),
            "stale_state_s": float(_STALE_STATE_S),
            "stale_state_chunks": list(self.stale_state_chunks),
            "cameras": list(self.cameras),
        }

    def describe(self) -> str:
        return (f"anchors: {self.n} from the record's own replans; "
                f"anchor camera {self.primary} exact {self.primary_exact}/{self.n}; "
                f"lambda {self.lambda_s * 1e3:.1f} ms reproduces "
                f"{self.lambda_reproduced}/{self.n}; "
                f"ambiguous {len(self.ambiguous_chunks)}; "
                f"state gap max {self.max_state_gap_s * 1e3:.2f} ms, "
                f"stale {len(self.stale_state_chunks)}")


def live_chunks(chunks: np.ndarray) -> np.ndarray:
    """The rows of `chunks.bin` that describe a real replan, oldest first.

    P9 appends every chunk P5 publishes; `chunk_id > 0` drops empty slots and
    the sort puts them in replan order.
    """
    if not len(chunks):
        return chunks[:0]
    rows = chunks[chunks["chunk_id"] > 0]
    return rows[np.argsort(rows["chunk_id"], kind="stable")]


def check_complete(chunks: np.ndarray) -> None:
    """Refuse a record whose chunk stream does not start at the first replan.

    A partial stream is not a partial answer: it would be a replay covering the
    tail of a rollout, filed under the whole rollout's name.
    """
    rows = live_chunks(chunks)
    if not len(rows):
        raise ValueError("this recording has no chunks: nothing to re-run.")
    ids = rows["chunk_id"].astype(np.int64)
    if int(ids[0]) != 1 or not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError(
            f"chunks.bin holds ids {int(ids[0])}..{int(ids[-1])} ({len(ids)} "
            f"rows), not 1..N: replans are missing, and a replay of what is "
            f"left would be filed under the whole rollout's name — refusing.")


def _emulate(t_recv: np.ndarray, t_trigger: float, t_obs_received: float,
             lam: float) -> int:
    """The frame P5 would have taken: nearest the trigger, among the visible;
    the nearest of all when nothing was visible yet."""
    visible = np.flatnonzero(t_recv + lam <= t_obs_received)
    pool = visible if len(visible) else np.arange(len(t_recv))
    return int(pool[np.argmin(np.abs(t_recv[pool] - t_trigger))])


def calibrate_lambda(chunks: np.ndarray, t_recv: np.ndarray,
                     grid=_LAMBDA_GRID) -> tuple[float, int]:
    """The publish latency that best explains the ANCHOR camera's choices.
    Ties go to the smallest: the smallest correction that explains the data."""
    best, best_hits = 0.0, -1
    for lam in grid:
        hits = 0
        for c in chunks:
            row = _emulate(t_recv, float(c["t_trigger"]),
                           float(c["t_obs_received"]), float(lam))
            if row >= 0 and t_recv[row] == float(c["t_anchor"]):
                hits += 1
        if hits > best_hits:
            best, best_hits = float(lam), hits
    return best, best_hits


def _ambiguous(t_recv, chosen, t_trigger, t_obs_received, lam) -> bool:
    """Would a nudge in the latency have changed the answer?"""
    for other in (chosen - 1, chosen + 1):
        if not 0 <= other < len(t_recv):
            continue
        if abs(t_recv[other] - t_trigger) >= abs(t_recv[chosen] - t_trigger):
            continue
        if abs(t_recv[other] + lam - t_obs_received) < _AMBIGUOUS_S:
            return True
    return False


def reconstruct(record, sources: set[str], primary: str, *,
                lam: float | None = None) -> tuple[list[AnchorSpec], Report]:
    """One `AnchorSpec` per replan in the recording.

    `record` is a `loader.Rollout`; `sources` the RECORDED cameras the policy's
    views are cut from; `primary` the recorded camera the anchor view is cut
    from.
    """
    chunks = live_chunks(record.chunks)
    check_complete(chunks)
    missing = sorted((sources | {primary}) - set(record.cameras))
    if missing:
        raise ValueError(f"the policy observes {missing} but the record has "
                         f"{sorted(record.cameras)}")

    t_recv = {name: np.asarray(record.cameras[name].index["t_recv"], np.float64)
              for name in sorted(sources | {primary})}
    t_state = np.asarray(record.state["t_recv"], dtype=np.float64)

    if lam is None:
        lam, reproduced = calibrate_lambda(chunks, t_recv[primary])
    else:
        _, reproduced = calibrate_lambda(chunks, t_recv[primary], grid=(lam,))

    report = Report(n=len(chunks), primary=primary, lambda_s=lam,
                    lambda_reproduced=reproduced, cameras=tuple(t_recv))
    specs: list[AnchorSpec] = []
    for i, c in enumerate(chunks):
        t_trigger = float(c["t_trigger"])
        t_anchor = float(c["t_anchor"])
        t_obs = float(c["t_obs_received"])

        frames, ambiguous = {}, []
        for name, times in t_recv.items():
            if name == primary:
                hit = np.flatnonzero(times == t_anchor)
                frames[name] = (int(hit[0]) if len(hit) == 1
                                else int(np.argmin(np.abs(times - t_anchor))))
            else:
                row = _emulate(times, t_trigger, t_obs, lam)
                frames[name] = row
                if _ambiguous(times, row, t_trigger, t_obs, lam):
                    ambiguous.append(name)
        primary_exact = bool(np.sum(t_recv[primary] == t_anchor) == 1)

        state_row = int(np.argmin(np.abs(t_state - t_anchor)))
        gap = abs(float(t_state[state_row]) - t_anchor)
        report.max_state_gap_s = max(report.max_state_gap_s, gap)
        if gap > _STALE_STATE_S:
            report.stale_state_chunks.append(int(c["chunk_id"]))
        if ambiguous:
            report.ambiguous_chunks.append(int(c["chunk_id"]))
        specs.append(AnchorSpec(
            index=i, chunk_id=int(c["chunk_id"]),
            k_first_action=int(c["k_first_action"]),
            n_actions=int(c["n_actions"]), t_trigger=t_trigger,
            t_anchor=t_anchor, t_obs_received=t_obs, frames=frames,
            state_row=state_row, primary_exact=primary_exact,
            ambiguous=tuple(ambiguous), state_gap_s=gap))

    report.primary_exact = sum(1 for s in specs if s.primary_exact)
    return specs, report


__all__ = ["AnchorSpec", "Report", "View", "calibrate_lambda", "check_complete",
           "live_chunks", "reconstruct", "views_of"]

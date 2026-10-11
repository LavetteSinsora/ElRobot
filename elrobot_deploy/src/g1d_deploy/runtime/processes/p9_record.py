"""P9 — the recorder.

Spawned per rollout; reads every ring, writes one record directory
(`rollout_record_utils/` reads it back). What a record holds and how it is
inspected: `docs/deploying_a_policy/rollout_inspection.md`.

Not a control loop. Nothing in the control path reads what it writes,
nothing waits on it, and a run behaves identically whether or not it exists.

**Non-interference is structural, not a promise.** The arena's primitives were
chosen so no reader can delay a writer — a seqlock writer is wait-free, and a
camera ring publishes with a single store. This process holds no lock
any writer wants, so it *cannot* block P1, P2 or P7. What is left is placement
(an E core, away from the camera cluster, at normal priority) and buffering (a
disk stall drops records; it never blocks a read).

**It cannot miss a ring write, and that is geometry rather than diligence.** A
target slot survives `N_body` ticks — 128 slots at 30 Hz is 4.3 seconds — and a
chunk slot about five. Polling and deduping on `(k, seq)` is therefore complete
unless this process stalls for seconds. No cooperation from P5 or P6, no
side-channel, no change to any hot loop.

**Images are passthrough, not encoding.** The board's image server already sends JPEG and
P2 keeps those bytes beside the pixels it decoded, so this appends them
unchanged: no CPU spent, no second lossy generation, and what lands on disk is
exactly what the robot sent.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import time

import numpy as np

from g1d_deploy.runtime import cpu_pinning as affinity
from g1d_deploy.runtime.shared_memory import Arena, build, open_all, schema
from g1d_deploy.runtime.timing import Pacer, now
from g1d_deploy.config_loading.resolved import DeployConfig, Hardware
from g1d_deploy.config_loading.derive import derive, servo_gains
from g1d_deploy.robot.kinematics import WAIST_JOINT_NAMES
from g1d_deploy.config_loading.rollout_settings import RolloutSettings, latched

#: How often the rings are swept. Far faster than needed for completeness (a
#: slot lives seconds) — the rate is chosen for `t_observed` precision instead,
#: which bounds when a target became AVAILABLE as opposed to when its deadline
#: says it should execute. That gap is what stall analysis is about.
POLL_HZ = 500.0

#: Bytes buffered per stream before a write. Large enough that writes are
#: sequential and rare, small enough that a crash loses a fraction of a second.
FLUSH_BYTES = 1 << 20

#: How many modified paths the manifest names before it stops listing them. The
#: count is always exact; only the list is capped.
_MAX_DIRTY_FILES = 50


def _jsonable(value):
    """A dataclass tree with numpy in it -> something `json.dumps` accepts.

    Written out rather than left to `default=str`, which turns an array into the
    string `"[0. 0. ...]"` — readable by a human, parseable by nobody. A
    `home_q` that cannot be loaded back as numbers is not a record of `home_q`.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)                       # Path, enum, anything exotic


def _cpu_topology() -> dict:
    """This machine's P/E split and physical-core map, or `{}` off Linux.

    Recorded beside the pin table because a table alone cannot be checked: cpu
    14 means nothing without knowing whether it is a performance thread and
    whose sibling it is.
    """
    from g1d_deploy.runtime import cpu_pinning as affinity
    try:
        return affinity.topology()
    except Exception:            # noqa: BLE001 — a record must never fail to write
        return {}


def _git_revision() -> dict:
    """Which commit produced this rollout, and whether the tree was clean.

    The config dump says what was *configured*; this says what the configured
    code actually was. Both are needed and neither substitutes for the other —
    a run from a dirty tree is not reproducible from its SHA alone, so the
    dirty file list is recorded rather than just a flag.

    Never raises. A record written outside a git checkout, or on a machine with
    no `git`, is still a record; losing the provenance is much better than
    losing the manifest, which is written from the same `finally` that flushes
    every stream.
    """
    root = pathlib.Path(__file__).resolve().parents[2]   # any directory inside the checkout
    try:
        def git(*args):
            return subprocess.run(("git", "-C", str(root), *args),
                                  capture_output=True, text=True, timeout=10)

        head = git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"sha": None, "error": head.stderr.strip()[:200]}
        status = git("status", "--porcelain")
        changed = [line[3:] for line in status.stdout.splitlines()]
        return {"sha": head.stdout.strip(),
                "dirty": bool(changed),
                # Capped: a tree with a thousand untracked files should not turn
                # the manifest into a directory listing.
                "dirty_files": sorted(changed)[:_MAX_DIRTY_FILES],
                "dirty_file_count": len(changed)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"sha": None, "error": f"{type(exc).__name__}: {exc}"[:200]}


def _observed(dtype: np.dtype) -> np.dtype:
    """An arena record plus the instant this process first saw it.

    `deadline` says when a target should be *executed*; `t_observed` bounds when
    it became *available*. Recording only the former would make it impossible to
    tell a target that arrived late from one that was always going to be late.
    """
    return np.dtype(dtype.descr + [("t_observed", "<f8")])


#: One stream, because there is one target ring (`schema.TARGET`). The
#: sweep below is generic over `dtype.names`, so the merged row's `hand` column
#: is recorded without this file knowing it exists.
TARGET_REC = _observed(np.dtype(schema.TARGET.descr + [("seq", "<u8")]))

#: One row per session transition: the whole control block, plus when this
#: process saw it. The whole block rather than the session word alone because
#: the other three are on the same cache line — `k_exec` says how far the tick
#: line had got at each edge, `newest_chunk_id` how many chunks P5 had
#: published, and `warm` is P5's bit sampled AT the edge rather than at
#: `close()`, which is the only place the manifest's `warm` can report from.
SESSION_REC = _observed(schema.CONTROL_WORDS)

FRAME_IDX = np.dtype([
    ("frame_id", "<u8"), ("t_recv", "<f8"), ("t_observed", "<f8"),
    ("offset", "<u8"), ("nbytes", "<u4"),
])



def _model_ref(urdf: str) -> str:
    """`hw.urdf` (resolved, absolute) as the package-relative path the robot
    file wrote when it is the package's own model, else as it is."""
    from g1d_deploy.config_loading.resolved import MODEL_ROOT
    try:
        return str(pathlib.Path(urdf).relative_to(MODEL_ROOT))
    except ValueError:
        return str(urdf)

class Stream:
    """An append-only file of fixed-size records, buffered.

    Records go out as raw bytes because the record's dtype IS the arena's dtype
    — no translation layer, therefore no schema drift, therefore a record file
    and the running system cannot come to disagree about what a body target is.
    """

    def __init__(self, path: pathlib.Path, dtype: np.dtype):
        self.path = path
        self.dtype = np.dtype(dtype)
        self._fh = open(path, "wb", buffering=0)
        self._buf = bytearray()
        self.count = 0

    def append(self, record) -> None:
        self._buf += np.asarray(record, dtype=self.dtype).tobytes()
        self.count += 1
        if len(self._buf) >= FLUSH_BYTES:
            self.flush()

    def append_bytes(self, payload: bytes) -> int:
        """For the JPEG stream, whose records are not fixed-size. Returns the
        byte offset the payload landed at, which is what the index stores."""
        offset = self._fh.tell() + len(self._buf)
        self._buf += payload
        if len(self._buf) >= FLUSH_BYTES:
            self.flush()
        return offset

    def flush(self) -> None:
        if self._buf:
            self._fh.write(self._buf)
            self._buf.clear()

    def close(self) -> None:
        self.flush()
        self._fh.close()


class Recorder:
    """One rollout's worth of streams, and the sweep that fills them."""

    def __init__(self, out_dir: pathlib.Path, cfg: DeployConfig, rings: dict,
                 *, t_floor: float = 0.0, code: dict | None = None,
                 provenance: dict | None = None, rollout: dict | None = None):
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.rings = rings

        #: An instant known to precede this rollout, used to tell a chunk THIS
        #: rollout published from one still sitting in the ring from the last
        #: one. See `_sweep_chunks`; `run` samples it before ACTIVE.
        self._chunk_floor = float(t_floor)
        #: The tree as it was when this PROCESS started, not as it is now. See
        #: `write_manifest`.
        self._code = code
        #: The launch's provenance (`rollout_record_utils.manifest.build` + INIT's
        #: lens decisions): the robot, machine and deploy files, policy entry,
        #: checkpoint, resolution sources, the infra's and the policy's own
        #: revisions. Written verbatim into the manifest.
        self._provenance = dict(provenance or {})
        #: This rollout's entry in the manifest's `rollouts` list.
        self._rollout = dict(rollout or {})

        self.state = Stream(self.dir / "state.bin", schema.STATE)
        self.chunks = Stream(self.dir / "chunks.bin",
                             _observed(rings["chunks"]._rec.dtype))
        self.targets = Stream(self.dir / "targets.bin", TARGET_REC)
        self.session = Stream(self.dir / "session.bin", SESSION_REC)
        #: The operator's own stream. The target rings already carry
        #: `intervene`, so WHO was driving each tick is recoverable without this
        #: — but only as a label. This is the raw input behind it: the controller
        #: poses, the triggers, the thumbsticks and both clocks, which is what
        #: makes a session replayable offline and a mapping complaint diagnosable
        #: rather than a matter of opinion. Empty on a run without --intervene.
        self.intervene = Stream(self.dir / "intervene.bin", schema.INTERVENE)
        #: P7's 500 Hz publish log: what the firmware was actually
        #: sent, as opposed to the 30 Hz targets it was walking between.
        self.command = Stream(self.dir / "command.bin", schema.COMMAND)
        self.frames = {
            name: (Stream(self.dir / f"{name}.jpg.bin", np.dtype("u1")),
                   Stream(self.dir / f"{name}.idx.bin", FRAME_IDX))
            for name in rings["camera_sources"]
        }

        # Dedupe state. `seq` is what separates "re-solved" from "silently kept
        # the old value": in async P6 re-solves ticks after a new chunk lands,
        # and an abandoned re-solve leaves the previous pass's value in the slot
        # — valid-looking, and NOT a gap.
        self._seen_targets: dict[int, int] = {}
        self._seen_chunk: dict[int, int] = {}
        self._seen_state = 0
        self._seen_intervene = 0
        self._seen_command = 0
        # `None`, not INIT: the first sweep must emit whatever state it finds
        # rather than treating the first phase as "no change". A run that never
        # transitions should still say which phase it spent itself in, and an
        # empty file reads exactly like a recorder that never ran.
        self._seen_session: int | None = None
        self._seen_frame: dict[str, int] = {n: 0 for n in rings["camera_sources"]}
        self.dropped_frames: dict[str, int] = {n: 0 for n in rings["camera_sources"]}

    # --- the sweep ---------------------------------------------------------

    def sweep(self) -> None:
        t = now()
        self._sweep_session(t)
        self._sweep_state()
        self._sweep_command()
        self._sweep_targets(self.rings["targets"], self.targets,
                            self._seen_targets, t)
        self._sweep_intervene()
        self._sweep_chunks(t)
        self._sweep_frames(t)

    def _sweep_session(self, t) -> None:
        """One row each time P0 moves the session word.

        **Nothing new is stored in the arena for this.** `session` is already an
        atomic aligned word with exactly one writer (P0's `_set`), and this
        process already reads it 500 times a second — it is `run()`'s loop
        condition. All that was missing is that nobody remembered the previous
        value. So this costs one integer compare per sweep on the unchanged
        path, and no hot loop anywhere is touched.

        **It is a SAMPLED signal, unlike every other stream here.** The rings are
        complete by geometry — a slot lives seconds, so polling cannot miss one.
        A session word has no ring behind it: a phase shorter than the
        2 ms poll period is simply invisible, and every `t_observed` is the edge
        plus up to one period. In practice the phases are seconds long (HOME
        ramps, WARMUP blocks on `warm`, READY blocks on the operator), but the
        file says when P9 SAW a phase, not when P0 entered it.

        **The first row is ACTIVE, never CHECK.** `run()` waits out CHECK, HOME,
        WARMUP and READY without sweeping at all, so this method's first call
        finds `session` already at ACTIVE and emits that as row one. Everything
        before it — including whether homing behaved — is P0's log alone to
        witness now; this file starts at the rollout, not at the robot waking
        up.

        Flushed on every append. Ten rows a run at 40 bytes will not reach the
        1 MB buffer, and the run where the phase log matters most is the one
        that ends in a `SIGKILL` with `close()` never reached.
        """
        control = self.rings["control"]
        session = control.session
        if session == self._seen_session:
            return
        out = np.zeros((), dtype=self.session.dtype)
        out["session"] = session
        out["k_exec"] = control.k_exec
        out["newest_chunk_id"] = control.newest_chunk_id
        out["warm"] = control.warm
        out["rollout_generation"] = control.rollout_generation
        out["t_observed"] = t
        self.session.append(out)
        self.session.flush()
        self._seen_session = session

    def _sweep_state(self) -> None:
        """The state ring, by absolute sample index.

        `index` is monotonic and never reused, so "everything newer than the
        last one recorded" needs no timestamps and cannot double-count a slot
        the writer has since overwritten.
        """
        ring = self.rings["state"]
        idx = ring._rec["index"]
        fresh = np.flatnonzero(idx > self._seen_state)
        if fresh.size == 0:
            return
        for i in fresh[np.argsort(idx[fresh])]:
            rec = ring._lock.snapshot(int(i), schema.STATE)
            if int(rec["index"]) <= self._seen_state:
                continue                      # overwritten mid-sweep
            self.state.append(rec)
            self._seen_state = int(rec["index"])

    def _sweep_command(self) -> None:
        """P7's publish log, by absolute index — the state sweep's rule."""
        ring = self.rings.get("command")
        if ring is None:
            return
        idx = ring._rec["index"]
        fresh = np.flatnonzero(idx > self._seen_command)
        if fresh.size == 0:
            return
        for i in fresh[np.argsort(idx[fresh])]:
            rec = ring._lock.snapshot(int(i), schema.COMMAND)
            if int(rec["index"]) <= self._seen_command:
                continue                      # overwritten mid-sweep
            self.command.append(rec)
            self._seen_command = int(rec["index"])

    def _sweep_intervene(self) -> None:
        """The intervention ring, by absolute row index.

        The state ring's rule exactly: `index` is monotonic and never reused, so
        "everything newer than the last one recorded" needs no timestamps and
        cannot double-count a slot the writer has since overwritten.

        Costs one `flatnonzero` over an 8 kB column per sweep on a run that never
        intervenes, where the answer is always empty — the same shape and the
        same cost as the state sweep, which runs on every rollout.
        """
        ring = self.rings["intervene"]
        idx = ring._rec["index"]
        fresh = np.flatnonzero(idx > self._seen_intervene)
        if fresh.size == 0:
            return
        for i in fresh[np.argsort(idx[fresh])]:
            rec = ring._lock.snapshot(int(i), schema.INTERVENE)
            if int(rec["index"]) <= self._seen_intervene:
                continue                      # overwritten mid-sweep
            self.intervene.append(rec)
            self._seen_intervene = int(rec["index"])

    def _sweep_targets(self, ring, stream, seen, t) -> None:
        """Every slot whose `(k, seq)` this has not already recorded.

        Copied **by field name off a full-record snapshot**, exactly as
        `_sweep_chunks` does, rather than through `ring.get()`. The difference is
        not stylistic. `get()` returns a `TargetEntry`, which carries `k`,
        `deadline` and the payload and nothing else — deliberately, so that P7's
        500 Hz `next_after` never copies fields it does not read. Naming fields
        here would mean the recorder could only ever store those, and a field
        added to the ring later would be silently zero-filled in every record:
        `q` would look ordinary, a residual would read as a perfect 0.0, and
        nothing anywhere would report a problem.

        Iterating `snap.dtype.names` makes that class of bug unrepresentable: the
        record's dtype IS the ring's dtype plus `seq` and `t_observed`, so every
        field the arena grows is recorded the day it is added, without this
        function being touched.
        """
        ks = ring._rec["k"]
        seqs = ring._seq
        for i in range(ring.n):
            seq = int(seqs[i])
            if seq == 0 or seq & 1:
                continue                      # never written, or mid-write
            k = int(ks[i])
            if k == schema.K_UNSET or seen.get(k) == seq:
                continue
            snap = ring._lock.snapshot(i, ring.DTYPE)
            if int(snap["k"]) != k:
                continue                      # rewritten mid-read
            out = np.zeros((), dtype=stream.dtype)
            for name in snap.dtype.names:
                out[name] = snap[name]
            out["seq"] = seq
            out["t_observed"] = t
            stream.append(out)
            seen[k] = seq

    def _sweep_chunks(self, t) -> None:
        """Every chunk slot this rollout has published into.

        **The floor is what keeps the PREVIOUS rollout out of this record.** A
        session serves many rollouts on one warmup and the chunk ring is
        three slots deep, so at the ACTIVE edge those three slots still hold the
        last three chunks of the rollout before this one. `Recorder` is built
        fresh per rollout, so `_seen_chunk` is empty and the first sweep would
        otherwise write all three out as if they were ours: the same chunk ids
        would appear in two records, at two different instants.

        `p5_policy.run` calls those leftovers "inert rather than stale", and for
        P6 they are: it asks which chunk covers a tick, and they span ticks this
        rollout never reaches. This process asks a different question — what is
        in the ring — so the same slots are not inert here at all.

        **The id cannot be the discriminator, and that is why this is a time.**
        Chunk ids restart at 1 every rollout, so the previous rollout's chunk 7
        and this one's occupy the same slot with the same id; keying on
        `(slot, id)` would have skipped the REAL chunk 7 and turned a duplicated
        row into a missing one. `t_chunk_written` comes off a monotonic clock
        that does not restart, so it separates them absolutely.

        Reading it costs nothing extra: a field of a structured array over
        shared memory is a strided view, so this is the same two scalar loads
        per slot the id check already does, and the ~11 KB payload is still
        copied only for a slot that is actually being recorded.
        """
        ring = self.rings["chunks"]
        ids = ring._rec["chunk_id"]
        written = ring._rec["t_chunk_written"]
        for i in range(ring.n):
            cid = int(ids[i])
            if cid == 0 or self._seen_chunk.get(i) == cid:
                continue
            # Deliberately BEFORE `_seen_chunk` is set, so a slot skipped here
            # is reconsidered on the next sweep rather than suppressed for the
            # rest of the run. That is also what makes a torn read of this one
            # field self-correcting: the cost is one 2 ms poll, not a lost row.
            if float(written[i]) <= self._chunk_floor:
                continue
            rec = ring._lock.snapshot(i, ring._rec.dtype)
            if int(rec["chunk_id"]) != cid:
                continue                      # rewritten mid-read
            out = np.zeros((), dtype=self.chunks.dtype)
            for name in rec.dtype.names:
                out[name] = rec[name]
            out["t_observed"] = t
            self.chunks.append(out)
            self._seen_chunk[i] = cid

    def _sweep_frames(self, t) -> None:
        """The compressed source of every frame published since the last sweep.

        One stream per ROBOT-FILE camera (`rings["camera_sources"]`), under its
        own name: a side_by_side head is recorded as the whole binocular JPEG,
        verbatim, read out of the view that carries it
        (`resolved.CameraView.carries_jpeg`). Inspection splits it for display.

        The ring keeps N frames, so a sweep that ran late catches up on every
        frame still resident (`CameraRing.jpegs_since`) instead of taking only
        the newest. A gap in `frame_id` is a frame lapped before this process
        saw it, torn, or not captured — and the camera overwriting rather than
        blocking is CORRECT: it must never wait on a recorder. Counting the gap
        is what keeps that loss visible instead of silent.
        """
        for name, region in self.rings["camera_sources"].items():
            last = self._seen_frame[name]
            got, newest = region.jpegs_since(last)
            if newest == last:
                continue
            blob, index = self.frames[name]
            for t_recv, fid, payload in got:
                if fid > last + 1:
                    self.dropped_frames[name] += fid - last - 1
                offset = blob.append_bytes(payload)
                idx = np.zeros((), dtype=FRAME_IDX)
                idx["frame_id"] = fid
                idx["t_recv"] = t_recv
                idx["t_observed"] = t
                idx["offset"] = offset
                idx["nbytes"] = len(payload)
                index.append(idx)
                last = fid
            # Whatever is left between the last recorded frame and the newest
            # one looked at was not recordable (see above).
            self.dropped_frames[name] += newest - last
            self._seen_frame[name] = newest

    # --- lifecycle ---------------------------------------------------------

    def write_manifest(self, hw: Hardware, extra: dict | None = None) -> None:
        """Everything needed to read these files without this code.

        `dtype.descr` for each stream is the point: an old record stays readable
        by its own recorded descriptor rather than by whatever the constant in
        `schema.py` says on the day someone opens it.
        """
        streams = {
            "state": self.state, "chunks": self.chunks,
            "targets": self.targets,
            # A stream rather than a list in the manifest: `reader.load`
            # walks `streams` generically and rebuilds each dtype from its own
            # `descr`, so this becomes readable by every existing reader with no
            # reader change at all. A bespoke JSON list would need bespoke
            # parsing in each of them.
            "session": self.session,
            # Same reasoning: a stream, so `reader.load` picks it up
            # generically and every existing reader gains it with no change.
            "intervene": self.intervene,
            "command": self.command,
        }
        kp, kd = servo_gains(self.cfg, hw)
        manifest = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "streams": {name: {"file": s.path.name, "descr": s.dtype.descr,
                               "records": s.count}
                        for name, s in streams.items()},
            "cameras": {
                name: {"blob": f"{name}.jpg.bin", "index": f"{name}.idx.bin",
                       "index_descr": FRAME_IDX.descr,
                       "frames": self.frames[name][1].count,
                       "dropped": self.dropped_frames[name]}
                for name in self.frames},
            # What the POLICY called each camera: a side_by_side
            # camera above is recorded whole, and these are the views the run
            # cut from it — which columns, which eye, how deep each ring was.
            # `cameras` names what was recorded; `observed_cameras`,
            # `primary_camera` and `camera_history_s` in `deploy_config` name
            # views.
            "camera_views": {
                v.name: {"source": v.source, "half": v.half, "x0": v.x0,
                         "y0": v.y0, "cropped": v.cropped,
                         "height": v.height, "width": v.width,
                         "slots": v.slots, "history_s": v.history_s}
                for v in self.cfg.views.values()},
            # EVERY field, not a chosen handful. The handful was chosen once and
            # then a run happened whose `home_hand` could not be recovered: the
            # record showed the idle gripper commanded at 5.4 rad while the
            # checked-in config said 0.0, and nothing in the file could say which
            # one the robot had actually been given. A record that cannot answer
            # "what was this run configured with" is a record you cannot compare
            # against another run.
            #
            # `P`, `H` and `dt` are derived properties rather than fields, so
            # they are added explicitly — `rollout_record_utils/reader.py` reads them by
            # those names, and a dump of fields alone would silently drop them.
            "deploy_config": dict(_jsonable(self.cfg),
                                  P=self.cfg.P, H=self.cfg.H, dt=self.cfg.dt),
            # WHEN this was sampled matters as much as what it says, and it is
            # sampled at PROCESS START — when the code now running was imported.
            #
            # Sampling in `close()` would report HEAD at the END of the rollout,
            # which is a different claim and can be a false one: a checkout
            # that changes between a run starting and a rollout closing would
            # attribute the record to code it never imported. A record that
            # misidentifies its own code is worse than one that omits it.
            "code": self._code if self._code is not None else _git_revision(),
            # Whether P5 declared itself warm before the run started. A rollout
            # whose first chunk cost 30 s and whose `warm` is false has already
            # told you which of those two facts caused the other.
            #
            # Sampled HERE, in `close()`, so strictly it is warm-at-the-end. In
            # a normal run that is the same claim — P0 gates READY on the bit
            # and it never clears — but in an aborted one it is not, and
            # `session.bin` carries the value at each phase edge. Left as it is
            # so that every reader of every existing record keeps working.
            "warm": bool(self.rings["control"].warm),
            "cpu": {k: (v if isinstance(v, list) else [v])
                    for k, v in hw.cpu.items()},
            "cpu_reserved": list(hw.cpu_reserved),
            # WHERE the table above came from. "file" means a human wrote it in
            # the machine file and `validate_pinning` confirmed the machine matches;
            # "derived" means it was computed from this machine's own topology at
            # startup. Without this the two are indistinguishable in a record,
            # and they answer different questions when a rollout is compared
            # against another machine's: a stated table is an intention, a
            # derived one is a description.
            "cpu_source": hw.cpu_source,
            "cpu_notes": list(hw.cpu_notes),
            # The topology the table was chosen against, so a record from one
            # deploy box can be read on another without guessing which CPU it
            # ran on. Empty off Linux, where there is nothing to read.
            "cpu_topology": _cpu_topology(),
            # Which columns of the state stream's `q` the body targets address.
            # Without this the two streams cannot be compared at all: state `q`
            # is every joint on the robot, a body target `q` is 14 arm then 3
            # waist, and nothing else in the record says which is which.
            "joints": {"arm": list(hw.joints_arm), "waist": list(hw.joints_waist)},
            # The joint INDICES above say which columns; these say what the
            # columns mean. `joints.waist` names lowstate 12/13/14 and stops
            # there, which is how a record comes to be read against the
            # wrong axis: the humanoid enum calls index 13 a roll and on the
            # G1-D it is the bow. A reader that takes the convention from the
            # record cannot inherit that mistake, and a reader that wants to
            # rebuild the geometry needs the model file this run actually
            # loaded — not whichever one the robot file points at today.
            # The gains P7 wrote at connect: the robot file's with the run's
            # `servo.arm_gain_scale` applied (`derive.servo_gains`), per lowcmd
            # motor index. A gain A/B is unreadable without them.
            "gains": {"kp": kp.tolist(), "kd": kd.tolist()},
            # Which socket each camera was read from.
            "camera_ports": {k: int(v) for k, v in hw.camera_ports.items()},
            # The torque model the servo's feedforward used (the robot file
            # [servo_model]); null on a robot without one.
            "servo_model": _jsonable(hw.servo_model),
            # The model as the robot file names it — package-relative for the
            # shipped one, so a record carries no path of THIS machine that a
            # reader would need (`manifest.robot_block`).
            "model": {"urdf": _model_ref(hw.urdf),
                      "waist_joint_names": list(WAIST_JOINT_NAMES)},
        }
        # What this launch was resolved FROM — the files themselves,
        # not just their names, since a profile can change after the run —
        # and which code, the infra's and the policy's, produced it
        # (`rollout_record_utils/manifest.py`). `code` above stays the
        # g1d_deploy tree's revision for every reader that already reads it.
        prov = self._provenance
        for key in ("infra", "robot", "machine", "deploy", "policy",
                    "checkpoint", "resolution", "crops"):
            if key in prov:
                manifest[key] = _jsonable(prov[key])
        # One entry per rollout this record covers: today exactly this one, in
        # its own directory. ALWAYS the settings it ran with and its prompt
        # — `run` passes the active block it latched at the
        # ACTIVE edge; `self.cfg`, which `deploy_config` above also dumps, is
        # already that block applied, so the fallback says the same thing.
        manifest["rollouts"] = [{
            "index": None, "dir": self.dir.name, "prompt": self.cfg.prompt,
            "generation": None,
            "active_config": RolloutSettings.of(self.cfg).as_dict(),
            **_jsonable(self._rollout)}]
        if extra:
            manifest.update(extra)
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                           default=str))

    def close(self, hw: Hardware, extra: dict | None = None) -> None:
        for s in (self.state, self.chunks, self.targets, self.session,
                  self.intervene, self.command):
            s.close()
        for blob, index in self.frames.values():
            blob.close()
            index.close()
        self.write_manifest(hw, extra)


def rollout_dir(base: pathlib.Path) -> pathlib.Path:
    """A directory for one rollout, named by when that rollout started.

    One stamped directory is one rollout: a launch serves many rollouts, and
    each gets its own `rollout_record/<stamp>/`, which is all
    `dashboards/inspect/discover.py` and `rollout_record_utils/reader.py` need
    to know.

    The suffix is for the case the clock cannot separate: two rollouts inside
    one second. A reset cannot physically produce that — the homing ramp alone
    is `home_min_s` — but two rollouts silently sharing a directory would
    interleave two records into one unreadable set of files, which is worth two
    lines to make impossible rather than merely unlikely.
    """
    base = pathlib.Path(base)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out, n = base / stamp, 1
    while out.exists():
        n += 1
        out = base / f"{stamp}-{n}"
    return out


def run(shm_name: str, cfg: DeployConfig, hw: Hardware, *,
        out_dir: str | pathlib.Path, provenance: dict | None = None,
        **kwargs) -> None:
    """Attach read-only, then one record per ACTIVE period.

    `out_dir` is the BASE — `rollout_record/` — and this stamps a directory
    inside it per rollout. A session now serves many rollouts on one warmup,
    and one record spanning all of them would be a file whose ticks run
    0…50000 and then start again at −90, which nothing that reads these records
    expects.

    Nothing is written before ACTIVE. CHECK, HOME, WARMUP and READY are setup —
    READY in particular waits on a human and can run for minutes — and none of
    it is the rollout: on a typical run it was roughly as much wrist-camera
    JPEG as the run itself, for a phase nobody reads back (homing is trusted,
    not re-checked per record). Waiting costs nothing here: this process is
    already polling `control.session` at `POLL_HZ` for the sweep loop below, so
    catching the ACTIVE edge is the same loop with writes turned off.

    A run that never reaches ACTIVE — Ctrl-C at the READY prompt is the common
    case — leaves nothing on disk: no directory, no manifest. There is nothing
    in that run worth a rollout record.

    **A record is exactly the ACTIVE period.** It ends the moment the operator
    presses Reset or Exit, because that is the moment the session word leaves
    ACTIVE. Neither ramp that follows is in it: what P0 drives afterwards — home
    on a reset, `exit_q` on an exit — is the robot being put away, not the
    rollout, and it is the same motion every time. P6 leaves on the same word
    (`p6_target.run`), so the recording and the thing being recorded stop
    together rather than one trailing the other.

    The last row of `session.bin` is therefore the phase the rollout ended INTO
    — HOME for a reset, EXITING for an exit — which is a record of why it ended
    rather than a record of the ramp. Without it, the two endings are
    indistinguishable: a file that just stops.

    This process is spawned ONCE and outlives every rollout. It is not restarted
    per record, because a recorder that is killed does not write its manifest:
    `stop` is SIGTERM and runs no `finally` (`_Children.wait`), so closing each
    `Recorder` from inside the process that owns it is the only arrangement in
    which every rollout gets a readable record.

    No `SCHED_FIFO`, deliberately: this is the one process in the system where
    being late is free, and it should yield to every process where it is not.
    """
    layout = build(cfg, derive(cfg, hw))
    arena = Arena.attach(shm_name, layout)
    try:
        rings = open_all(arena, cfg)
        affinity.pin(hw, "p9_record")

        control = rings["control"]
        pacer = Pacer(POLL_HZ)

        # Sampled ONCE, here, because this is the instant the code that will
        # produce every rollout of this session was imported. See
        # `Recorder.write_manifest` for why the end of a rollout is the wrong
        # place to ask.
        code = _git_revision()
        # Rollouts of THIS session, counted from 1 — the manifest's `rollouts`
        # index, so records of one launch can be put back in order.
        n_rollout = 0

        while True:
            # An instant that provably precedes this rollout, taken on every
            # pre-ACTIVE poll rather than once at the edge.
            #
            # **The sampling point is the whole correctness argument.** Nothing
            # publishes a chunk before ACTIVE — P5 is warming, not solving — so
            # a time read here cannot be later than this rollout's first chunk,
            # however the two processes interleave at the edge. Reading it after
            # ACTIVE instead would be a race: P5 wakes on the same word and
            # could publish chunk 1 first, which this would then discard.
            t_floor = now()
            while control.session < schema.ACTIVE:
                t_floor = now()
                pacer.wait()
            if control.session != schema.ACTIVE:
                # Straight past ACTIVE to EXITING/STOPPED: abandoned at READY
                # and never started. Nothing to record, this time or ever.
                return

            n_rollout += 1
            # The rollout's own settings and prompt, latched at the
            # edge like every other process: the record's `deploy_config` is
            # then what actually ran — its `P` and `dt` are what `record_read`
            # reads the tick line with — and not the launch defaults.
            rollout_cfg, prompt, generation = latched(control, cfg)
            rollout_cfg = dataclasses.replace(rollout_cfg, prompt=prompt)
            recorder = Recorder(
                rollout_dir(out_dir), rollout_cfg, rings, t_floor=t_floor,
                code=code, provenance=provenance,
                rollout={"index": n_rollout, "generation": generation,
                         "prompt": prompt,
                         "active_config":
                             RolloutSettings.of(rollout_cfg).as_dict(),
                         "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
            try:
                while control.session == schema.ACTIVE:
                    recorder.sweep()
                    pacer.wait()
                # One last sweep for what landed in the ~2 ms between the final
                # pass and this process noticing the word. Not a grace period:
                # P6 leaves ACTIVE on the same word (`p6_target.run`), so there
                # is nothing still being solved to wait for.
                recorder.sweep()
            finally:
                recorder.close(hw)

            if control.session >= schema.STOPPED:
                return
    finally:
        arena.close()

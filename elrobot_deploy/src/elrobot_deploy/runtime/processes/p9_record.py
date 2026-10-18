"""P9: the recorder. Every ring in, one record directory per rollout out.

Ported from g1d's P9 (`legacy_g1d/src/elrobot_deploy/runtime/processes/p9_record.py`), for one
arm on one bus, and with the JPEG encoding g1d did not need (its board sent JPEG; our cameras
are this Mac's, and P2 publishes pixels). `elrobot_deploy.record` reads a record back, also
while it is being written (the Astra harness tails it).

Started by P0 at INIT when the run records (`run --record`, every `astra` session); runs until
STOPPED. **A record is exactly one ACTIVE period** (Start to Reset / Exit / a fault): the ramps
around it are the arm being put away, the same motion every time. Nothing is written before the
first ACTIVE, and a run that never starts leaves nothing on disk.

    <dir>/<stamp>/                  (or the name P0 gives, for an Astra session)
      manifest.json                 every stream's dtype and count, the cameras, the profile
                                    resolved, the code's revision; rewritten at the end
      state.bin                     every bus read (`schema.STATE`, 250 Hz)
      targets.bin                   every target written (`schema.TARGET` + seq + t_observed)
      intervene.bin                 every operator row (`schema.INTERVENE`)
      session.bin                   one row per change of the control words P9 watches
      <camera>.jpg.bin              every frame, JPEG, back to back
      <camera>.idx.bin              one row per frame: its times, crop, exposure, where its
                                    bytes are (`FRAME_IDX`)

**Not a control loop.** Nothing reads what it writes, nothing waits on it, and no reader of the
arena can delay a writer (seqlocks, single-store publishes). What it can miss is geometry: a
state slot lives ~4 s, a target slot ~2 s, a camera frame `slots / fps` (8 at 30 fps: 0.27 s).
A JPEG of the lens square costs ~2 ms, so the camera ring is the tight one; a frame it could
not copy in time is COUNTED (`dropped` in the manifest), never silently absent.

**A fisheye is recorded as its lens square** (P2's crop, cut, not resized: what is outside it
is the sensor's black), a normal camera whole. The crop is in the index, so a pixel maps back
onto the sensor.

**Live.** Every stream is flushed every `FLUSH_S` (a JPEG blob before its index), so a reader
tailing the files is never more than that behind, and a crash loses no more.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import time

import numpy as np

from elrobot_deploy.runtime.shared_memory import Arena, layout_for, open_all, schema
from elrobot_deploy.runtime.timing import Pacer, now

#: How often the rings are swept: well inside the camera ring's life (0.27 s).
POLL_HZ = 200.0
#: How often every stream reaches the disk.
FLUSH_S = 0.1
#: How many changed paths the manifest names.
_MAX_DIRTY_FILES = 50


def _observed(dtype: np.dtype, *extra) -> np.dtype:
    """An arena record plus the instant this process first saw it."""
    return np.dtype(np.dtype(dtype).descr + list(extra) + [("t_observed", "<f8")])


#: A target as written, with its slot's seqlock count (a re-solve rewrites the same tick).
TARGET_REC = _observed(schema.TARGET, ("seq", "<u8"))

#: One row each time one of these control words changes.
SESSION_REC = np.dtype([
    ("session", "<u8"), ("k_exec", "<i8"), ("intervene", "<u8"), ("bus", "<u8"),
    ("wait", "<u8"), ("t_observed", "<f8"),
])
SESSION_WORDS = ("session", "intervene", "bus", "wait")

#: One row per recorded frame. `crop` is where the encoded image sits on the sensor
#: (x0, y0, x1, y1, end exclusive, may overhang); `offset` / `nbytes` its bytes in the blob.
FRAME_IDX = np.dtype([
    ("frame_id", "<u8"), ("t_capture", "<f8"), ("t_recv", "<f8"), ("t_observed", "<f8"),
    ("crop", "<i8", (4,)), ("exposure_owner", "<i8"), ("exposure", "<i8"), ("gain", "<i8"),
    ("offset", "<u8"), ("nbytes", "<u8"),
])


def jsonable(value):
    """A dataclass tree with numpy and paths in it -> what `json.dumps` takes."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def git_revision() -> dict:
    """Which commit produced this record, and whether the tree was clean. Never raises."""
    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        def git(*args):
            return subprocess.run(("git", "-C", str(root), *args), capture_output=True,
                                  text=True, timeout=10, check=False)

        head = git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"sha": None, "error": head.stderr.strip()[:200]}
        changed = [line[3:] for line in git("status", "--porcelain").stdout.splitlines()]
        return {"sha": head.stdout.strip(), "dirty": bool(changed),
                "dirty_files": sorted(changed)[:_MAX_DIRTY_FILES],
                "dirty_file_count": len(changed)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"sha": None, "error": f"{type(exc).__name__}: {exc}"[:200]}


class Stream:
    """An append-only file of fixed-size records (or, for a JPEG blob, bytes), buffered."""

    def __init__(self, path: pathlib.Path, dtype: np.dtype):
        self.path = path
        self.dtype = np.dtype(dtype)
        self._fh = open(path, "wb", buffering=0)       # noqa: SIM115 (closed by close())
        self._buf = bytearray()
        self._size = 0
        self.count = 0

    def append(self, record) -> None:
        self._buf += np.asarray(record, dtype=self.dtype).tobytes()
        self.count += 1

    def append_bytes(self, payload: bytes) -> int:
        """Returns the byte offset the payload lands at."""
        offset = self._size + len(self._buf)
        self._buf += payload
        self.count += 1
        return offset

    def flush(self) -> None:
        if self._buf:
            self._fh.write(self._buf)
            self._size += len(self._buf)
            self._buf.clear()

    def close(self) -> None:
        self.flush()
        self._fh.close()


class Recorder:
    """One rollout's streams, and the sweep that fills them."""

    def __init__(self, out_dir: pathlib.Path, cfg, rings: dict, *, t_floor: float,
                 code: dict | None = None, rollout: dict | None = None):
        import cv2

        self._cv2 = cv2
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.rings = rings
        self.code = code if code is not None else git_revision()
        self.rollout = dict(rollout or {})
        self.t_floor = float(t_floor)
        self.quality = int(cfg.record.jpeg_quality)

        self.state = Stream(self.dir / "state.bin", schema.STATE)
        self.targets = Stream(self.dir / "targets.bin", TARGET_REC)
        self.intervene = Stream(self.dir / "intervene.bin", schema.INTERVENE)
        self.session = Stream(self.dir / "session.bin", SESSION_REC)
        self.cams = {c.name: c for c in cfg.cameras if c.name in rings["cameras"]}
        self.frames = {name: (Stream(self.dir / f"{name}.jpg.bin", np.dtype("u1")),
                              Stream(self.dir / f"{name}.idx.bin", FRAME_IDX))
                       for name in self.cams}
        self._pixels = {name: np.empty(rings["cameras"][name].shape, np.uint8)
                        for name in self.cams}

        # Only what this rollout produced: everything already in the rings at the edge is
        # the run before it (homing, the READY hold).
        self._seen_state = int(rings["state"]._rec["index"].max())
        self._seen_intervene = int(rings["intervene"].meta()["newest"])
        self._seen_targets: dict[int, int] = {}
        self._seen_words: tuple | None = None
        self._seen_frame = {n: rings["cameras"][n].newest_id() for n in self.cams}
        self.dropped = {n: 0 for n in self.cams}
        self._t_flush = now()
        self.write_manifest(complete=False)

    # --- the sweep -----------------------------------------------------------------------

    def sweep(self) -> None:
        t = now()
        self._sweep_session(t)
        self._sweep_state()
        self._sweep_targets(t)
        self._sweep_intervene()
        self._sweep_frames(t)
        if t - self._t_flush >= FLUSH_S:
            self.flush()
            self._t_flush = t

    def flush(self) -> None:
        for s in (self.state, self.targets, self.intervene, self.session):
            s.flush()
        for blob, index in self.frames.values():
            blob.flush()                     # the bytes before the rows that point at them
            index.flush()

    def _sweep_session(self, t: float) -> None:
        control = self.rings["control"]
        words = tuple(getattr(control, w) for w in SESSION_WORDS)
        if words == self._seen_words:
            return
        out = np.zeros((), dtype=SESSION_REC)
        for name, value in zip(SESSION_WORDS, words):
            out[name] = value
        out["k_exec"] = control.k_exec
        out["t_observed"] = t
        self.session.append(out)
        self._seen_words = words

    def _sweep_by_index(self, ring_rec, lock, dtype, seen: int, stream: Stream) -> int:
        """Every row with an absolute `index` past `seen`, in order (the state and operator
        rings: monotonic, never reused, so nothing is counted twice)."""
        idx = ring_rec["index"]
        fresh = np.flatnonzero(idx > seen)
        for i in fresh[np.argsort(idx[fresh], kind="stable")]:
            rec = lock.snapshot(int(i), dtype)
            if int(rec["index"]) <= seen:
                continue                      # overwritten mid-sweep
            stream.append(rec)
            seen = int(rec["index"])
        return seen

    def _sweep_state(self) -> None:
        ring = self.rings["state"]
        self._seen_state = self._sweep_by_index(ring._rec, ring._lock, schema.STATE,
                                                self._seen_state, self.state)

    def _sweep_intervene(self) -> None:
        ring = self.rings["intervene"]
        self._seen_intervene = self._sweep_by_index(ring._rec, ring._lock, schema.INTERVENE,
                                                    self._seen_intervene, self.intervene)

    def _sweep_targets(self, t: float) -> None:
        """Every slot whose `(k, seq)` is new, copied by field name off a full snapshot (a
        field the ring grows is recorded the day it is added)."""
        ring = self.rings["targets"]
        ks, seqs, deadlines = ring._rec["k"], ring._seq, ring._rec["deadline"]
        for i in range(ring.n):
            seq = int(seqs[i])
            if seq == 0 or seq & 1:
                continue
            k = int(ks[i])
            if k == schema.K_UNSET or self._seen_targets.get(k) == seq:
                continue
            if float(deadlines[i]) <= self.t_floor:
                self._seen_targets[k] = seq   # before this rollout: never ours
                continue
            snap = ring._lock.snapshot(i, ring.DTYPE)
            if int(snap["k"]) != k:
                continue
            out = np.zeros((), dtype=TARGET_REC)
            for name in snap.dtype.names:
                out[name] = snap[name]
            out["seq"] = seq
            out["t_observed"] = t
            self.targets.append(out)
            self._seen_targets[k] = seq

    def _encoded(self, name: str, pixels: np.ndarray, crop) -> bytes | None:
        from elrobot_deploy.runtime.cameras import lens
        x0, y0, x1, _ = crop
        if self.cams[name].lens == "fisheye" and x1 > x0:
            img = lens.cut(pixels, x0, y0, x1 - x0)
        else:
            img = pixels
        ok, buf = self._cv2.imencode(".jpg", img, [self._cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None

    def _sweep_frames(self, t: float) -> None:
        """Every frame published since the last sweep, oldest first. One the ring no longer
        holds (or that was overwritten during the copy) is counted as dropped."""
        for name, ring in ((n, self.rings["cameras"][n]) for n in self.cams):
            newest = ring.newest_id()
            last = self._seen_frame[name]
            if newest <= last:
                continue
            first = max(last + 1, newest - ring.n + 1)
            self.dropped[name] += first - last - 1
            blob, index = self.frames[name]
            for fid in range(first, newest + 1):
                meta = ring.read(fid, self._pixels[name])
                jpg = None if meta is None else self._encoded(name, self._pixels[name],
                                                               meta.crop)
                if jpg is None:
                    self.dropped[name] += 1
                    continue
                row = np.zeros((), dtype=FRAME_IDX)
                row["frame_id"] = meta.frame_id
                row["t_capture"] = meta.t_capture
                row["t_recv"] = meta.t_recv
                row["t_observed"] = t
                row["crop"] = meta.crop
                row["exposure_owner"] = meta.exposure_owner
                row["exposure"] = meta.exposure
                row["gain"] = meta.gain
                row["offset"] = blob.append_bytes(jpg)
                row["nbytes"] = len(jpg)
                index.append(row)
            self._seen_frame[name] = newest

    # --- lifecycle -----------------------------------------------------------------------

    def _streams(self) -> dict:
        return {"state": self.state, "targets": self.targets, "intervene": self.intervene,
                "session": self.session}

    def write_manifest(self, *, complete: bool) -> None:
        """Everything needed to read these files without this code: each stream's own
        `dtype.descr`, so an old record stays readable whatever `schema.py` says today. Written
        at the start (`complete: false`, no counts yet: a live reader needs the dtypes) and
        again at the end."""
        from elrobot_deploy.robot.elrobot_joints import JOINT_NAMES
        manifest = {
            "format": "elrobot-record/1",
            "complete": complete,
            "created": self.rollout.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ended": time.strftime("%Y-%m-%dT%H:%M:%S") if complete else None,
            "streams": {name: {"file": s.path.name, "descr": s.dtype.descr, "records": s.count}
                        for name, s in self._streams().items()},
            "cameras": {
                name: {"blob": f"{name}.jpg.bin", "index": f"{name}.idx.bin",
                       "index_descr": FRAME_IDX.descr, "frames": self.frames[name][1].count,
                       "dropped": self.dropped[name], "lens": self.cams[name].lens,
                       "encoded": ("the lens square" if self.cams[name].lens == "fisheye"
                                   else "the whole frame"),
                       "sensor": [self.cams[name].width, self.cams[name].height],
                       "fps": self.cams[name].fps}
                for name in self.cams},
            "joints": list(JOINT_NAMES),
            "units": {"q": "rad", "deadline / t / t_capture": "s, time.monotonic of this Mac",
                      "pose": "x y z m, qw qx qy qz, base_link (x forward, y left, z up)",
                      "load": "fraction of stall torque, signed", "gripper": "servo rad"},
            "config": jsonable(self.cfg),
            "code": self.code,
            "rollout": jsonable(self.rollout),
        }
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, default=str))
        tmp.replace(self.dir / "manifest.json")

    def close(self) -> None:
        for s in self._streams().values():
            s.close()
        for blob, index in self.frames.values():
            blob.close()
            index.close()
        self.write_manifest(complete=True)


def rollout_dir(base: pathlib.Path, name: str | None = None) -> pathlib.Path:
    """A directory for one rollout: `name` if given (the Astra session's id), else when it
    started; a suffix when that exists already."""
    base = pathlib.Path(base)
    stem = name or time.strftime("%Y%m%d-%H%M%S")
    out, n = base / stem, 1
    while out.exists():
        n += 1
        out = base / f"{stem}-{n}"
    return out


def run(shm_name: str, cfg, *, out_dir: str | pathlib.Path, record_name: str | None = None
        ) -> None:
    """Attach, then one record per ACTIVE period, until STOPPED."""
    arena = Arena.attach(shm_name, layout_for(cfg))
    try:
        rings = open_all(arena)
        control = rings["control"]
        pacer = Pacer(POLL_HZ)
        code = git_revision()            # the code this session imported, sampled once
        n_rollout = 0
        while True:
            t_floor = now()
            while control.session < schema.ACTIVE:
                t_floor = now()
                pacer.wait()
            while control.session != schema.ACTIVE:
                if control.session >= schema.STOPPED:
                    return
                pacer.wait()             # HOME or READY after a reset: wait for the next Start
                t_floor = now()
            n_rollout += 1
            path = rollout_dir(out_dir, record_name)
            recorder = Recorder(path, cfg, rings, t_floor=t_floor, code=code,
                                rollout={"index": n_rollout,
                                         "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         "t_start": t_floor})
            print(f"[P9] recording rollout {n_rollout} to {path}", flush=True)
            try:
                while control.session == schema.ACTIVE:
                    recorder.sweep()
                    pacer.wait()
                recorder.sweep()
            finally:
                recorder.close()
                print(f"[P9] rollout {n_rollout} recorded: {recorder.state.count} bus reads, "
                      + ", ".join(f"{n} {recorder.frames[n][1].count} frames "
                                  f"({recorder.dropped[n]} dropped)" for n in recorder.frames),
                      flush=True)
    finally:
        arena.close()


__all__ = [
    "FRAME_IDX",
    "SESSION_REC",
    "TARGET_REC",
    "Recorder",
    "jsonable",
    "rollout_dir",
    "run",
]

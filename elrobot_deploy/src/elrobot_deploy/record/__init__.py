"""Reading a rollout record back (what P9 writes: `runtime/processes/p9_record.py`).

    rec = load("logs/records/20261017-143200")
    rec.state["q"], rec.targets["pose"], rec.cameras["wrist"].index["t_capture"]
    jpg = rec.cameras["wrist"].jpeg(0)          # the bytes as recorded
    img = rec.cameras["wrist"].image(0)         # BGR

    tail = Tail(path)                           # the same, while P9 is still writing it
    new = tail.poll()                           # {"state": rows, "wrist": index rows, ...}

Every dtype comes from the record's own `manifest.json`, never from today's `schema.py`, so an
old record reads as it was written. A stream is read in whole records only: a reader never sees
half a row of a file being appended to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _dtype(descr) -> np.dtype:
    """A `dtype.descr` as JSON gave it back (lists) -> the dtype."""
    def fix(item):
        name, kind, *shape = item
        if isinstance(kind, list):
            kind = [fix(i) for i in kind]
        return (name, kind, tuple(shape[0])) if shape else (name, kind)
    full = np.dtype([fix(i) for i in descr])
    # An aligned record's padding comes back as unnamed void fields: keep the layout, drop
    # the names numpy would invent for them.
    keep = [n for n, item in zip(full.names, descr) if item[0] != ""]
    return np.dtype({"names": keep, "formats": [full.fields[n][0] for n in keep],
                     "offsets": [full.fields[n][1] for n in keep], "itemsize": full.itemsize})


def manifest(path) -> dict:
    return json.loads((Path(path) / "manifest.json").read_text())


@dataclass
class CameraStream:
    """One camera's frames: `index` (one row per frame, `p9_record.FRAME_IDX`) and the blob."""

    name: str
    index: np.ndarray
    blob: Path
    info: dict

    def jpeg(self, i: int) -> bytes:
        row = self.index[i]
        with open(self.blob, "rb") as fh:
            fh.seek(int(row["offset"]))
            return fh.read(int(row["nbytes"]))

    def image(self, i: int) -> np.ndarray:
        import cv2
        return cv2.imdecode(np.frombuffer(self.jpeg(i), np.uint8), cv2.IMREAD_COLOR)

    def nearest(self, t: float) -> int:
        """The frame captured nearest `t` (monotonic)."""
        return int(np.argmin(np.abs(self.index["t_capture"] - t)))


@dataclass
class Record:
    path: Path
    manifest: dict
    state: np.ndarray
    targets: np.ndarray
    intervene: np.ndarray
    session: np.ndarray
    cameras: dict


def _read_whole(path: Path, dtype: np.dtype, start: int = 0) -> np.ndarray:
    """The whole records of `path` from record `start` on."""
    if not path.exists():
        return np.zeros(0, dtype)
    n = path.stat().st_size // dtype.itemsize
    if n <= start:
        return np.zeros(0, dtype)
    with open(path, "rb") as fh:
        fh.seek(start * dtype.itemsize)
        return np.frombuffer(fh.read((n - start) * dtype.itemsize), dtype=dtype).copy()


def load(path) -> Record:
    path = Path(path)
    m = manifest(path)
    streams = {name: _read_whole(path / s["file"], _dtype(s["descr"]))
               for name, s in m["streams"].items()}
    cams = {name: CameraStream(name, _read_whole(path / c["index"], _dtype(c["index_descr"])),
                               path / c["blob"], c)
            for name, c in m["cameras"].items()}
    return Record(path=path, manifest=m, state=streams["state"], targets=streams["targets"],
                  intervene=streams["intervene"], session=streams["session"], cameras=cams)


class Tail:
    """A record being written, read incrementally: each `poll()` returns the rows that reached
    the disk since the last one, per stream and per camera index. A camera's index row is
    returned only once its JPEG bytes are on disk too (P9 flushes the blob first)."""

    def __init__(self, path):
        self.path = Path(path)
        m = manifest(self.path)
        self.manifest = m
        self._streams = {name: (self.path / s["file"], _dtype(s["descr"]))
                         for name, s in m["streams"].items()}
        self._cams = {name: (self.path / c["index"], _dtype(c["index_descr"]),
                             self.path / c["blob"])
                      for name, c in m["cameras"].items()}
        self._seen = {name: 0 for name in list(self._streams) + list(self._cams)}

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(self._cams)

    def poll(self) -> dict:
        out = {}
        for name, (file, dtype) in self._streams.items():
            rows = _read_whole(file, dtype, self._seen[name])
            self._seen[name] += len(rows)
            out[name] = rows
        for name, (file, dtype, blob) in self._cams.items():
            rows = _read_whole(file, dtype, self._seen[name])
            if len(rows):
                have = blob.stat().st_size if blob.exists() else 0
                ok = (rows["offset"] + rows["nbytes"]) <= have
                if not ok.all():
                    rows = rows[:int(np.argmin(ok))]
            self._seen[name] += len(rows)
            out[name] = rows
        return out

    def jpeg(self, camera: str, row) -> bytes:
        blob = self._cams[camera][2]
        with open(blob, "rb") as fh:
            fh.seek(int(row["offset"]))
            return fh.read(int(row["nbytes"]))

    def complete(self) -> bool:
        try:
            return bool(manifest(self.path).get("complete"))
        except (OSError, ValueError):
            return False


__all__ = ["CameraStream", "Record", "Tail", "load", "manifest"]

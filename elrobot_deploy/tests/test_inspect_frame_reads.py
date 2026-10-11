"""`Camera.jpeg` under the inspection server's threads.

The server answers image requests concurrently, and a scrub sends many for one
camera at once. With a shared seek-then-read, 29% of concurrent responses were
another frame's bytes (measured 2026-09-28); the read must be positional.
"""

from __future__ import annotations

import concurrent.futures as cf
import io

import numpy as np

from g1d_deploy.dashboards.inspect.loader import Camera

N = 300


def _camera(tmp_path, *, in_memory=False):
    frames = [bytes([i % 251]) * (1000 + 37 * i) for i in range(N)]
    offsets = np.cumsum([0] + [len(f) for f in frames[:-1]])
    index = np.zeros(N, dtype=[("offset", "<u8"), ("nbytes", "<u4"), ("frame_id", "<u8")])
    index["offset"] = offsets
    index["nbytes"] = [len(f) for f in frames]
    index["frame_id"] = np.arange(N)
    raw = b"".join(frames)
    if in_memory:
        blob = io.BytesIO(raw)
    else:
        path = tmp_path / "cam.jpg.bin"
        path.write_bytes(raw)
        blob = open(path, "rb")                              # noqa: SIM115
    return Camera("cam", index, blob, dropped=0), frames


def _hammer(cam, frames):
    order = list(range(N)) * 4
    with cf.ThreadPoolExecutor(32) as ex:
        got = list(ex.map(cam.jpeg, order))
    return sum(g != frames[i] for g, i in zip(got, order))


def test_concurrent_reads_of_a_file_blob_return_their_own_frame(tmp_path):
    cam, frames = _camera(tmp_path)
    try:
        assert _hammer(cam, frames) == 0
    finally:
        cam.close()


def test_an_in_memory_blob_is_serialised_rather_than_raced(tmp_path):
    cam, frames = _camera(tmp_path, in_memory=True)
    assert _hammer(cam, frames) == 0

"""The overlay: the same gripper the viewer already draws, from other chunks.

The anchoring test is the first one — built from the recording's OWN chunks, the
overlay must reproduce `scene.build`'s amber ghost exactly. That pins the whole
chain (tick -> chunk -> row -> jaws -> world) against code that is already on
screen, so a solve's overlay can only differ by the chunks it was built from.
"""

from __future__ import annotations

import numpy as np
import pytest

from elrobot_deploy.dashboards.inspect import loader, timeline
from elrobot_deploy.dashboards.inspect import scene as scene_mod
from elrobot_deploy.dashboards.web.pack import pack as _pack
from elrobot_deploy.dashboards.web.pack import spans as _spans

from elrobot_deploy.dashboards.inspect.modes.replay import anchors
from elrobot_deploy.dashboards.inspect.modes.replay.overlay import OverlayBuilder

from . import rollout_fixtures as fixtures

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = fixtures.write_record(tmp_path_factory.mktemp("rec") / "r")
    roll = loader.load(path)
    tl = timeline.build(roll, 30.0)
    sc = scene_mod.build(roll, tl)
    builder = OverlayBuilder(roll, tl, sc)
    live = anchors.live_chunks(roll.chunks)
    yield {"roll": roll, "tl": tl, "scene": sc, "builder": builder, "live": live}
    roll.close()


def test_live_chunks_reproduce_the_viewer_ghost(built):
    out = built["builder"].build(built["live"], None)
    parts = dict(out["parts"])
    sc = built["scene"]
    both = parts["valid"].astype(bool) & np.asarray(sc.ghost_valid).astype(bool)
    assert both.sum() > 0
    assert np.abs(parts["ghost"][both] - sc.ghost[both]).max() < 1e-6
    assert np.abs(parts["ghost_tcp"][both] - sc.ghost_tcp[both]).max() < 1e-6


def test_slots_are_the_viewers_own(built):
    """The browser indexes meshes it already uploaded, so the slot list has to
    be the loaded Scene's, not a second computation of it."""
    out = built["builder"].build(built["live"], None)
    assert out["meta"]["ghost_geoms"] == list(built["scene"].ghost_geoms)


def test_the_arc_passes_through_the_ghost(built):
    out = built["builder"].build(built["live"], None)
    p = dict(out["parts"])
    checked = 0
    for f in range(built["tl"].n):
        c, j = int(p["chunk_of"][f]), int(p["row_of"][f])
        if c < 0 or not p["valid"][f]:
            continue
        assert np.abs(p["arc"][c, j] - p["ghost_tcp"][f]).max() < 1e-6
        checked += 1
    assert checked > 0


def test_a_solve_against_itself_is_zero_error(built):
    """Not bit-zero: `ghost_tcp` is stored float32 for the wire and the
    reference is computed in float64, so the floor is ~3e-5 mm — 30 picometres,
    and a useful demonstration that the two paths agree to the storage."""
    out = built["builder"].build(built["live"], built["live"])
    assert float(dict(out["parts"])["err_mm"].max()) < 1e-3
    assert out["meta"]["err_mm_p50"] == 0.0


def test_error_is_the_tcp_distance(built):
    """Shift one chunk by a known amount and the readout must show it."""
    moved = built["live"].copy()
    moved["pose"][:, :, :, 0] += 0.01                # 10 mm along x
    out = built["builder"].build(moved, built["live"])
    err = dict(out["parts"])["err_mm"]
    valid = dict(out["parts"])["valid"].astype(bool)
    assert np.allclose(err[valid], 10.0, atol=1e-3)


def test_rows_past_the_end_are_held(built):
    """A chunk covers P ticks and the next lands later, so the last ticks of a
    period run on a chunk that has run dry — which is what P7 does with them."""
    out = built["builder"].build(built["live"], None)
    p = dict(out["parts"])
    n_rows = int(p["arc_n"][0])
    assert set(p["row_of"][p["row_of"] >= 0].tolist()) <= set(range(n_rows))
    assert int(p["row_of"].max()) == n_rows - 1


def test_a_solves_tail_is_carried(built):
    """The live recorder stores P rows; a solve stores H, and the extra is the
    tail the policy predicted and never got to use."""
    wide = built["live"].copy()
    for c in range(len(wide)):
        for j in range(int(wide["n_actions"][c]), wide["pose"].shape[1]):
            wide["pose"][c, j] = wide["pose"][c, int(wide["n_actions"][c]) - 1]
    out = built["builder"].build(wide, built["live"])
    p = dict(out["parts"])
    assert int(p["arc_n"][0]) == fixtures.H
    assert int(p["arc_exec"][0]) == fixtures.P


def test_packing_and_spans_agree(built):
    """The browser slices the buffer by the spans, so the two must be built from
    the same list — and every float section must land 4-byte aligned."""
    parts = built["builder"].build(built["live"], built["live"])["parts"]
    blob, spans = _pack(parts), _spans(parts)
    assert len(blob) >= max(o + n for o, n in spans.values())
    for name, arr in parts:
        off, nbytes = spans[name]
        assert nbytes == arr.nbytes
        if arr.dtype.itemsize >= 4:
            assert off % 4 == 0, name
        got = np.frombuffer(blob, dtype=arr.dtype, count=arr.size,
                            offset=off).reshape(arr.shape)
        assert np.array_equal(got, arr), name

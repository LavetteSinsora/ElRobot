"""`reader.arena`: every region of the live arena, read-only.

For a policy that needs what `obs` and the reader's windows do not give — the
cursor P7 is executing, the chunks it published and when each tick is due
(real-time chunking), P6's targets. Every read the processes use works; every
write raises, because each region has exactly one writer.
"""

from __future__ import annotations

import numpy as np
import pytest

from elrobot_deploy.runtime.processes.p5_policy import history_reader
from elrobot_deploy.runtime.shared_memory import ArenaRings, open_read_only, schema

from .test_arena import _chunk
from .test_p5_policy import StubPolicy


@pytest.fixture
def ro(arena, cfg):
    return open_read_only(arena, cfg)


def test_reads_see_what_the_writers_wrote(rings, ro, cfg):
    rings["control"].k_exec = 42
    _chunk(rings, cfg, 3, 40, cfg.P, 7.0)
    rings["control"].newest_chunk_id = 3
    assert ro.control.k_exec == 42 and ro.control.newest_chunk_id == 3
    (c,) = ro.chunks.covering(41)
    assert c.chunk_id == 3 and c.pose[0, 0] == 7.0


def test_a_chunk_is_read_whole_by_id_and_gone_once_its_slot_is_reused(rings, ro,
                                                                      cfg):
    _chunk(rings, cfg, 1, 10, cfg.P, 1.0)
    got = ro.chunks.get(1)
    assert int(got["k_first_action"]) == 10 and int(got["n_actions"]) == cfg.P
    got["pose"][...] = 99.0                          # a private copy
    assert ro.chunks.get(1)["pose"][0, 0, 0] == 1.0
    _chunk(rings, cfg, 1 + ro.chunks.n, 20, cfg.P, 2.0)   # same slot
    assert ro.chunks.get(1) is None


@pytest.mark.parametrize("write", [
    lambda ro: setattr(ro.control, "k_exec", 1),
    lambda ro: setattr(ro.control, "newest_chunk_id", 1),
    lambda ro: ro.targets.write(5, 1.0, np.zeros(schema.BODY_DOF), intervene=0),
    lambda ro: ro.chunks.write(
        chunk_id=1, k_first_action=0, n_actions=1,
        pose=np.zeros((1, schema.N_HANDS, schema.POSE_DIM)),
        hand=np.zeros((1, schema.N_HANDS, schema.HAND_DIM)),
        t_trigger=0.0, t_anchor=0.0, t_obs_received=0.0, t_cpu_processed=0.0,
        t_chunk_received=0.0, t_chunk_written=0.0),
], ids=["k_exec", "newest_chunk_id", "target", "chunk"])
def test_every_write_raises(ro, write):
    with pytest.raises(ValueError, match="read-only"):
        write(ro)


def test_the_live_reader_carries_it_and_the_dummy_does_not(rings, ro, cfg):
    live = dict(rings, read_only=ro)
    assert isinstance(history_reader(live, StubPolicy(cfg)).arena, ArenaRings)
    assert history_reader(rings, StubPolicy(cfg)).arena is None

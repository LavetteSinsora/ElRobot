"""`[control] blend = "package.module:function"`: a deploy file names its own
blend, resolved at launch in g1d_deploy's Python (P6 runs it there), and each
contributor carries its `row` so an age-weighted blend can be written."""

from __future__ import annotations

import pytest

from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.config_loading.validate import validate
from g1d_deploy.runtime.execution import chunk_blend

from .test_arena import _chunk


def test_a_module_path_resolves_to_that_function():
    assert (chunk_blend.get("g1d_deploy.runtime.execution.chunk_blend:newest")
            is chunk_blend.newest)


@pytest.mark.parametrize("spec", ["no_such_module_g1d:fn",
                                  "g1d_deploy.runtime.execution.chunk_blend:nope",
                                  "g1d_deploy.runtime.execution.chunk_blend:_BLENDS",
                                  "median"])
def test_a_blend_that_does_not_resolve_is_refused_at_launch(cfg, hw, spec):
    with pytest.raises(ConfigError, match="blend"):
        validate(cfgmod.replace(cfg, blend=spec), hw)


def test_each_contributor_knows_its_row_in_its_chunk(rings, cfg):
    _chunk(rings, cfg, 1, 10, cfg.P, 1.0)
    _chunk(rings, cfg, 2, 11, cfg.P, 2.0)
    got = rings["chunks"].covering(12)
    assert [(c.chunk_id, c.row) for c in got] == [(1, 2), (2, 1)]

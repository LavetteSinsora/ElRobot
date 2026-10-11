"""A blend of your own: a mode-aware temporal ensemble.

    [control]
    sync = false
    blend = "blend.py:consistent_ensemble"

    elrobot_deploy check-policy examples/custom_modules/deploy.toml

The shipped default, `temporal_ensemble` (ACT's), averages every chunk's
prediction of a tick. That is right while consecutive chunks describe the same
motion, and wrong when they do not: a multimodal policy (diffusion, flow) may
go left of an object in one chunk and right of it in the next, and the
average of the two is a path straight through the object that neither chunk
chose.

So this blend averages only the chunks that AGREE with the newest one at this
tick — within `AGREE_M` of it at every flange (joint mode: within
`AGREE_RAD` at every joint) — with ACT's weights, and drops the rest. When the
policy keeps its plan it behaves like `temporal_ensemble` (smooth seams); when
the policy changes its mind it behaves like `newest` (a clean switch, no
blend of two plans).

What P6 hands a blend (`docs/infra_logic/chunk_blending.md`, "What a blend
receives and returns"): every adopted chunk covering
the tick, oldest first, quaternions already in the newest one's hemisphere,
each with its `row`. What it must return: one whole `Action`, pose, hand and
waist from the same decision — `chunk_blend.weighted` does that for any
weights. It runs in P6, under elrobot_deploy's own Python, once per tick per
re-solve: pure numpy, microseconds.
"""

from __future__ import annotations

import numpy as np

from elrobot_deploy.runtime.execution.chunk_blend import (TEMPORAL_ENSEMBLE_M,
                                                      newest, weighted)

#: How far a chunk's flange may be from the newest's, at this tick, and still
#: count as the same plan. A few centimetres: well above the chunk-to-chunk
#: jitter of a consistent policy, well below a different way around an object.
AGREE_M = 0.03
#: The same for joint-mode chunks, per joint.
AGREE_RAD = 0.15


def _agrees(c, ref) -> bool:
    if c.pose is not None:
        return float(np.max(np.linalg.norm(c.pose[:, :3] - ref.pose[:, :3],
                                           axis=1))) <= AGREE_M
    return float(np.max(np.abs(c.arm_q - ref.arm_q))) <= AGREE_RAD


def consistent_ensemble(contributors):
    ref = contributors[-1]
    kept = [c for c in contributors if _agrees(c, ref)]      # always keeps ref
    if len(kept) == 1:
        return newest(kept)
    rows = np.array([max(int(c.row), 0) for c in kept], np.float64)
    w = np.exp(TEMPORAL_ENSEMBLE_M * (rows - rows.max()))
    return weighted(kept, w / w.sum())

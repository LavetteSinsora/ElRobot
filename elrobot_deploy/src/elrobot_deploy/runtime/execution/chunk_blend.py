"""Many chunks' opinions about one tick, reduced to one action.

Called by P6 once per tick (`runtime/processes/p6_target.py`). Design, and how
to write a blend: `docs/infra_logic/chunk_blending.md`.

The blend is a *pluggable function*: contributors in, one action out, chosen per
deploy file. What the infra fixes is not the method but the two things around
it, and both live here rather than inside the functions:

  1. **Hemisphere alignment is the framework's.** `-q` and `q` are the same
     rotation; averaged naively they are the identity — a 180-degree lurch that
     no test of a single chunk can produce. Doing the sign flip inside each
     blend function means every new function re-implements it, and one of them
     eventually will not. `apply()` is therefore the only entry
     point P6 uses, and a blend function never sees unaligned input.

  2. **The hand and the waist blend with the pose** (or the joints) — same contributors, same
     weights, same commit point. A crossfade that executed one chunk's grip
     against another chunk's pose would close the jaw somewhere between the two
     trajectories. That is why the blend functions take whole `Contribution`s
     and return a whole `Action`, instead of being a function over poses that
     the caller then has to remember to apply three times.

Sync always yields exactly one contributor, so everything here is inert
there; it is async's overlap that the method actually resolves.
"""

from __future__ import annotations

import functools
from typing import Callable, NamedTuple, Sequence

import numpy as np

from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.shared_memory.rings import Contribution


class Action(NamedTuple):
    """One tick of the contract, after the contributors are reduced to one.

    Exactly one of `pose` (eef) and `arm_q` (joint) is set. `waist` is None
    unless a joint-mode policy commanded one — or an operator did (an
    intervention row always carries the measured trunk) — and None means P6 owns it.
    """

    pose: np.ndarray | None         # (2, 7)  absolute flange pose, base frame, wxyz
    hand: np.ndarray                # (2, 6)  end-effector command
    waist: np.ndarray | None        # (3,)    commanded joint angles
    arm_q: np.ndarray | None = None  # (2, 7)  joint mode: commanded arm joints


# --- the framework's part ---------------------------------------------------


def hemisphere_align(contributors: Sequence[Contribution]) -> list[Contribution]:
    """Flip every contributor's quaternions into the newest one's hemisphere.

    The newest contributor is the reference because it is the one the
    newest-wins method would have picked outright: whatever the method, the
    result stays on the side of the double cover that the most recent inference
    chose, so a blend never drifts to the antipode between two replans.

    Per arm, not per contributor: the two arms are independent rotations and a
    left flange whose quaternion needs flipping says nothing about the right.
    """
    if len(contributors) < 2 or contributors[-1].pose is None:
        return list(contributors)       # one opinion, or joint mode: no quaternions
    ref = contributors[-1].pose[:, 3:]                      # (2, 4)
    out = []
    for c in contributors:
        flip = np.sum(c.pose[:, 3:] * ref, axis=1) < 0.0    # (2,)
        if flip.any():
            pose = c.pose.copy()
            pose[flip, 3:] *= -1.0
            c = c._replace(pose=pose)
        out.append(c)
    return out


def apply(fn: Callable[[Sequence[Contribution]], Action],
          contributors: Sequence[Contribution]) -> Action:
    """Align, then blend. P6 calls this and never a blend function directly."""
    if not contributors:
        raise ValueError("blend of an empty contributor set")
    return fn(hemisphere_align(contributors))


# --- the methods ------------------------------------------------------------


def newest(contributors: Sequence[Contribution]) -> Action:
    """Highest `chunk_id` wins: each new chunk takes over outright.

    The right choice when the policy already makes consecutive chunks agree
    (real-time chunking). `covering()` returns ascending `chunk_id`, so the
    newest is the last. The copies are deliberate: the contributor's arrays
    came out of a seqlock read and the caller is about to hand them to the IK,
    which must not alias anything the ring can rewrite.
    """
    c = contributors[-1]
    return Action(_copy(c.pose), c.hand.copy(), _copy(c.waist), _copy(c.arm_q))


def _copy(a):
    return None if a is None else a.copy()


def _mean(w, blocks):
    """The weighted average of `blocks`, or None if any contributor lacks it.

    "Any", not "all": within one run every chunk speaks one mode, so a mixed
    set is a waist that some chunks command and some leave to P6 — and
    averaging a commanded trunk with an absent one has no meaning to report.
    """
    if any(b is None for b in blocks):
        return None
    return np.tensordot(w, np.stack(blocks), axes=(0, 0))


def weighted(contributors: Sequence[Contribution], w) -> Action:
    """The weighted average of whole actions, `w` summing to 1 — pose, hand,
    waist and joints with the SAME weights, quaternions averaged linearly and
    renormalised (valid because `apply()` aligned their hemispheres; over one
    replan period the contributors are close, where linear and geodesic
    averages agree to the fourth decimal). The building block of every
    averaging blend, shipped or yours."""
    if len(contributors) == 1:
        return newest(contributors)
    w = np.asarray(w, dtype=np.float64)
    hand = _mean(w, [c.hand for c in contributors])
    waist = _mean(w, [c.waist for c in contributors])
    arm_q = _mean(w, [c.arm_q for c in contributors])
    pose = _mean(w, [c.pose for c in contributors])
    if pose is None:
        # Joint mode: joint angles average linearly, no hemisphere to mind.
        return Action(None, hand, waist, arm_q)
    for h in range(schema.N_HANDS):
        norm = float(np.linalg.norm(pose[h, 3:]))
        if norm > 1e-9:
            pose[h, 3:] /= norm
        else:
            # A vanishing norm needs contributors a half-turn apart *from each
            # other*, which alignment against the newest does not exclude.
            # There is no average to report, so defer to the newest rather than
            # normalize a rounding error into an orientation.
            pose[h] = contributors[-1].pose[h]
    return Action(pose, hand, waist, arm_q)


#: ACT's `m` (Zhao et al., 2023, "temporal ensembling"): per TICK of a
#: prediction's age. ACT used 0.01 with one prediction per step.
TEMPORAL_ENSEMBLE_M = 0.01


def temporal_ensemble(contributors: Sequence[Contribution], *,
                      m: float = TEMPORAL_ENSEMBLE_M) -> Action:
    """ACT's temporal ensembling: every chunk's prediction of this tick,
    averaged with `w_i = exp(-m * i)`, `i = 0` the OLDEST prediction. The
    default blend.

    ACT indexes predictions by the step they were made at, one per step. Here a
    chunk arrives every P ticks, so a prediction's age is how far into its own
    chunk the tick lies (`Contribution.row`): the oldest chunk has the largest
    row, and `exp(-m * i)` over ACT's steps is `exp(m * row)` normalised —
    same weights, whatever P is. Older predictions weigh slightly more, which
    ACT found smoother; `m = 0` is a plain mean, and a large `m` leans on the
    oldest. Contributors without a row (-1) are averaged equally.

    The effect: consecutive chunks are blended over their H - P overlapping
    ticks instead of switched at the seam, which smooths chunk boundaries at
    the cost of reacting more slowly to what the newest chunk says. A model
    whose chunks already agree (real-time chunking) wants `newest` instead.
    """
    rows = np.array([max(int(c.row), 0) for c in contributors], np.float64)
    if any(int(c.row) < 0 for c in contributors):
        rows[:] = 0.0
    w = np.exp(float(m) * (rows - rows.max()))       # shifted: no overflow
    return weighted(contributors, w / w.sum())


#: The shipped blends. `temporal_ensemble` is the default (`DeployConfig.blend`);
#: in sync every tick has one contributor, so the blend changes nothing there.
_BLENDS: dict[str, Callable[..., Action]] = {
    "temporal_ensemble": temporal_ensemble,
    "newest": newest,
}


def get(name, cfg=None) -> Callable[[Sequence[Contribution]], Action]:
    """Resolve `deploy_config.blend` to a function.

    Three spellings: a shipped name (`"temporal_ensemble"`, `"newest"`); a
    `"package.module:function"` or `"path/file.py:function"` (made absolute at
    resolution, against the deploy file), imported here — which is how a
    deploy file names its own blend; or a callable, passed through unchanged
    (tests, and code that builds a config in Python). `cfg` supplies the
    shipped blends' knob (`temporal_ensemble_m`).

    A custom function has the shipped ones' signature,
    `fn(contributors) -> Action`: every adopted chunk covering the tick,
    ascending `chunk_id`, quaternions already in the newest one's hemisphere
    (`apply`), each with its `row` in its own chunk. It runs in P6 — the
    infra's Python, not the policy's — once per tick, inside P6's per-tick
    budget, so it must be pure numpy and fast, and return a WHOLE `Action`
    (pose or arm_q, hand, waist) so they come from one decision.
    """
    if callable(name):
        return name
    if isinstance(name, str) and ":" in name:
        from elrobot_deploy.policy_contract.loading import load_attr
        try:
            fn = load_attr(name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"blend {name!r} cannot be imported: {exc}") from exc
        if not callable(fn):
            raise ValueError(f"blend {name!r} is not a function")
        return fn
    if name == "temporal_ensemble" and cfg is not None:
        m = float(getattr(cfg, "temporal_ensemble_m", TEMPORAL_ENSEMBLE_M))
        return functools.partial(temporal_ensemble, m=m)
    try:
        return _BLENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown blend {name!r}; known: {sorted(_BLENDS)}, or "
            f"\"package.module:function\"") from None


__all__ = ["Action", "TEMPORAL_ENSEMBLE_M", "apply", "get", "hemisphere_align",
           "newest", "temporal_ensemble", "weighted"]

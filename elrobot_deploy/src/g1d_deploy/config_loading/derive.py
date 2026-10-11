"""What is computed from the resolved settings: ring sizes and servo gains.

`Derived` is computed, printed at startup, and never written down. A hand-set
ring size does not fail when it is too small: it wraps, returns a stale record,
and the run looks plausible.

Computed once by P0 and again, identically, by every process that attaches:
the sizes feed `runtime/shared_memory/layout.py`, whose hash each process
checks. Design: `docs/infra_logic/arena_layout_and_reads.md`, "The regions".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from g1d_deploy.config_loading.resolved import DeployConfig, Hardware, next_pow2

@dataclass(frozen=True)
class Derived:
    """The derived sizes. Printed at startup; never configured."""

    dt: float
    chunk_slots: int
    n_target: int
    state_slots: int
    n_contrib_max: int
    intervene_slots: int
    command_slots: int = 256
    #: Slots per camera VIEW ring, `resolved.camera_slots` of the history the
    #: policy declared for it. A view absent here gets
    #: `CAMERA_MIN_SLOTS` — a hand-built `Derived` in a test means what it did.
    camera_slots: dict = field(default_factory=dict)

    def describe(self) -> str:
        cams = " ".join(f"{n}:{k}" for n, k in sorted(self.camera_slots.items()))
        return (f"dt={self.dt*1e3:.2f}ms  chunk_slots={self.chunk_slots}  "
                f"target_slots={self.n_target}  "
                f"state_slots={self.state_slots}  "
                f"intervene_slots={self.intervene_slots}  "
                f"command_slots={self.command_slots}  "
                f"max_contributors={self.n_contrib_max}  "
                f"camera_slots=[{cams}]")


#: The smallest replan period `validate` accepts — and so the worst case the
#: chunk ring is sized for. See `derive`.
P_MIN = 2


def derive(cfg: DeployConfig, hw: Hardware) -> Derived:
    """The ring sizes. **Independent of every switchable setting.**

    The replan period, sync/async and the control rate can change between
    rollouts (`rollout_settings.RolloutSettings`), and the arena is never resized — so
    the two rings whose size depends on `P` are sized for the WORST `P` any
    rollout of this launch could ask for, from `H` alone:

    * the chunk ring holds `ceil(H/P) + 1` live contributors, largest at the
      smallest period `validate` allows (`P_MIN`);
    * the target ring holds `H + 2P` ticks, largest at the largest period,
      `P = H`, so `next_pow2(3H)`.

    A few kB either way, and it is what lets P change without reallocating.
    Nothing here reads `cfg.P`, so two configs differing only in
    rollout settings build the SAME layout — the hash every process checks.
    `dt` is not part of the layout; it is printed, and the rollout's own dt is
    what P5 and P6 schedule with.
    """
    H = cfg.H
    n_contrib = -(-H // min(P_MIN, H))            # ceil, at the worst P
    return Derived(
        dt=cfg.dt,
        # +1 over the live-contributor count: at an exact fit, chunk n+S
        # overwrites chunk n at the moment chunk n's last tick expires, with P6
        # possibly mid-read of the old slot.
        chunk_slots=n_contrib + 1,
        # ONE target ring for arm, waist and hand. The horizon, plus two
        # replan periods of slack so a tick cannot be overwritten while P7 is
        # still walking toward it — at the largest period a rollout may pick.
        n_target=next_pow2(3 * H),
        state_slots=next_pow2(hw.lowstate_hz * cfg.state_history_s),
        n_contrib_max=n_contrib,
        # The intervention ring, sized off the POLL rate rather than off any
        # number of ticks: its rows are the operator's stream, not the tick line,
        # and the only reader that wants more than the newest one is P9. Four
        # seconds of polls rounds to 1024 rows (~344 kB), which is ~11 s of the
        # ~90 Hz at which rows are actually written — comfortably more than P9's
        # sweep period, which is the only thing that has to fit.
        #
        # Allocated whether or not `--intervene` is passed. A region that exists
        # and is never written costs one `mmap` extent of address space and keeps
        # the layout hash the same for every process in the run, which is the
        # thing that must not depend on a command-line flag.
        intervene_slots=next_pow2(hw.pico_poll_hz * 4),
        # P7's own publish log (`schema.COMMAND`), read only by P9. Half a
        # second of servo periods: P9 sweeps at 500 Hz, so this survives a
        # recorder stall of ~250 sweeps before a command goes unrecorded — and
        # an index gap in `command.bin` says so when it does.
        command_slots=next_pow2(hw.body_hz * 0.5),
        # One ring per camera VIEW, as deep as the history the policy declared
        # for it plus slack (`camera_slots`). Sized here, once, like every other
        # ring: the arena is never resized, so a history longer than this is
        # refused at load (`policy_contract.check_against`) rather than silently lapped.
        camera_slots={name: v.slots for name, v in cfg.views.items()},
    )


def servo_gains(cfg: DeployConfig, hw: Hardware) -> tuple[np.ndarray, np.ndarray]:
    """The `(kp, kd)` (29,) this run puts on the wire: the robot file's, with
    `cfg.servo.arm_gain_scale` applied to the arm joints only (kd by its square
    root, which keeps each joint's damping ratio). The ONE derivation — P7
    writes these at connect and P9 records them."""
    kp = np.asarray(hw.kp, dtype=np.float64).copy()
    kd = np.asarray(hw.kd, dtype=np.float64).copy()
    s = float(cfg.servo.arm_gain_scale)
    if s != 1.0:
        arm = list(hw.joints_arm)
        kp[arm] *= s
        kd[arm] *= np.sqrt(s)
    return kp, kd


__all__ = ["P_MIN", "Derived", "derive", "servo_gains"]

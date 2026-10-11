"""Read a rollout record, and say what the robot actually did.

    python -m elrobot_deploy.rollout_record_utils.reader <run_dir>
    python -m elrobot_deploy.rollout_record_utils.reader <run_dir> --homing

P9 writes raw arrays and a manifest that carries each stream's dtype, so the
files are readable without this module — `np.fromfile` and the recorded `descr`
is the whole contract. This exists because "readable in principle" is not the
same as "read": a recorder nobody can query is a recorder nobody checks.

`--homing` answers one specific question, because it is the one that comes up
first on a new robot: **did the arm ramp, or did it jump?** P0 writes homing as
ticks `k < 0`, one every `dt`, and P7 interpolates between them at the servo
rate. Three ways that can go wrong and all three are visible here:

* the ramp was never written as a ramp (deadlines not `dt` apart);
* it was written but not followed (measured `q` does not track the targets);
* it was followed, and the motion really was that fast (the ramp is too short
  for the distance — `home_min_s`).

The measured trajectory is the arbiter. Commanded targets say what was asked
for; only `state.bin` says what the arm did.

Used by P9's tests, the inspection loader (`dashboards/inspect/loader.py`) and
by hand. Dtypes always come from the record's own manifest.
"""

import argparse
import json
import pathlib

import numpy as np

from elrobot_deploy.runtime.shared_memory import schema


def _dtype(descr) -> np.dtype:
    """Rebuild a dtype from the manifest's own record of it (never from
    `schema.py`, which may have moved on since the record was written)."""
    return np.dtype([tuple(field) for field in descr])


#: The columns of the `hand_targets` view (`_target_views`). `intervene` too:
#: a reader that filters hand targets on the intervention counter would
#: otherwise find no column and silently keep every row, including the policy
#: targets a takeover orphaned.
_HAND_VIEW = [("k", "<i8"), ("deadline", "<f8"), ("intervene", "<u8"),
              ("cmd", "<f8", (schema.N_HANDS, schema.HAND_DIM)),
              ("seq", "<u8"), ("t_observed", "<f8")]


def _target_views(out: dict) -> None:
    """Two views of the one `targets` stream P9 records, for readers that want
    the body or the hand half on its own:

      * `body_targets` — the same rows (joint targets, IK input and residual);
      * `hand_targets` — `k`, `deadline`, `intervene` and the row's `hand` as
        `cmd`.

    A record with no targets (nothing was commanded) gets empty views.
    """
    targets = out.get("targets")
    if targets is None:
        targets = out["targets"] = np.empty(0)
    out["body_targets"] = targets
    names = targets.dtype.names or ()
    view = np.zeros(len(targets), dtype=np.dtype(_HAND_VIEW))
    for name, src in (("k", "k"), ("deadline", "deadline"),
                      ("intervene", "intervene"), ("cmd", "hand"),
                      ("seq", "seq"), ("t_observed", "t_observed")):
        if src in names:
            view[name] = targets[src]
    out["hand_targets"] = view


def load(run_dir) -> dict:
    """Every fixed-size stream, by name. Camera blobs are left on disk."""
    run_dir = pathlib.Path(run_dir).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    out = {"manifest": manifest, "dir": run_dir}
    for name, spec in manifest["streams"].items():
        path = run_dir / spec["file"]
        out[name] = (np.fromfile(path, dtype=_dtype(spec["descr"]))
                     if path.exists() else np.empty(0))
    _target_views(out)
    return out


def body_joint_index(manifest: dict) -> np.ndarray | None:
    """Which columns of state `q` a body target's 17 entries correspond to.

    Recorded in the manifest. A record without it
    falls back to the CURRENT robot file, which is right whenever the
    record came from this machine and this config — the common case, and the
    only one where the comparison is meaningful anyway.
    """
    joints = manifest.get("joints")
    if joints is None:
        try:
            from elrobot_deploy.config_loading.resolved import load_hardware
            from elrobot_deploy.config_loading.errors import ConfigError

            hw = load_hardware()
            joints = {"arm": list(hw.joints_arm), "waist": list(hw.joints_waist)}
        except (ConfigError, OSError, ImportError):
            # Reading a record must not require a working robot file — the
            # commanded half of the report needs no joint map at all, and it is
            # better to lose the measured half than the whole file.
            return None
    return np.asarray(list(joints["arm"]) + list(joints["waist"]), dtype=int)


#: `schema.chunk_dtype`'s `mode` word, restated so reading a record never
#: imports the policy package: 0 = eef, 1 = joint.
CHUNK_MODES = {0: "eef", 1: "joint"}


def chunk_actions(row) -> dict:
    """One recorded chunk's EXECUTED rows (`[:n_actions]`), by mode.

    `{"mode", "hand", "waist", "pose" | "arm_q"}` — `pose` for an eef chunk,
    `arm_q` `(n, 2, 7)` for a joint one, and `waist` the joint policy's
    `waist_q` (required in joint mode) or None for an eef chunk, whose waist
    P6 decided — it is in the targets' `q[14:17]`.

    Records written before the chunk slot carried a mode (arena schema < 9)
    have no `mode` column: every one of those is eef, and its `waist` column
    WAS the policy's command, so it is returned as such.
    """
    names = row.dtype.names or ()
    n = int(row["n_actions"])
    mode = CHUNK_MODES.get(int(row["mode"]), "eef") if "mode" in names else "eef"
    has_waist = mode == "joint" or "mode" not in names
    out = {"mode": mode, "hand": np.array(row["hand"][:n]),
           "waist": np.array(row["waist"][:n]) if has_waist else None}
    if mode == "eef":
        out["pose"] = np.array(row["pose"][:n])
    else:
        out["arm_q"] = np.array(row["arm_q"][:n])
    return out


def chunk_modes(chunks) -> dict[str, int]:
    """How many recorded chunks spoke each mode — one key for any real run."""
    if chunks is None or not len(chunks):
        return {}
    if "mode" not in (chunks.dtype.names or ()):
        return {"eef": len(chunks)}
    modes, counts = np.unique(np.asarray(chunks["mode"]), return_counts=True)
    return {CHUNK_MODES.get(int(m), f"mode {int(m)}"): int(c)
            for m, c in zip(modes, counts)}


def summarize(rec: dict) -> str:
    m = rec["manifest"]
    lines = [f"{rec['dir'].name}   created {m['created']}"]
    cfg = m["deploy_config"]
    lines.append(f"  {'sync' if cfg['sync'] else 'async'}  "
                 f"P={cfg['P']}  H={cfg['H']}  {cfg['control_hz']} Hz")
    lines.append(f"  checkpoint: {cfg['checkpoint'] or '(none — EchoPolicy)'}")
    modes = chunk_modes(rec.get("chunks"))
    if modes:
        lines.append("  actions: " + ", ".join(f"{m} ({c} chunks)"
                                               for m, c in modes.items()))
    # Both of these are recorded because a run once could not answer them.
    if "warm" in m:
        lines.append(f"  warm at start: {bool(m['warm'])}"
                     + ("" if m["warm"] else "   <-- the first inference paid "
                                             "the compile"))
    code = m.get("code") or {}
    if code.get("sha"):
        dirty = f", {code.get('dirty_file_count', 0)} file(s) modified" \
            if code.get("dirty") else ""
        lines.append(f"  code: {code['sha'][:12]}{dirty}")
    lines.append("")
    for name, spec in m["streams"].items():
        lines.append(f"  {name:14} {spec['records']:>8} records")
    for name, cam in m.get("cameras", {}).items():
        drop = cam["dropped"]
        flag = "" if not drop else f"   <-- {drop} DROPPED"
        lines.append(f"  {name:14} {cam['frames']:>8} frames{flag}")
    lines.append("")
    lines.append(_ik_residual(rec))
    return "\n".join(lines)


def _ik_residual(rec: dict) -> str:
    """What the IK could not reach, per tick (`body_targets["err"]`).

    Reported by default rather than behind a flag, because the failure it
    describes is invisible in every other column: an unreachable target comes
    out of the QP as a perfectly ordinary joint angle, and the arm simply stops
    short. It has run to 82 mm for the last third of a rollout with nothing
    else saying so.

    NaN means no IK produced the tick — homing and the exit ramp are P0's — and
    those are counted separately rather than dropped, so "the record has almost
    no solves in it" cannot look like "the solves were all fine".
    """
    body = rec.get("body_targets")
    if body is None or not len(body) or "err" not in body.dtype.names:
        return "  ik residual: not recorded (this record predates it)"
    err = np.asarray(body["err"], dtype=float)
    solved = err[~np.isnan(err)]
    if not len(solved):
        return f"  ik residual: no solved ticks ({len(err)} written by P0)"
    mm = solved * 1e3
    worst = f"{mm.max():.1f} mm"
    flag = "" if mm.max() < 10.0 else "   <-- the arm did not reach"
    return (f"  ik residual, {len(solved)} solves: "
            f"p50 {np.percentile(mm, 50):.2f}  p99 {np.percentile(mm, 99):.2f}  "
            f"max {worst}{flag}")


def homing_report(rec: dict) -> str:
    """Did the arm ramp to home, or step to it?"""
    body = rec["body_targets"]
    if not len(body):
        return "no body targets recorded"
    # `k <= 0`, not `k < 0`: `home_ticks` runs `[1-n … 0]` and tick 0 is the
    # LAST homing tick — the one that lands exactly on home_q. The run proper
    # starts at k = 1, so nothing after homing is swept up by this.
    home = body[body["k"] <= 0]
    if not len(home):
        return ("no ticks with k <= 0 — homing was not recorded. Either the "
                "record starts after HOME, or the arm was already at home_q.")

    home = home[np.argsort(home["k"])]
    lines = [
        f"homing: {len(home)} ticks, k = {home['k'][0]} … {home['k'][-1]}",
        f"  commanded span   {home['deadline'][-1] - home['deadline'][0]:.3f} s",
    ]
    if len(home) < 2:
        # `cosine_ramp` floors at one sample, so this is reachable rather than
        # hypothetical — and a one-tick ramp IS a step, by construction.
        lines.append("  a single tick: this is a step by construction, not a "
                     "ramp. Check home_min_s.")
        return "\n".join(lines)
    gaps = np.diff(home["deadline"])
    lines.append(f"  tick spacing     {gaps.min()*1e3:.2f} … {gaps.max()*1e3:.2f} ms "
                 f"(median {np.median(gaps)*1e3:.2f})")

    # How far the COMMAND actually travelled, per joint. A ramp that spans 3 s
    # but whose targets are all identical is not a ramp.
    q = np.stack([np.asarray(r) for r in home["q"]])
    travel = np.abs(q[-1] - q[0])
    lines.append(f"  commanded travel max {travel.max():.4f} rad "
                 f"on joint {int(np.argmax(travel))}")

    state = rec.get("state")
    if state is None or not len(state):
        lines.append("  (no state stream — cannot say what the arm DID)")
        return "\n".join(lines)

    idx = body_joint_index(rec["manifest"])
    if idx is None:
        lines.append("  (no joint map — cannot line the two streams up)")
        return "\n".join(lines)

    t0, t1 = float(home["deadline"][0]), float(home["deadline"][-1])
    during = state[(state["t_recv"] >= t0) & (state["t_recv"] <= t1)]
    lines.append(f"  measured samples in that window: {len(during)}")
    if len(during) < 3:
        lines.append("  too few to judge — was P1 producing?")
        return "\n".join(lines)

    qm = np.stack([np.asarray(r) for r in during["q"]])[:, idx]
    moved = np.abs(qm.max(axis=0) - qm.min(axis=0))
    j = int(np.argmax(moved))
    lines.append(f"  measured travel  max {moved.max():.4f} rad on joint {j} "
                 f"(commanded {travel[j]:.4f} on that joint)")

    # WHEN it moved is the whole question. Normalised cumulative motion of the
    # busiest joint against normalised time: a cosine ramp crosses half its
    # travel at the midpoint, a step crosses it almost immediately.
    trace = qm[:, j]
    span = trace.max() - trace.min()
    if span < 1e-3:
        lines.append("  the arm barely moved — it was already at home_q")
        return "\n".join(lines)

    progress = np.abs(trace - trace[0]) / span
    t_rel = (during["t_recv"] - t0) / max(1e-9, t1 - t0)
    half = float(t_rel[np.searchsorted(progress >= 0.5, True)]
                 if (progress >= 0.5).any() else np.nan)
    lines.append(f"  half the motion done at t = {half:.2f} of the window")
    if np.isnan(half):
        verdict = "never reached halfway — the arm did not follow"
    elif half < 0.2:
        verdict = ("STEP — the motion was over before the ramp was. P7 is not "
                   "interpolating these targets, or they all arrived at once")
    elif half > 0.8:
        verdict = "LATE — the arm lagged the ramp badly"
    else:
        verdict = "RAMP — the arm tracked the ramp as intended"
    lines.append(f"  verdict: {verdict}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--homing", action="store_true",
                    help="did the arm ramp to home, or step to it?")
    args = ap.parse_args(argv)

    rec = load(args.run_dir)
    print(summarize(rec))
    if args.homing:
        print()
        print(homing_report(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

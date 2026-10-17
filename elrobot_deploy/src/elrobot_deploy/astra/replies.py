"""The text Astra reads: the "Current observation" section (the opening message and every
`send_action` reply) and an "Action N result" section.

Labels say what a number is ("virtual TCP, base frame", "measured at deadline"); what they
MEAN (the frames, the rotation order, deadline ticks) is explained once, in the session's
README.md, not in every reply. The same code builds the opening observation and every later
one.
"""

from __future__ import annotations


def vec(v, nd: int, width: int = 0) -> str:
    return "[" + ", ".join(f"{float(x):{width}.{nd}f}" for x in v) + "]"


def pose_line(xyz, rpy, gripper_mm) -> str:
    return (f"position {vec(xyz, 3, 6)} m   rotation (roll, pitch, yaw) {vec(rpy, 1, 6)}°   "
            f"gripper opening {gripper_mm:.0f} mm")


def short_pose(d: dict) -> str:
    return (f"{vec(d['xyz'], 3)} m  {vec(d['rpy'], 0)}°  "
            f"{d['gripper_mm']:3.0f} mm")


def now_section(row: dict, cameras) -> str:
    """`row`: one tick of the in-memory timeline (`harness.Ingest.ticks`)."""
    return "\n".join([
        f"# Current observation (tick {row['tick']}, t={row['t']:.2f} s)",
        f"[image observation_now.jpg: {' | '.join(cameras)}]",
        "",
        f"Virtual TCP, base frame:  {pose_line(row['meas_xyz'], row['meas_rpy'], row['meas_gripper_mm'])}",
        "",
        ("Your next action's waypoints are offsets from the arm's last commanded pose, in its "
         "TCP frame."),
    ])


def result_section(out: dict, cameras, step_dir: str) -> str:
    """`out`: an action's `execution_outcome.json`."""
    a = out["action"]
    cols = ", ".join(str(t) for t in out["history_ticks"])
    lines = [
        (f"# Action {a} result (tick {out['tick_start']} → {out['tick_end']}, "
         f"{out['duration_s']:.1f} s)"),
        f"[image observation_history.jpg: rows {', '.join(cameras)}; columns tick {cols}]",
        "",
    ]
    if out.get("interrupted"):
        lines += [f"INTERRUPTED: {out['interrupted']}. The arm holds where it stopped.", ""]
    lines.append(f" {'wp':>2}  {'deadline tick':<14}  {'commanded':<44}  measured at deadline")
    for w in out["waypoints"]:
        if w.get("deadline_tick") is None:
            lines.append(f" {w['wp']:>2}  {'(not reached)':<14}  {short_pose(w['commanded']):<44}")
            continue
        lines.append(f" {w['wp']:>2}  {w['deadline_tick']:<14}  {short_pose(w['commanded']):<44}"
                     f"  {short_pose(w['measured'])}")
    r = out.get("max_ik_residual")
    if r:
        lines.append(f"Largest IK residual: {r['mm']:.1f} mm / {r['deg']:.1f}° at tick "
                     f"{r['tick']}.")
    lines += [
        "",
        "For more detail:",
        f"  this action's full result:  {step_dir}/execution_outcome.json",
        (f"  per-tick data:              sqlite3 timeline.sqlite \"SELECT tick, t, meas_z_m, "
         f"cmd_z_m FROM timeline WHERE action_id = {a}\""),
    ]
    return "\n".join(lines)


def error_text(problems, *, moved: bool = False) -> str:
    head = "send_action refused" if not moved else "send_action failed"
    body = "\n".join(f"  - {p}" for p in problems)
    tail = "The robot did not move." if not moved else "The arm holds where it stopped."
    return f"{head}:\n{body}\n{tail}"


__all__ = ["error_text", "now_section", "pose_line", "result_section", "short_pose", "vec"]

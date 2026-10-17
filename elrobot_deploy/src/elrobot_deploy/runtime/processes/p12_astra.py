"""P12: the Astra source. Timed waypoint paths in (from the harness), operator rows out.

Started by P0 at CHECK of an `astra` session, in place of the phone source (P11): the two are
never in one session, because each is the only writer of `control.intervene`. Runs until
STOPPED. It drives exactly the way the phone does, through the intervention ring (P6 composes
each row with the TCP pose it freezes at the counter change, solves the IK, the bus process
executes), so nothing downstream knows whether a person or an agent is driving.

ONE ACTION = ONE ENGAGEMENT

    1. compose   Astra's waypoints are offsets from the REFERENCE pose: the last composed
                 waypoint of the previous action, or, for a rollout's first action (or after an
                 interrupted one), `fk` of the command the bus process holds. `astra.trajectory`
                 times the path; `JointCheck` solves it with the IK first and stretches every
                 segment whose joint speeds would pass `joint_speed_margin` x the limit.
    2. press     the counter goes ODD. P6 freezes its anchor: `fk` of the held command, the
                 same configuration read here, so a row's `delta` (target relative to the
                 anchor) lands on the composed pose exactly.
    3. probe     rows at rest (the anchor itself) until P6 has published a target of this
                 world: proof that it is taking rows before any of them moves the arm.
    4. stream    one row per control tick, each written `lead_s` before its deadline (P6 takes
                 the NEWEST row, so a path written all at once would jump to its end).
    5. settle    the last pose for `settle_s`: the bus process's filter converges onto it.
    6. release   the counter goes EVEN: the arm holds its last command until the next action.

The arm therefore never moves between actions (Astra thinking = a held arm), and a release
can be forced at any instant (the session left ACTIVE, the bus process stopped following):
the counter goes even and the arm holds where it is.

THE SOCKET: one Unix socket (path from P0), JSON lines, the harness the only client:

    -> {"op": "action", "id": 7, "waypoints": [...], "speed": "normal"}
    <- {"op": "planned", "id": 7, "t0": ..., "dt": ..., "waypoints": [{pose, gripper_mm,
        deadline}], "ref": {...}, "t_end": ...}   or   {"op": "error", "id": 7, "error": ...}
    <- {"op": "done", "id": 7, "t_release": ..., "interrupted": null | "why"}
"""

from __future__ import annotations

import json
import math
import os
import socket

import numpy as np

from elrobot_deploy.astra.frames import GripperMap
from elrobot_deploy.astra.trajectory import Limits, Plan, Reference, build, parse_action
from elrobot_deploy.robot.transforms import quat_from_mat
from elrobot_deploy.runtime import timing as clock
from elrobot_deploy.runtime.shared_memory import Arena, layout_for, open_all, schema

#: How long the probe waits for P6 to take a row before the action is refused.
P6_WAIT_S = 5.0
#: A planning round may ask a segment to grow by at most this (an IK branch flip asks for
#: absurd factors; it then shows as residual instead).
MAX_STRETCH = 4.0
#: The socket's poll period while idle (how fast STOPPED is noticed).
ACCEPT_S = 0.2


class JointCheck:
    """`stretch` for `trajectory.build`: solve the sampled path with the IK, uncapped, seeded
    from the anchor, and say per segment how much longer it must take for every joint to stay
    under `margin x qdot_max`. Only for the shipped `tcp_qp` solver (a custom one is trusted
    with its own caps)."""

    def __init__(self, cfg, *, margin: float):
        from elrobot_deploy.robot.elrobot_model import ElRobotModel
        from elrobot_deploy.runtime.ik.tcp_qp import TcpSolver
        options = dict(cfg.ik.options)
        options.pop("nominal_q", None)
        self.solver = TcpSolver(ElRobotModel(cfg.urdf), dt=cfg.dt,
                                qdot_max=np.full(7, 1e3), **options)
        self.limit = np.asarray(cfg.qdot_max[:schema.ARM_DOF], np.float64) * float(margin)
        self.seed = None
        self.worst = (0.0, -1)           # the last check's largest ratio and its joint

    def __call__(self, plan: Plan) -> np.ndarray:
        q = np.asarray(self.seed, np.float64)
        self.solver.set_nominal(q)
        qs = [q]
        for i in range(1, plan.n):
            pose = np.r_[plan.p[i], quat_from_mat(plan.R[i])]
            q, _, _ = self.solver.solve(pose, q, budget=0.05)
            qs.append(q)
        qd = np.abs(np.diff(np.asarray(qs), axis=0)) / plan.dt          # (N-1, 7)
        ratio = qd / self.limit
        per_sample = ratio.max(axis=1)
        seg = plan.segment_of()[1:]
        need = np.ones(len(plan.seg_s))
        np.maximum.at(need, seg, per_sample)
        j = int(np.argmax(ratio.max(axis=0)))
        self.worst = (float(ratio.max()), j)
        return np.minimum(need, MAX_STRETCH)


class AstraSource:
    """Everything P12 does, without the socket (the tests drive it)."""

    def __init__(self, rings, cfg, *, fk=None, joint_check=None, log=print):
        self.cfg = cfg
        self.control = rings["control"]
        self.rows = rings["intervene"]
        self.state = rings["state"]
        self.targets = rings["targets"]
        self.log = log
        self.limits = Limits.from_config(cfg)
        self.gripper = GripperMap.from_config(cfg)
        if fk is None:
            from elrobot_deploy.robot.elrobot_model import ElRobotModel
            model = ElRobotModel(cfg.urdf)
            fk = model.fk
        #: q7 -> 4x4 TCP pose, base_link (the model P6's solver uses).
        self.fk = fk
        self.joint_check = joint_check
        self.ref: Reference | None = None
        self.counter = self.control.intervene
        if self.counter % 2:
            self._bump()                  # a previous source died engaged: hand back

    # --- the counter ---------------------------------------------------------------------

    def _bump(self) -> None:
        self.counter += 1
        self.control.intervene = self.counter

    def release(self) -> None:
        if self.counter % 2:
            self._bump()
            self.rows.note(engaged=0)

    # --- one action ----------------------------------------------------------------------

    def _held(self) -> np.ndarray | None:
        snap = self.state.latest()
        if snap is None or not np.all(np.isfinite(snap["q_cmd"])):
            return None
        return np.asarray(snap["q_cmd"], np.float64).copy()

    def why_not(self) -> str | None:
        """Why an action cannot run now, or None."""
        session = self.control.session
        if session != schema.ACTIVE:
            return (f"the robot is not in a rollout (session {schema.SESSION_NAMES.get(session)}"
                    f"): nothing moves until the operator presses Start")
        bus = self.control.bus
        if bus not in (schema.BUS_RUNNING, schema.BUS_DRY):
            return (f"the bus process is {schema.BUS_NAMES.get(bus, '?')}"
                    f" ({self.control.fault()[1] or 'stopped'}): the arm is not following")
        return None

    def plan(self, args) -> tuple[Plan, np.ndarray]:
        """Check, compose and time one action. -> (plan, the held command it anchors on)."""
        waypoints, speed, errors = parse_action(args, self.limits)
        if errors:
            raise ValueError("; ".join(errors))
        held = self._held()
        if held is None:
            raise RuntimeError("no state rows: the bus process is not reading the arm")
        if self.ref is None:
            T = self.fk(held[:schema.ARM_DOF])
            self.ref = Reference(T[:3, 3].copy(), T[:3, :3].copy(),
                                 float(self.gripper.mm(held[schema.GRIPPER])))
        stretch = None
        if self.joint_check is not None:
            self.joint_check.seed = held[:schema.ARM_DOF]
            stretch = self.joint_check
        plan = build(self.ref, waypoints, self.limits, self.cfg.dt, speed=speed,
                     stretch=stretch)
        return plan, held

    def execute(self, plan: Plan, held: np.ndarray, on_start=None) -> dict:
        """Press, probe, stream, settle, release. Blocks for the whole motion. `on_start(t0)`
        is called once the first sample's deadline is fixed. -> what happened."""
        dt, lead = self.cfg.dt, self.cfg.teleop.lead_s
        T_a = self.fk(held[:schema.ARM_DOF])
        p_a, R_a = T_a[:3, 3].copy(), T_a[:3, :3].copy()
        self._bump()                                      # odd: we drive
        self.rows.note(engaged=1)
        counter = self.counter
        t_grid = clock.now() + lead + dt
        n = 0
        why = None

        def write(i_grid: int, p, R, g_mm) -> bool:
            nonlocal why
            deadline = t_grid + i_grid * dt
            clock.sleep_until(deadline - lead)
            why = self.why_not()
            if why is not None:
                return False
            delta = np.r_[p - p_a, quat_from_mat(R @ R_a.T)]
            self.rows.write(intervene=counter, deadline=deadline, delta=delta,
                            grip=float(self.gripper.q(g_mm)), t_device=deadline - lead,
                            t_recv=clock.now(), engage=1, grip_button=0)
            return True

        try:
            # The probe: rest rows until P6 has published one of this world.
            give_up = clock.now() + P6_WAIT_S
            while True:
                if not write(n, plan.p[0], plan.R[0], plan.g[0]):
                    return self._done(None, why)
                n += 1
                newest = self.targets.newest_full()
                if newest is not None and int(newest["intervene"]) == counter:
                    break
                if clock.now() > give_up:
                    return self._done(None, f"P6 took no target within {P6_WAIT_S:.0f} s "
                                            f"(the IK process is not running?)")
            t0 = t_grid + n * dt
            if on_start is not None:
                on_start(t0)
            for i in range(plan.n):
                if not write(n + i, plan.p[i], plan.R[i], plan.g[i]):
                    return self._done(t0, why)
            n += plan.n
            for _ in range(math.ceil(self.cfg.astra.settle_s / dt)):
                if not write(n, plan.p[-1], plan.R[-1], plan.g[-1]):
                    return self._done(t0, why)
                n += 1
            clock.sleep_until(t_grid + (n - 1) * dt)      # the last row has come due
            p, R, g = plan.targets[-1]
            self.ref = Reference(p.copy(), R.copy(), float(g))
            return self._done(t0, None)
        finally:
            self.release()

    def _done(self, t0, why) -> dict:
        self.release()
        if why is not None:
            self.ref = None                   # re-anchor on the held command next time
        return {"t0": t0, "t_release": clock.now(), "interrupted": why}

    def forget(self) -> None:
        """A rollout ended: the next one starts from the held command."""
        self.ref = None


def _pose_list(p, R) -> list:
    return np.r_[p, quat_from_mat(R)].tolist()


def handle(source: AstraSource, msg: dict, send) -> None:
    """One request from the harness."""
    if msg.get("op") == "ping":
        send({"op": "pong"})
        return
    if msg.get("op") != "action":
        send({"op": "error", "id": msg.get("id"), "error": f"unknown op {msg.get('op')!r}"})
        return
    aid = msg.get("id")
    why = source.why_not()
    if why is not None:
        send({"op": "error", "id": aid, "error": why})
        return
    t_plan = clock.now()
    try:
        plan, held = source.plan({"waypoints": msg.get("waypoints"),
                                  "speed": msg.get("speed", "normal")})
    except (ValueError, RuntimeError) as exc:
        send({"op": "error", "id": aid, "error": str(exc)})
        return
    note = ""
    if source.joint_check is not None:
        ratio, j = source.joint_check.worst
        note = f", joint {j + 1} at {ratio * 100:.0f} % of its margin"
    source.log(f"[P12] action {aid}: {len(plan.targets)} waypoints, {plan.duration:.2f} s "
               f"(planned in {(clock.now() - t_plan) * 1e3:.0f} ms{note})")

    def on_start(t0: float) -> None:
        ref = plan.ref
        send({"op": "planned", "id": aid, "t0": t0, "dt": plan.dt, "n": plan.n,
              "t_end": t0 + (plan.n - 1) * plan.dt, "duration_s": plan.duration,
              "ref": {"pose": _pose_list(ref.p, ref.R), "gripper_mm": ref.gripper_mm},
              "waypoints": [{"index": i, "pose": _pose_list(p, R), "gripper_mm": g,
                             "deadline": t0 + plan.knots[i] * plan.dt}
                            for i, (p, R, g) in enumerate(plan.targets, start=1)]})

    result = source.execute(plan, held, on_start=on_start)
    if result["t0"] is None:
        send({"op": "error", "id": aid, "error": result["interrupted"]})
        return
    send({"op": "done", "id": aid, **result})


def run(shm_name: str, cfg, *, socket_path: str, ready=None) -> None:
    """P12 as a process. Runs until the session reaches STOPPED. `ready` (P0's latch) is set
    once the model is loaded and the socket listens: CHECK waits for it."""
    arena = Arena.attach(shm_name, layout_for(cfg))
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    source = None
    try:
        rings = open_all(arena)
        control = rings["control"]
        check = None
        if cfg.ik.solver == "tcp_qp":
            check = JointCheck(cfg, margin=cfg.astra.joint_speed_margin)
        source = AstraSource(rings, cfg, fk=check.solver.robot.fk if check else None,
                             joint_check=check, log=lambda s: print(s, flush=True))
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        server.bind(socket_path)
        server.listen(1)
        server.settimeout(ACCEPT_S)
        print(f"[P12] the Astra source is listening on {socket_path}", flush=True)
        if ready is not None:
            ready.set()
        while control.session < schema.STOPPED:
            if control.session != schema.ACTIVE:
                source.forget()
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            _serve(conn, source, control)
    finally:
        if source is not None:
            source.release()
        server.close()
        try:
            os.unlink(socket_path)
        except OSError:
            pass
        arena.close()


def _serve(conn, source: AstraSource, control) -> None:
    conn.settimeout(ACCEPT_S)
    buf = b""

    def send(obj) -> None:
        conn.sendall(json.dumps(obj).encode() + b"\n")

    try:
        while control.session < schema.STOPPED:
            if control.session != schema.ACTIVE:
                source.forget()
            try:
                chunk = conn.recv(65536)
            except TimeoutError:
                continue
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    try:
                        msg = json.loads(line)
                    except ValueError as exc:
                        send({"op": "error", "error": f"bad JSON: {exc}"})
                        continue
                    handle(source, msg, send)
    except (BrokenPipeError, ConnectionResetError):
        return
    finally:
        conn.close()


__all__ = ["AstraSource", "JointCheck", "handle", "run"]

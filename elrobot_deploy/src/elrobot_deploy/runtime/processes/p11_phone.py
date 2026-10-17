"""P11: the phone source. UDP packets from the iPhone in, operator rows into the arena.

Started by P0 at CHECK; runs until STOPPED. A blocking receive (no polling delay): each
datagram is stamped on arrival, parsed (`teleop/phone.py`), echoed back, and, while the clutch
is held during ACTIVE, turned into one row of the intervention ring for P6:

    clutch      `Console`: the engage button, held. A press re-anchors everything.
    motion      `RoomFrame`: the phone's motion since the press, in the robot's axes
                (room-frame: forward is where the phone's top pointed at the press).
    gripper     `Gripper`: hold-to-close on the phone's clock, squeeze-bounded against the
                measured jaw (the state ring), stored bounded.
    deadline    `Pace`: capture time + lead, re-pinned at every press.

THE COUNTER. P11 is the only writer of `control.intervene`: it bumps it to ODD at a press and
to EVEN at a release. Every bump abandons the bus process's segment in flight (so a release
stops the arm at once: it holds where it was) and P6's latched state.

THE GUARDS: each forces a release, and a release forced by a guard needs the clutch to be
let go and pressed again (`Console.release`):

    tracking    the phone's tracking state is not "normal"
    world       `world_epoch` changed (world origin reset, relocalization)
    jump        one frame moved more than `max_step_m` / `max_step_deg` (a tracking glitch)
    stale       no packet for `stale_s` (WiFi, the app backgrounded, the phone locked)
    heading     the phone's top pointed (nearly) straight up or down at the press
    session     the rollout ended (RESET, exit)

A release means HOLD: there is no policy behind the operator, so the arm keeps its last command.
The workspace box is P6's (it needs the TCP pose).

Bench mode, no robot, to check the grip before anything moves:

    uv run python -m elrobot_deploy.runtime.processes.p11_phone --bench

prints, while the clutch is held, the phone's motion since the press in the operator's forward /
left / up, and its rotation about those axes. Push the phone forward: "fwd" must grow.
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time

import numpy as np

from elrobot_deploy.robot.transforms import mat_from_quat, quat_angle, quat_from_mat
from elrobot_deploy.runtime import timing as clock
from elrobot_deploy.runtime.shared_memory import Arena, build, open_all, schema
from elrobot_deploy.runtime.teleop import phone
from elrobot_deploy.runtime.teleop.intervention import (
    Console,
    Gripper,
    HeadingUndefined,
    Pace,
    RoomFrame,
)

#: The socket's receive timeout: how often the loop checks the session word and staleness when
#: no packet arrives.
RECV_TIMEOUT_S = 0.02
#: A sequence number this far below the last one is a restarted app, not a reordered packet.
SEQ_RESTART = 10_000


class PhoneLoop:
    """Everything P11 does with one packet, without the socket (the tests drive it)."""

    def __init__(self, rings, cfg, *, log=print):
        self.cfg = cfg
        t = cfg.teleop
        self.control = rings["control"]
        self.rows = rings["intervene"]
        self.state = rings["state"]
        self.targets = rings["targets"]
        self.log = log
        self.console = Console()
        self.room = RoomFrame(t.phone_forward, t.base_forward, t.base_up, scale=t.scale)
        g = schema.GRIPPER
        self.gripper = Gripper(t.gripper_rate, cfg.limits.lower[g], cfg.limits.upper[g],
                               squeeze_margin=cfg.executor.squeeze_margin)
        self.pace = Pace(t.lead_s)
        self.max_step_m, self.max_step = t.max_step_m, t.max_step
        self.stale_s = t.stale_s
        # The counter: ours alone. A previous P11 that died mid-engagement left it odd; hand back.
        self.counter = self.control.intervene
        if self.counter % 2:
            self._bump()
        self.last: phone.PhonePacket | None = None      # the previous packet, any state
        self.t_last_recv: float | None = None
        self.epoch: int | None = None
        self.why = ""                                     # the last forced release's reason
        self._forced = False                              # released by a guard, button still down
        self.samples = 0
        self.released = 0
        # Per engagement: (t_recv - t_device) of every row, for the latency report.
        self._lat: list[float] = []

    # --- the counter ---------------------------------------------------------------------

    def _bump(self) -> None:
        self.counter += 1
        self.control.intervene = self.counter

    @property
    def engaged(self) -> bool:
        return self.console.engaged

    # --- releases ------------------------------------------------------------------------

    def release(self, why: str) -> None:
        """Force a release (a guard, staleness, the session). The arm holds."""
        if self.console.release() == -1:
            self._bump()
            self._report()
        self.why = why
        self._forced = True
        self.released += 1
        self.rows.note(engaged=0, released=self.released)
        self.log(f"[P11] released: {why}")

    def _report(self) -> None:
        if len(self._lat) > 1:
            jitter = (np.asarray(self._lat) - self._lat[0]) * 1e3
            self.log(f"[P11] engagement: {len(self._lat)} rows, latency vs the first "
                     f"p50 {np.percentile(jitter, 50):+.1f} ms, p99 "
                     f"{np.percentile(jitter, 99):+.1f} ms, max {jitter.max():+.1f} ms; "
                     f"the deadline map slid {self.pace.slid * 1e3:.1f} ms")
        self._lat = []

    # --- one packet ----------------------------------------------------------------------

    def on_packet(self, pkt: phone.PhonePacket, t_recv: float) -> str:
        """Handle one packet. Returns the echo's `state`."""
        last = self.last
        if last is not None and pkt.seq <= last.seq and last.seq - pkt.seq < SEQ_RESTART:
            return self._state()                          # reordered / duplicated: drop it
        self.samples += 1
        self.t_last_recv = t_recv
        self.last = pkt
        epoch_changed = self.epoch is not None and pkt.world_epoch != self.epoch
        self.epoch = pkt.world_epoch
        self.rows.note(samples=self.samples, t_last_sample=t_recv, fresh=1,
                       track=pkt.track, world_epoch=pkt.world_epoch)

        active = self.control.session == schema.ACTIVE
        if self.engaged and pkt.engage:       # a packet with the clutch up is a plain release
            why = None
            if not active:
                why = "the rollout ended"
            elif pkt.track != phone.TRACK_NORMAL:
                why = f"tracking {pkt.track_text or 'lost'}"
            elif epoch_changed:
                why = "the phone's world origin changed"
            elif last is not None and (pkt.seq <= last.seq):
                why = "the phone app restarted"
            elif last is not None and self._jumped(last, pkt):
                why = "the phone pose jumped (a tracking glitch)"
            if why is not None:
                self.release(why)

        if not pkt.engage:
            self._forced = False
        if not active or pkt.track != phone.TRACK_NORMAL:
            # The clutch does nothing outside a rollout or without tracking; a press held
            # through it must be let go first.
            if pkt.engage and not self.engaged:
                self.console.release()
            return self._state()

        edge = self.console.update(pkt.engage)
        if edge == 1:
            self._press(pkt, t_recv)
        elif edge == -1:
            self._bump()
            self._report()
            self.rows.note(engaged=0)
        if self.engaged:
            self._row(pkt, last, t_recv)
        return self._state()

    def _jumped(self, a: phone.PhonePacket, b: phone.PhonePacket) -> bool:
        if float(np.linalg.norm(b.T[:3, 3] - a.T[:3, 3])) > self.max_step_m:
            return True
        R = a.T[:3, :3].T @ b.T[:3, :3]
        return quat_angle(quat_from_mat(R)) > self.max_step

    def _press(self, pkt: phone.PhonePacket, t_recv: float) -> None:
        try:
            self.room.freeze(pkt.T)
        except HeadingUndefined as exc:
            self.console.release()
            self.why, self._forced = f"{exc}: point it forward and press again", True
            self.log(f"[P11] not engaged: {self.why}")
            return
        self.pace.reset()
        snap = self.state.latest()
        jaw = float(snap["q_cmd"][schema.GRIPPER]) if snap is not None else self.gripper.lo
        self.gripper.engage(jaw)
        self.why = ""
        self._lat = []
        self._bump()                                      # odd: the operator drives
        self.rows.note(engaged=1)

    def _row(self, pkt: phone.PhonePacket, last, t_recv: float) -> None:
        dt = 0.0 if self._lat == [] or last is None else pkt.t - last.t
        snap = self.state.latest()
        measured = float(snap["q"][schema.GRIPPER]) if snap is not None else None
        grip = self.gripper.step(pkt.grip, dt, measured)
        deadline = self.pace.due(pkt.t, t_recv)
        self._lat.append(t_recv - pkt.t)
        self.rows.write(intervene=self.counter, deadline=deadline, delta=self.room.delta(pkt.T),
                        grip=grip, t_device=pkt.t, t_recv=t_recv, engage=1,
                        grip_button=int(pkt.grip))
        lat = np.asarray(self._lat[-120:])
        self.rows.note(latency_s=float(lat[-1] - lat.min()))

    def _state(self) -> str:
        if self.control.session != schema.ACTIVE:
            return "inactive"
        if self.engaged:
            return "engaged"
        return "released" if self._forced else "idle"

    # --- no packet -----------------------------------------------------------------------

    def on_idle(self, t_now: float) -> None:
        """Called when the socket timed out (or between packets): staleness and the session."""
        fresh = self.t_last_recv is not None and t_now - self.t_last_recv < self.stale_s
        self.rows.note(fresh=int(fresh))
        if self.engaged:
            if not fresh:
                self.release(f"no packet from the phone for {self.stale_s * 1e3:.0f} ms")
            elif self.control.session != schema.ACTIVE:
                self.release("the rollout ended")

    def close(self) -> None:
        """Leave on an EVEN counter (hand back), whatever happened."""
        if self.engaged:
            self.release("the phone source is stopping")
        if self.counter % 2:
            self._bump()


def run(shm_name: str, cfg) -> None:
    """P11 as a process. Runs until the session reaches STOPPED."""
    arena = Arena.attach(shm_name, build())
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rings = open_all(arena)
        sock.bind(("0.0.0.0", int(cfg.teleop.port)))
        sock.settimeout(RECV_TIMEOUT_S)
        loop = PhoneLoop(rings, cfg)
        print(f"[P11] listening on UDP {cfg.teleop.port} ({_local_ip()})", flush=True)
        control = rings["control"]
        bad = 0
        try:
            while control.session < schema.STOPPED:
                try:
                    data, addr = sock.recvfrom(65535)
                except TimeoutError:
                    loop.on_idle(clock.now())
                    continue
                t_recv = clock.now()
                try:
                    pkt = phone.parse(data)
                except phone.BadPacket as exc:
                    bad += 1
                    if bad in (1, 10, 100) or bad % 1000 == 0:
                        print(f"[P11] bad packet #{bad} from {addr[0]}: {exc}", flush=True)
                    continue
                state = loop.on_packet(pkt, t_recv)
                try:
                    sock.sendto(phone.echo(pkt, t_mac=t_recv, state=state, why=loop.why,
                                           intervene=loop.counter), addr)
                except OSError:
                    pass
                loop.on_idle(t_recv)
        finally:
            loop.close()
    finally:
        sock.close()
        arena.close()


def _local_ip() -> str:
    """This machine's LAN address, for the operator to type into the app (best effort)."""
    import subprocess
    for iface in ("en0", "en1", "en2", "en3"):
        try:
            ip = subprocess.run(["ipconfig", "getifaddr", iface], capture_output=True,
                                text=True, timeout=1).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            break
        if ip:
            return ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# --- bench mode -----------------------------------------------------------------------------


def bench(cfg, *, port: int | None = None, out=sys.stdout) -> None:
    """No arena, no robot: print the phone's motion since each press, in the operator's axes."""
    port = int(port or cfg.teleop.port)
    t = cfg.teleop
    room = RoomFrame(t.phone_forward, t.base_forward, t.base_up)
    console = Console()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    print(f"bench: listening on UDP {port} ({_local_ip()}). Hold the phone as you will drive "
          f"(screen left, top forward), hold ENGAGE, and move it. Ctrl-C to stop.", file=out)
    B = np.c_[t.base_forward, np.cross(t.base_up, t.base_forward), t.base_up]
    t_print = 0.0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except TimeoutError:
                continue
            try:
                pkt = phone.parse(data)
            except phone.BadPacket as exc:
                print(f"bad packet: {exc}", file=out)
                continue
            sock.sendto(phone.echo(pkt, t_mac=clock.now(),
                                   state="engaged" if console.engaged else "idle"), addr)
            edge = console.update(pkt.engage and pkt.track == phone.TRACK_NORMAL)
            if edge == 1:
                try:
                    room.freeze(pkt.T)
                    print("-- pressed: anchored", file=out)
                except HeadingUndefined as exc:
                    console.release()
                    print(f"-- not anchored: {exc}", file=out)
            elif edge == -1:
                print("-- released", file=out)
            now = time.monotonic()
            if console.engaged and now - t_print > 0.1:
                t_print = now
                fwd, left, up = room.axes(pkt.T) * 1e3
                d = room.delta(pkt.T)
                rv = _rotvec(B.T @ mat_from_quat(d[3:]) @ B)        # about forward, left, up
                print(f"fwd {fwd:+7.1f}  left {left:+7.1f}  up {up:+7.1f} mm   "
                      f"rot about fwd {rv[0]:+6.1f}  left {rv[1]:+6.1f}  up {rv[2]:+6.1f} deg   "
                      f"grip {'ON ' if pkt.grip else 'off'}  track {pkt.track_text}",
                      file=out, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def _rotvec(R) -> np.ndarray:
    """A rotation matrix as axis x angle, degrees."""
    q = quat_from_mat(R)
    angle = quat_angle(q)
    s = math.sin(angle / 2)
    axis = np.zeros(3) if s < 1e-9 else q[1:] / s
    return np.degrees(axis * angle)


def main(argv=None) -> int:
    from elrobot_deploy.config_loading.config import load
    ap = argparse.ArgumentParser(description="The phone source's bench mode (no robot).")
    ap.add_argument("--bench", action="store_true", required=True)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--profile", default=None)
    args = ap.parse_args(argv)
    bench(load(args.profile), port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["PhoneLoop", "bench", "run"]

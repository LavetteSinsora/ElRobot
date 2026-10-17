"""The whole teleop chain on simulated servos: phone packets over real UDP -> P11 -> P6 (the real
IK) -> the bus process, under P0. One session of ~10 s."""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest

from elrobot_deploy.config_loading.config import load
from elrobot_deploy.runtime.launcher import Launcher
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.teleop.fake_phone import trajectory
from elrobot_deploy.runtime.teleop.phone import packet
from tests.conftest import arena_name

pytest.importorskip("mink")


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_the_phone_drives_the_simulated_arm():
    port = free_port()
    # The synthetic camera rides along, so the whole session (P2 included) is the real one.
    cfg = load(sets=["bus.driver=sim", f"teleop.port={port}", "poses.min_s=0.3",
                     "poses.velocity_margin=0.6"]).cameras_for(astra=False).with_cameras(fake=True)
    lines = []
    p0 = Launcher(cfg, arena_name=arena_name(), log=lines.append, auto_home=True,
                  auto_start=True)
    seen = {"states": [], "tcp": []}

    from elrobot_deploy.robot.elrobot_model import ElRobotModel
    model = ElRobotModel(cfg.urdf)     # loaded first: a stalled phone would (rightly) go stale

    def phone():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.001)
        seq, t_go = 0, None
        while True:
            t = 0.0 if t_go is None else time.monotonic() - t_go
            if t > 6.0:
                break
            T, engage, grip = trajectory(t, radius=0.03, freq=0.5, idle=0.5, seconds=4.5)
            seq += 1
            now = time.monotonic()
            sock.sendto(packet(seq, now, T, engage=engage, grip=grip), ("127.0.0.1", port))
            try:
                state = json.loads(sock.recv(65535))["state"]
                if not seen["states"] or seen["states"][-1] != state:
                    seen["states"].append(state)
                if t_go is None and state == "idle":
                    t_go = time.monotonic()
            except (TimeoutError, OSError):
                pass
            if engage and p0.rings is not None and seq % 6 == 0:
                snap = p0.rings["state"].latest()
                seen["tcp"].append(model.fk(snap["q"][:7])[:3, 3].copy())
            time.sleep(1 / 60)
        sock.close()
        seen["counter"] = p0.rings["control"].intervene
        p0.rings["control"].request = schema.REQ_EXIT

    threading.Thread(target=phone, daemon=True).start()
    assert p0.run() == 0, "\n".join(lines)
    assert seen["states"][-3:] == ["idle", "engaged", "idle"] or \
        seen["states"][-2:] == ["engaged", "idle"], seen["states"]
    tcp = np.asarray(seen["tcp"])
    span = tcp.max(0) - tcp.min(0)
    # The phone circled 3 cm in the operator's left/up plane: base x (left/right) and z (up).
    assert span[0] == pytest.approx(0.06, abs=0.012), span
    assert span[2] == pytest.approx(0.06, abs=0.012), span
    # One press and one release: the counter ended even, after at least one engagement.
    assert seen["counter"] >= 2 and seen["counter"] % 2 == 0

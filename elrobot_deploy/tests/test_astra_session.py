"""The Astra harness end to end: the MCP server over real stdio, and a whole launch on
simulated servos with synthetic cameras (P0, P2 x2, P6, P7, P9, P12 and the harness, real):
two episodes, each begun by `start` and ended by `finish`."""

from __future__ import annotations

import json
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from elrobot_deploy.astra.mcp_server import Server
from elrobot_deploy.astra.workspace import prepare
from elrobot_deploy.config_loading.config import load
from elrobot_deploy.runtime.launcher import Launcher
from elrobot_deploy.runtime.shared_memory import schema
from tests.conftest import arena_name


def _stub_harness(path: str, image: Path, seen: list) -> socket.socket:
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(4)

    def loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            req = json.loads(conn.makefile().readline())
            seen.append(req)
            conn.sendall(json.dumps({"text": f"did {req['tool']}", "images": [str(image)],
                                     "is_error": False}).encode() + b"\n")
            conn.close()

    threading.Thread(target=loop, daemon=True).start()
    return srv


def test_the_mcp_server_speaks_json_rpc_on_stdio_and_forwards_tool_calls():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:      # short: a socket path <= 104 bytes
        d = Path(d)
        sock = str(d / "h.sock")
        image = d / "x.jpg"
        image.write_bytes(b"\xff\xd8fakejpeg")
        seen: list = []
        srv = _stub_harness(sock, image, seen)
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                        "clientInfo": {"name": "codex", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "send_action", "arguments": {"waypoints": []}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "fly"}},
            {"jsonrpc": "2.0", "id": 5, "method": "resources/list"},
        ]
        proc = subprocess.run(
            [sys.executable, "-m", "elrobot_deploy.astra.mcp_server", "--socket", sock],
            input="".join(json.dumps(m) + "\n" for m in msgs), capture_output=True, text=True,
            timeout=30, check=False)
        srv.close()
    out = [json.loads(line) for line in proc.stdout.splitlines()]
    assert [o["id"] for o in out] == [1, 2, 3, 4, 5]          # no answer to a notification
    assert out[0]["result"]["protocolVersion"] == "2025-03-26"
    assert {t["name"] for t in out[1]["result"]["tools"]} == {
        "start", "send_action", "inspect_observation", "finish"}
    content = out[2]["result"]["content"]
    assert content[0] == {"type": "text", "text": "did send_action"}
    assert content[1]["type"] == "image" and content[1]["mimeType"] == "image/jpeg"
    assert out[3]["error"]["code"] == -32602 and out[4]["error"]["code"] == -32601
    assert seen == [{"tool": "send_action", "args": {"waypoints": []}}]


def test_two_astra_episodes_on_simulated_servos(tmp_path):
    pytest.importorskip("mink")
    cfg = load(sets=["bus.driver=sim", "poses.min_s=0.3", "poses.velocity_margin=0.6",
                     f'astra.workspace="{tmp_path}/astra"',
                     f'record.dir="{tmp_path}/records"']).cameras_for(astra=True).with_cameras(
        fake=True)
    with tempfile.TemporaryDirectory(dir="/tmp") as sockets:
        launch = prepare(cfg, socket_dir=sockets)
        p0 = Launcher(cfg, arena_name=arena_name(), auto_home=True, log=lambda s: None,
                      astra=launch)
        got: dict = {}

        def agent():
            try:
                mcp = Server(launch.harness_socket)

                def call(name, args):
                    r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": {"name": name, "arguments": args}})["result"]
                    return r["isError"], r["content"]

                end = time.monotonic() + 60
                while time.monotonic() < end:          # the harness is up once it answers
                    if "not reachable" not in call("send_action", {})[1][0]["text"]:
                        break
                    time.sleep(0.1)
                got["early"] = call("send_action", {"waypoints": [
                    {"xyz": [0, 0, 0], "rpy": [0, 0, 0], "gripper_mm": 10}]})
                got["start"] = call("start", {"task": "touch the table"})
                got["again"] = call("start", {"task": "another"})
                got["act"] = call("send_action", {"waypoints": [
                    {"xyz": [0.03, 0, 0], "rpy": [0, 0, 0], "gripper_mm": 40},
                    {"xyz": [0.03, 0.02, -0.02], "rpy": [0, 0, 15], "gripper_mm": 10}]})
                got["bad"] = call("send_action", {"waypoints": [
                    {"xyz": [0, 0, 0], "rpy": [0, 0, 0], "gripper_mm": 99}]})
                got["insp"] = call("inspect_observation", {"ticks": [0, 3, 6],
                                                           "cams": ["wrist"]})
                got["fin"] = call("finish", {"success": True, "summary": "touched it"})
                got["after"] = call("send_action", {"waypoints": [
                    {"xyz": [0, 0, 0], "rpy": [0, 0, 0], "gripper_mm": 10}]})
                got["start2"] = call("start", {"task": "wave"})       # waits for the homing
                got["act2"] = call("send_action", {"waypoints": [
                    {"xyz": [0, 0.02, 0], "rpy": [0, 0, 0], "gripper_mm": 55}]})
                got["fin2"] = call("finish", {"success": False, "summary": "waved"})
            finally:
                time.sleep(1.5)
                p0.rings["control"].request = schema.REQ_EXIT

        threading.Thread(target=agent, daemon=True).start()
        assert p0.run() == 0

    ws = Path(launch.workspace)
    assert (ws / "AGENTS.md").exists() and (ws / "codex_config.toml").exists()
    assert "x forward" in (ws / "AGENTS.md").read_text()
    assert not re.search(r"\$[a-z_]", (ws / "AGENTS.md").read_text())    # all filled
    assert got["early"][0] and "call `start`" in got["early"][1][0]["text"]
    err, content = got["start"]
    assert not err, content[0]["text"]
    assert [c["type"] for c in content] == ["text", "image"]
    assert "Task: touch the table" in content[0]["text"]
    assert "# Current observation" in content[0]["text"]
    assert got["again"][0] and "still running" in got["again"][1][0]["text"]

    err, content = got["act"]
    text = content[0]["text"]
    assert not err, text
    assert [c["type"] for c in content] == ["text", "image", "image"]
    assert "# Action 1 result" in text and "[image episodes/" in text
    episodes = sorted(p for p in (ws / "episodes").iterdir() if not p.is_symlink())
    assert len(episodes) == 2
    first, second = episodes
    out = json.loads((first / "actions/0001/execution_outcome.json").read_text())
    last = out["waypoints"][-1]
    # Simulated servos follow: the measured pose at the deadline is the commanded one.
    for a, b in zip(last["commanded"]["xyz"], last["measured"]["xyz"]):
        assert a == pytest.approx(b, abs=0.004)
    assert last["measured"]["gripper_mm"] == pytest.approx(10, abs=2)
    assert out["interrupted"] is None and out["tick_end"] > out["tick_start"]
    assert out["reference"]["rpy"][1] == pytest.approx(45, abs=2)      # started at home
    assert out["reference"]["xyz"][0] == pytest.approx(0.25, abs=0.01)  # 25 cm forward (+x)

    assert got["bad"][0] and "99 is outside" in got["bad"][1][0]["text"]
    assert not got["insp"][0] and len(got["insp"][1]) == 2              # one sheet of 3 ticks
    assert not got["fin"][0] and "going home" in got["fin"][1][0]["text"]
    assert got["after"][0] and "finished" in got["after"][1][0]["text"]
    assert not got["start2"][0], got["start2"][1][0]["text"]
    assert not got["act2"][0], got["act2"][1][0]["text"]
    assert not got["fin2"][0]

    for f in ("episode.json", "actions.jsonl", "actions/0001/observation_now.jpg",
              "actions/0001/observation.txt", "actions/0001/action.json",
              "actions/0002/observation_history.jpg", "actions/0002/inspections/001.jpg",
              "actions/0002/rejected.jsonl", "record/manifest.json"):
        assert (first / f).exists(), f
    e1 = json.loads((first / "episode.json").read_text())
    e2 = json.loads((second / "episode.json").read_text())
    assert (e1["task"], e1["finish"]["success"], e1["finish"]["actions"]) == (
        "touch the table", True, 1)
    assert (e2["task"], e2["finish"]["success"]) == ("wave", False)
    assert e1["record"] != e2["record"]                                 # one record each
    assert e1["clock"]["anchor_camera"] == "wrist"
    db = sqlite3.connect(first / "timeline.sqlite")
    n, executing = db.execute("SELECT count(*), sum(phase = 'executing') FROM timeline").fetchone()
    assert n > 30 and executing > 10
    path, = db.execute("SELECT wrist_path FROM timeline WHERE action_id = 1 LIMIT 1").fetchone()
    assert (first / path).exists()
    assert (ws / "episodes" / "current").resolve() == second.resolve()

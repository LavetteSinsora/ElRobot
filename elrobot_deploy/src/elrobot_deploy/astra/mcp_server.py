"""The MCP server Codex starts for the ElRobot: four tools, forwarded to the harness.

    python -m elrobot_deploy.astra.mcp_server [--socket PATH]

Registered by the workspace's `.codex/config.toml` (written at every launch; Codex reads it once
the project is trusted). Stateless: each tool call is one request on the harness's socket (always
`/tmp/elrobot-astra/harness.sock`, `astra.workspace.SOCKET_DIR`, so it needs no argument) and
one answer, whose images (files in the workspace) go back as MCP image content; their paths are
in the text too. MCP over stdio is
newline-delimited JSON-RPC 2.0; nothing but protocol messages is ever written to stdout (logs
go to stderr). Stdlib only, so it starts in well under a second.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
from pathlib import Path

PROTOCOL = "2025-06-18"
SERVER = {"name": "elrobot-astra", "version": "1.0.0"}

_WAYPOINT = {
    "type": "object",
    "properties": {
        "xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                "description": "metres along the reference pose's TCP x (out of the jaws), "
                               "y (the opening direction), z"},
        "rpy": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                "description": "roll, pitch, yaw in degrees, in the reference TCP frame "
                               "(R = Rz(yaw) Ry(pitch) Rx(roll))"},
        "gripper_mm": {"type": "number",
                       "description": "the gripper opening at this waypoint, mm"},
    },
    "required": ["xyz", "rpy", "gripper_mm"],
    "additionalProperties": False,
}

#: `astra.workspace.SOCKET_DIR` / harness.sock (not imported: this module stays stdlib only).
DEFAULT_SOCKET = "/tmp/elrobot-astra/harness.sock"

TOOLS = [
    {
        "name": "start",
        "description": (
            "Begin an episode with this task (the person's words). The robot starts from its "
            "home pose; returns the episode's folder and the current observation. Refused "
            "while an episode is running (finish it first)."),
        "inputSchema": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_action",
        "description": (
            "Move the arm through waypoints. Every waypoint is an offset from the arm's last "
            "commanded pose, in that pose's TCP frame (all from the same reference, not "
            "chained), with the gripper opening it should have there. The timing is chosen "
            "for you; the arm does not stop at waypoints. Returns when the motion is over: the "
            "current observation, the motion's history, and commanded vs measured per "
            "waypoint. README.md in the session folder explains frames and units."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "waypoints": {"type": "array", "items": _WAYPOINT, "minItems": 1},
                "speed": {"type": "string", "enum": ["normal", "slow"],
                          "description": "slow takes twice as long (default normal)"},
            },
            "required": ["waypoints"],
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_observation",
        "description": (
            "Labelled camera images of past ticks: 1 tick gives every camera at that tick, 2 "
            "to 4 ticks one column per tick, more gives several images. `cams` picks a "
            "subset. (A single raw frame can also be opened by its path under frames/.)"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticks": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                "cams": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ticks"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": (
            "End the episode with your verdict: success (is the task done), a summary of what "
            "you did, and optional lessons for next time. The robot then goes home; no robot "
            "tool works afterwards."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
                "lessons": {"type": "string"},
            },
            "required": ["success", "summary"],
            "additionalProperties": False,
        },
    },
]


def log(text: str) -> None:
    print(f"[elrobot-astra mcp] {text}", file=sys.stderr, flush=True)


def call_harness(sock_path: str, tool: str, args: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
        s.sendall(json.dumps({"tool": tool, "args": args}).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n", 1)[0])
    finally:
        s.close()


def tool_result(out: dict) -> dict:
    content = [{"type": "text", "text": out.get("text", "")}]
    for path in out.get("images", []):
        try:
            data = base64.b64encode(Path(path).read_bytes()).decode()
        except OSError as exc:
            content.append({"type": "text", "text": f"(image {path} unreadable: {exc})"})
            continue
        content.append({"type": "image", "data": data, "mimeType": "image/jpeg"})
    return {"content": content, "isError": bool(out.get("is_error"))}


class Server:
    def __init__(self, sock_path: str):
        self.sock_path = sock_path

    def handle(self, msg: dict) -> dict | None:
        """One JSON-RPC message -> its response (None for a notification)."""
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            return None                              # notifications/initialized and the like
        if method == "initialize":
            version = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL
            return _ok(mid, {"protocolVersion": version, "capabilities": {"tools": {}},
                             "serverInfo": SERVER,
                             "instructions": "AGENTS.md in the workspace explains the robot and these tools."})
        if method == "ping":
            return _ok(mid, {})
        if method == "tools/list":
            return _ok(mid, {"tools": TOOLS})
        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            if name not in {t["name"] for t in TOOLS}:
                return _err(mid, -32602, f"unknown tool {name!r}")
            try:
                out = call_harness(self.sock_path, name, params.get("arguments") or {})
            except (OSError, ValueError) as exc:
                out = {"text": (f"the robot is not reachable ({exc}): ask the person to start "
                                f"`elrobot_deploy astra` (in ElRobot/elrobot_deploy/) and press "
                                f"Home"), "is_error": True}
            return _ok(mid, tool_result(out))
        return _err(mid, -32601, f"method {method!r} not supported")


def _ok(mid, result) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The ElRobot Astra MCP server (stdio).")
    ap.add_argument("--socket", default=DEFAULT_SOCKET,
                    help=f"the harness's socket (default {DEFAULT_SOCKET})")
    args = ap.parse_args(argv)
    server = Server(args.socket)
    log(f"forwarding to {args.socket}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError as exc:
            out = _err(None, -32700, f"parse error: {exc}")
        else:
            out = server.handle(msg) if isinstance(msg, dict) else \
                _err(None, -32600, "batches are not supported")
        if out is not None:
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["TOOLS", "Server", "main", "tool_result"]

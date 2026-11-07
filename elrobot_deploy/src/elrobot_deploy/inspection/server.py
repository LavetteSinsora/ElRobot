"""Loopback HTTP server for files-only rollout inspection. There are no control endpoints."""

from __future__ import annotations

import json
import math
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from elrobot_deploy.dashboard.server import STATIC as LIVE_STATIC
from elrobot_deploy.inspection.loader import Catalog

STATIC = Path(__file__).parent / "static"


def handler(catalog):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, content_type="application/json", status=200):
            if content_type == "application/json":
                data = json.dumps(data, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                self.get()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (KeyError, IndexError, FileNotFoundError):
                self.send({"error": "Not found"}, status=404)
            except (ValueError, TypeError, OSError) as exc:
                self.send({"error": str(exc)}, status=400)

        def get(self):
            url = urlsplit(self.path)
            path = unquote(url.path)
            query = parse_qs(url.query)
            key = query.get("id", [""])[0]
            if path in ("/", "/index.html"):
                return self.file(STATIC / "index.html")
            if path.startswith("/static/"):
                root = STATIC.resolve()
                f = (root / path[len("/static/"):]).resolve()
                if not f.is_relative_to(root):
                    raise KeyError(path)
                return self.file(f)
            if path.startswith("/vendor/"):
                root = (LIVE_STATIC / "vendor").resolve()
                f = (root / path[len("/vendor/"):]).resolve()
                if not f.is_relative_to(root):
                    raise KeyError(path)
                return self.file(f)
            if path == "/api/records":
                return self.send({"records": catalog.records()})
            if path not in ("/api/record", "/api/robot", "/api/sample", "/api/frame") and not path.startswith("/mesh/"):
                raise KeyError(path)
            r = catalog.get(key, reload=path == "/api/record")
            if path == "/api/record":
                return self.send(r.data())
            if path == "/api/robot":
                return self.send(r.robot)
            if path == "/api/sample":
                t = float(query.get("t", [0])[0])
                if not math.isfinite(t):
                    raise ValueError("Time must be finite")
                return self.send(r.sample(t))
            if path == "/api/frame":
                name = query.get("camera", [""])[0]
                i = int(query.get("index", ["-1"])[0])
                cam = r.record.cameras[name]
                if not 0 <= i < len(cam.index):
                    raise IndexError(i)
                return self.send(cam.jpeg(i), "image/jpeg")
            if path.startswith("/mesh/"):
                i = int(path[len("/mesh/"):])
                if not 0 <= i < len(r.meshes):
                    raise IndexError(i)
                return self.send(r.meshes[i].read_bytes(), "model/stl")

        def file(self, path):
            kind = {".js": "text/javascript", ".html": "text/html; charset=utf-8",
                    ".css": "text/css"}.get(path.suffix, mimetypes.guess_type(path)[0])
            self.send(path.read_bytes(), kind or "application/octet-stream")

    return Handler


def serve(root, *, episodes=None, port=8770):
    catalog = Catalog(root, episodes)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler(catalog))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd

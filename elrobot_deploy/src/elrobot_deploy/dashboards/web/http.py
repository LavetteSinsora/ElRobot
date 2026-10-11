"""The HTTP base the inspection dashboard runs on: stdlib, no build step, no CDN.

It runs on the deploy box beside the robot, so it is `http.server` from the
standard library. What it adds over a bare `BaseHTTPRequestHandler`:

  * an `App` with a route table (`get` / `post` decorators, exact paths or
    prefixes) and a list of static directories searched in order, so a
    dashboard's own `static/` shadows `web/static/` and a mode can add one;
  * a `Request` handed to each route, with the query, the JSON body, and the
    three replies a route ever sends (`json`, `bytes`, `error`);
  * the error policy: a route that raises `KeyError` / `ValueError` /
    `FileNotFoundError` is a 400 with the message, anything else a 500 with the
    traceback on the console, and an unknown path a 404.

`serve(app, host, port)` runs it and calls `app.close()` on Ctrl-C.

Used by the inspection server (`dashboards/inspect/server.py`) and its modes
(`docs/deploying_a_policy/rollout_inspection.md`, "Writing a mode").
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json",
}

#: Where the shared client files are: `gl.js` (the robot renderer),
#: `cameras.js`, `marks.js`, `style.css`. Served under `/static/` by every app
#: after its own directory, so the two views cannot disagree about what the
#: robot looks like.
COMMON_STATIC = Path(__file__).resolve().parent / "static"


class Request:
    """One request, as a route sees it."""

    def __init__(self, handler: BaseHTTPRequestHandler, path: str, query: dict[str, list[str]]):
        self._h = handler
        self.path = path
        self.query = query
        self.replied = False

    def arg(self, name: str, default: str | None = None) -> str | None:
        vals = self.query.get(name)
        return vals[0] if vals else default

    def body(self) -> dict:
        n = int(self._h.headers.get("Content-Length", 0))
        raw = self._h.rfile.read(n) if n else b""
        return json.loads(raw or b"{}")

    # --- replies -------------------------------------------------------------

    def bytes(self, payload: bytes, ctype: str, code: int = 200,
              extra: dict[str, str] | None = None) -> None:
        h = self._h
        h.send_response(code)
        h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(payload)))
        for k, v in (extra or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(payload)
        self.replied = True

    def json(self, obj, code: int = 200) -> None:
        self.bytes(json.dumps(obj).encode(), "application/json", code)

    def error(self, exc: Exception | str, code: int = 400) -> None:
        msg = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        self.json({"error": msg}, code)


Route = Callable[[Request], None]


class App:
    """A dashboard: its routes, its static directories, and what to close."""

    def __init__(self, name: str, static: list[Path] | None = None):
        self.name = name
        #: Searched in order; the first hit wins. A dashboard lists its own
        #: directory first and `COMMON_STATIC` last.
        self.static_dirs: list[Path] = [Path(p).resolve() for p in (static or [])]
        self._exact: dict[tuple[str, str], Route] = {}
        self._prefix: list[tuple[str, str, Route]] = []
        self._closers: list[Callable[[], None]] = []

    # --- registration -------------------------------------------------------

    def get(self, path: str):
        return self._register("GET", path)

    def post(self, path: str):
        return self._register("POST", path)

    def _register(self, method: str, path: str):
        def deco(fn: Route) -> Route:
            if path.endswith("/"):
                self._prefix.append((method, path, fn))
            else:
                self._exact[(method, path)] = fn
            return fn
        return deco

    def on_close(self, fn: Callable[[], None]) -> None:
        self._closers.append(fn)

    def close(self) -> None:
        for fn in reversed(self._closers):
            fn()

    # --- dispatch -----------------------------------------------------------

    def dispatch(self, method: str, req: Request) -> None:
        path = req.path
        if method == "GET" and path == "/":
            return self.static(req, "index.html")
        if method == "GET" and path.startswith("/static/"):
            return self.static(req, path[len("/static/"):])
        fn = self._exact.get((method, path))
        if fn is None:
            for m, prefix, cand in self._prefix:
                if m == method and path.startswith(prefix):
                    fn = cand
                    break
        if fn is None:
            return req.json({"error": "not found"}, 404)
        try:
            fn(req)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            if not req.replied:
                req.error(exc, 400)
        except Exception as exc:
            traceback.print_exc()
            if not req.replied:
                req.error(exc, 500)

    def static(self, req: Request, name: str) -> None:
        for root in self.static_dirs:
            target = (root / name).resolve()
            if target.is_file() and root in target.parents:
                ctype = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
                # No-store: the app is edited while it runs, and a cached app.js
                # that disagrees with the server is a confusing ten minutes.
                return req.bytes(target.read_bytes(), ctype,
                                 extra={"Cache-Control": "no-store"})
        req.json({"error": "not found"}, 404)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = app.name
        protocol_version = "HTTP/1.1"          # keep-alive: frames come in bursts
        #: TCP_NODELAY. A reply is two writes (headers, then body) on a keep-alive
        #: connection; with Nagle on, a body whose last segment is short waits for
        #: the browser's delayed ACK — a steady ~40 ms per head-camera frame here,
        #: against ~1 ms for the wrist frames, whose sizes happened to dodge it.
        disable_nagle_algorithm = True

        def log_message(self, fmt, *args):
            pass

        def _serve(self, method: str) -> None:
            url = urlparse(self.path)
            req = Request(self, unquote(url.path), parse_qs(url.query))
            try:
                app.dispatch(method, req)
            except Exception as exc:
                traceback.print_exc()
                if not req.replied:
                    req.error(exc, 500)

        def do_GET(self) -> None:
            self._serve("GET")

        def do_POST(self) -> None:
            self._serve("POST")

    return Handler


def serve(app: App, host: str = "127.0.0.1", port: int = 8770) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"{app.name} -> http://{host}:{port}")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        app.close()
        httpd.server_close()


__all__ = ["COMMON_STATIC", "App", "Request", "Route", "make_handler", "serve"]

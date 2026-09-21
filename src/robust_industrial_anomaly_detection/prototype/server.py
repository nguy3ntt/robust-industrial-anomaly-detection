"""Loopback-only HTTP application for the M6 local inspector."""

from __future__ import annotations

import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, cast
from urllib.parse import parse_qs, urlparse

from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    PrototypeError,
)
from robust_industrial_anomaly_detection.prototype.reporting import (
    ReportError,
    build_inspection_report,
)


class PrototypeServer(ThreadingHTTPServer):
    """Threading server carrying one shared, internally locked model engine."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], engine: InspectionEngine) -> None:
        super().__init__(address, PrototypeHandler)
        self.engine = engine


class PrototypeHandler(BaseHTTPRequestHandler):
    server: PrototypeServer
    protocol_version = "HTTP/1.1"
    static_root = Path(__file__).with_name("static")
    static_files: ClassVar[dict[str, tuple[str, str]]] = {
        "/": ("workstation.html", "text/html; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/styles.css": ("workstation.css", "text/css; charset=utf-8"),
    }

    def log_message(self, format: str, *args: object) -> None:
        message = format % args
        print(f"[prototype] {self.client_address[0]} {message}", flush=True)

    def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()

    def _send_bytes(self, payload: bytes, *, status: HTTPStatus, content_type: str) -> None:
        self._headers(status, content_type, len(payload))
        self.wfile.write(payload)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._send_bytes(encoded, status=status, content_type="application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def _content_length(self, limit: int) -> int:
        raw = self.headers.get("Content-Length")
        try:
            length = int(raw or "0")
        except ValueError as exc:
            raise PrototypeError("invalid Content-Length header") from exc
        if length < 1 or length > limit:
            raise PrototypeError(f"request body must contain 1 to {limit:,} bytes")
        return length

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/health":
            self._send_json({"status": "ready", **self.server.engine.runtime_info()})
            return
        static = self.static_files.get(route)
        if static is None:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        path = self.static_root / static[0]
        try:
            content = path.read_bytes()
        except OSError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "static asset unavailable")
            return
        self._send_bytes(content, status=HTTPStatus.OK, content_type=static[1])

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            if route == "/api/inspect":
                self._inspect()
            elif route == "/api/report":
                self._report()
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except (PrototypeError, ReportError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # defensive HTTP boundary
            print(f"[prototype] internal error: {type(exc).__name__}: {exc}", flush=True)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "inspection failed")

    def _inspect(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        category = query.get("category", [""])[0]
        filename = self.headers.get("X-Filename", "inspection-image")
        length = self._content_length(self.server.engine.settings.max_upload_bytes)
        payload = self.rfile.read(length)
        result = self.server.engine.inspect_bytes(
            payload,
            filename=filename,
            category=category,
        )
        self._send_json(result.public_dict())

    def _report(self) -> None:
        limit = min(self.server.engine.settings.max_upload_bytes, 2_000_000)
        length = self._content_length(limit)
        raw = json.loads(self.rfile.read(length))
        if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
            raise ReportError("report body must contain a results array")
        results = cast(list[dict[str, object]], raw["results"])
        report = build_inspection_report(
            results,
            runtime_info=self.server.engine.runtime_info(),
            max_results=self.server.engine.settings.max_batch_files,
        )
        encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="inspection-report.json"')
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)


def validate_loopback_host(host: str) -> None:
    """Reject network-visible bindings before a socket is created."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise PrototypeError("M6 only permits loopback binding")


def serve(
    engine: InspectionEngine,
    *,
    host: str,
    port: int,
    open_browser: bool = False,
) -> None:
    """Serve until interrupted, defaulting to the configured loopback address."""

    validate_loopback_host(host)
    server = PrototypeServer((host, port), engine)
    print(f"Local inspector ready at http://{host}:{port}", flush=True)
    print("Uploads are processed in memory and are not written to disk.", flush=True)
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("Stopping local inspector.", flush=True)
    finally:
        server.server_close()

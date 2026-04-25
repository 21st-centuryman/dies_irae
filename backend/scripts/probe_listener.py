"""Tiny diagnostic listener: logs every incoming HTTP request.

Captures method, path, query, headers, and body. Tries to pretty-print
JSON; otherwise prints up to 256 bytes of body as text + a hex dump.
Each full request is also written to /tmp/probe_<n>.bin for later
inspection.

Usage: python scripts/probe_listener.py [port]

Defaults to port 3000, binds 0.0.0.0 so it's reachable from LAN.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qsl

DUMP_DIR = "/tmp"
counter = 0


def hex_preview(data: bytes, n: int = 64) -> str:
    head = data[:n]
    return " ".join(f"{b:02x}" for b in head) + ("..." if len(data) > n else "")


def text_preview(data: bytes, n: int = 256) -> str:
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"<{len(data)} bytes, not UTF-8>"
    if len(s) > n:
        return s[:n] + f"... [{len(s) - n} more chars]"
    return s


class Handler(BaseHTTPRequestHandler):
    def _handle(self) -> None:
        global counter
        counter += 1
        n = counter

        ts = datetime.now().isoformat(timespec="milliseconds")
        url = urlparse(self.path)
        query = dict(parse_qsl(url.query, keep_blank_values=True))

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""

        ctype = (self.headers.get("Content-Type") or "").lower()

        print(f"\n{'=' * 70}")
        print(f"[#{n}] {ts}  {self.command}  {url.path}")
        if query:
            print(f"  query: {query}")
        print(f"  from:  {self.client_address[0]}:{self.client_address[1]}")
        print(f"  headers ({len(self.headers)}):")
        for k, v in self.headers.items():
            print(f"    {k}: {v}")
        print(f"  body:  {len(body)} bytes  content-type={ctype!r}")

        if body:
            dump_path = os.path.join(DUMP_DIR, f"probe_{n:03d}.bin")
            try:
                with open(dump_path, "wb") as f:
                    f.write(body)
                print(f"  saved: {dump_path}")
            except OSError as e:
                print(f"  (could not save: {e})")

            if "application/json" in ctype or body.lstrip().startswith((b"{", b"[")):
                try:
                    parsed = json.loads(body)
                    print("  JSON parsed:")
                    print(json.dumps(parsed, indent=2, ensure_ascii=False)[:4000])
                except json.JSONDecodeError as e:
                    print(f"  JSON parse failed: {e}")
                    print(f"  text: {text_preview(body)}")
            elif "text/" in ctype or "x-www-form-urlencoded" in ctype:
                print(f"  text:  {text_preview(body)}")
            else:
                print(f"  text:  {text_preview(body)}")
                print(f"  hex:   {hex_preview(body)}")

        print("=" * 70, flush=True)

        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        )
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "received": len(body), "n": n}).encode())

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle

    def do_OPTIONS(self) -> None:
        # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        )
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # Suppress the default per-line access log; we print our own.
        return


def main(port: int) -> int:
    addr = ("0.0.0.0", port)
    print(f"probe listener on http://{addr[0]}:{addr[1]} — waiting for requests…", flush=True)
    print(f"dumps go to {DUMP_DIR}/probe_NNN.bin", flush=True)
    httpd = ThreadingHTTPServer(addr, Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    sys.exit(main(port))

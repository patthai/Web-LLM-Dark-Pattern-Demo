#!/usr/bin/env python3
"""
Proxy HTTP server for WebLLM model downloads.

- Serves static files from the current directory normally.
- For /model/<model_id>/... requests:
    1. Serves the file from ./model/... if it exists locally.
    2. Otherwise proxies from HuggingFace CDN, streams the response,
       saves the file to disk, and updates an in-memory progress dict.
- GET /model-progress  → JSON snapshot of all active downloads.

Usage:
    python3 server.py [port]    # default port 8000
"""

import json
import os
import ssl
import sys
import threading
import urllib.request
import urllib.error
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer
from pathlib import Path

# ── Threading HTTP server ─────────────────────────────────────
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

HF_BASE   = "https://huggingface.co/mlc-ai"
MODEL_DIR = Path("model")
CHUNK     = 65536   # 64 KB read chunks

# ── Shared progress state (thread-safe) ──────────────────────
_prog_lock = threading.Lock()
_progress  = {}   # model_id → {file, pct, downloaded, total, done}

def _set_prog(model_id, file, pct, downloaded, total, done=False):
    with _prog_lock:
        _progress[model_id] = {
            "file": file, "pct": pct,
            "downloaded": downloaded, "total": total, "done": done
        }

def _clear_prog(model_id):
    with _prog_lock:
        _progress.pop(model_id, None)

def _get_prog():
    with _prog_lock:
        return dict(_progress)


class ProxyHandler(SimpleHTTPRequestHandler):

    # ── CORS for every response ──────────────────────────────
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.end_headers()

    # ── Route requests ───────────────────────────────────────
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/model-progress":
            self._handle_progress()
        elif path.startswith("/model/"):
            self._handle_model(head_only=False)
        else:
            super().do_GET()

    def do_HEAD(self):
        path = self.path.split("?")[0]
        if path.startswith("/model/"):
            self._handle_model(head_only=True)
        else:
            super().do_HEAD()

    # ── /model-progress endpoint ─────────────────────────────
    def _handle_progress(self):
        data = json.dumps(_get_prog()).encode()
        self.send_response(200)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ── Main model handler ───────────────────────────────────
    def _handle_model(self, head_only=False):
        rel = self.path.split("?")[0][7:]          # strip /model/
        parts = rel.split("/", 1)
        if len(parts) < 2 or not parts[1]:
            self.send_error(400, "Missing file path")
            return

        model_id, file_path = parts[0], parts[1]
        local_path = MODEL_DIR / model_id / file_path

        if local_path.exists():
            self._serve_local(local_path, head_only)
        else:
            # file_path already contains "resolve/main/..." as appended by WebLLM
            hf_url = f"{HF_BASE}/{model_id}/{file_path}"
            self._proxy_and_cache(hf_url, local_path, model_id, head_only)

    # ── Serve from local disk ────────────────────────────────
    def _serve_local(self, path, head_only=False):
        try:
            size = path.stat().st_size
            ct   = self.guess_type(str(path))
            self.send_response(200)
            self.send_header("Content-Type",   ct)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            if not head_only:
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(CHUNK)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        except Exception:
            pass

    # ── Proxy from HuggingFace and cache to disk ─────────────
    def _proxy_and_cache(self, url, local_path, model_id, head_only=False):
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 WebLLM-Proxy/1.0"},
            )
            ctx = ssl.create_default_context()
            ctx.options |= ssl.OP_IGNORE_UNEXPECTED_EOF
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                ct = resp.headers.get("Content-Type", "application/octet-stream")
                cl = resp.headers.get("Content-Length")

                self.send_response(200)
                self.send_header("Content-Type", ct)
                if cl:
                    self.send_header("Content-Length", cl)
                self.end_headers()

                if head_only:
                    return

                local_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded = 0
                total      = int(cl) if cl else 0
                name       = local_path.name
                last_pct   = -1

                _set_prog(model_id, name, 0, 0, total)

                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        self.wfile.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100 // total
                            if pct != last_pct:
                                last_pct = pct
                                _set_prog(model_id, name, pct, downloaded, total)
                                mb  = downloaded / 1_048_576
                                tot = total / 1_048_576
                                print(
                                    f"\r  {name}: {pct:3d}%  {mb:.1f}/{tot:.1f} MB",
                                    end="", file=sys.stderr, flush=True,
                                )

                # Atomic rename: only keep if fully downloaded
                tmp_path.rename(local_path)
                _set_prog(model_id, name, 100, total, total, done=True)
                if total > 0:
                    print(f"\r  {name}: 100%  {total/1_048_576:.1f} MB  ✓",
                          file=sys.stderr)
                else:
                    print(f"  {name}  ✓", file=sys.stderr)

        except urllib.error.HTTPError as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            _clear_prog(model_id)
            print(f"\n  HF {e.code}: {url}", file=sys.stderr)
            try:
                self.send_error(e.code, str(e.reason))
            except Exception:
                pass

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            _clear_prog(model_id)
            print(f"\n  Error proxying {url}: {e}", file=sys.stderr)
            try:
                self.send_error(502, str(e))
            except Exception:
                pass

    # ── Quiet access log ─────────────────────────────────────
    def log_message(self, fmt, *args):
        try:
            path = str(args[0]) if args else ""
            if "/model/" in path and ".progress" not in path:
                status = args[1] if len(args) > 1 else "?"
                if status not in ("200",):
                    print(f"  {status}  {path}", file=sys.stderr)
        except Exception:
            pass


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8000))
    MODEL_DIR.mkdir(exist_ok=True)
    httpd = ThreadingHTTPServer(("", port), ProxyHandler)
    print(f"Serving  http://localhost:{port}/", file=sys.stderr)
    print(f"Cache    ./{MODEL_DIR}/", file=sys.stderr)
    print("Press Ctrl-C to stop.\n", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.", file=sys.stderr)

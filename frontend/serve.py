"""
serve.py
Serves the GeoSynth frontend from the repo root so that
frontend files can reference outputs/predictions/* correctly.

Usage (from repo root or frontend/):
    python frontend/serve.py

The dashboard will be available at:
    http://127.0.0.1:8000/frontend/
"""

import os
import sys
import subprocess
import http.server
import socketserver

PORT      = 8000
HOST      = "127.0.0.1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rebuild_report():
    script = os.path.join(REPO_ROOT, "frontend", "build_report.py")
    if not os.path.isfile(script):
        print("[WARN] build_report.py not found, skipping rebuild.")
        return
    print("[INFO] Rebuilding report-data.js ...")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("[WARN] build_report.py returned errors:")
        print(result.stderr.strip() or result.stdout.strip())


class SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        if args and str(args[1]) not in ("200", "304"):
            print("[HTTP] %s %s" % (self.path, args[1]))

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    rebuild_report()

    os.chdir(REPO_ROOT)

    with ReusableTCPServer((HOST, PORT), SilentHandler) as httpd:
        url = "http://{}:{}/frontend/".format(HOST, PORT)
        print("")
        print("  GeoSynth is running at: {}".format(url))
        print("  Serving from: {}".format(REPO_ROOT))
        print("  Press Ctrl+C to stop.")
        print("")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[INFO] Server stopped.")


if __name__ == "__main__":
    main()

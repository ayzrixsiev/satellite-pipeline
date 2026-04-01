"""Serve the report page with Python's built-in HTTP server.

This is just a convenience wrapper so you can launch the report with one command
instead of remembering the exact `http.server` invocation.
"""

from __future__ import annotations

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORT = 8000


def main() -> None:
    # Refresh the browser payload on every server start so the page reflects
    # the latest metrics and saved prediction images.
    runpy.run_path(str(PROJECT_ROOT / "frontend" / "build_report.py"), run_name="__main__")

    # We serve from the project root so the frontend can reach both:
    # - `frontend/index.html`
    # - `outputs/predictions/...`
    os.chdir(PROJECT_ROOT)

    server = ThreadingHTTPServer(("127.0.0.1", PORT), SimpleHTTPRequestHandler)
    print(f"Serving GeoSynth report at http://127.0.0.1:{PORT}/frontend/")
    server.serve_forever()


if __name__ == "__main__":
    main()

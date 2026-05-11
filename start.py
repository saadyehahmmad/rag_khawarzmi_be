"""
Development server entry point.

Starts uvicorn with parameters tuned for Windows to avoid WinError 10055
(WSAENOBUFS — socket buffer / port exhaustion):

  - timeout_keep_alive=5    close idle HTTP keep-alive sockets faster
  - limit_concurrency=40    reject new requests beyond limit (503) before
                            the Windows socket table fills up
  - backlog=64              smaller listen queue, reduces buffered connections
  - loop="asyncio"          explicit selector — avoids uvloop issues on Windows
  - workers=1               single worker; avoids spawning extra sub-processes
                            that each occupy a socket handle

Run with:
    python start.py
or for prod-like (no reload):
    python start.py --no-reload
"""

from __future__ import annotations

import os
import sys

import uvicorn

# Windows multiprocessing requires the entry-point to be guarded so the
# spawned worker process does not re-execute the top-level uvicorn.run() call.
if __name__ == "__main__":
    _reload = "--no-reload" not in sys.argv
    port = int(os.getenv("PORT", "8000"))
    # Default to 1 worker (safe for memory-constrained hosts like Render free tier).
    # Set UVICORN_WORKERS=4 on hosts with ≥4 GB RAM.
    workers = int(os.getenv("UVICORN_WORKERS", "1"))

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=_reload,
        # ── Windows socket-buffer mitigations (WinError 10055) ──────────
        timeout_keep_alive=5,   # close idle keep-alive connections after 5 s
        limit_concurrency=40,   # cap concurrent requests; returns 503 beyond this
        backlog=64,             # keep the accept queue small
        loop="asyncio",         # explicit loop, prevents sub-process socket leaks
        # ── Logging ─────────────────────────────────────────────────────
        # Uvicorn's own access log is redundant; middleware emits JSON logs.
        access_log=False,
    )

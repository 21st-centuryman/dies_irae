"""FastAPI entrypoint for the drone swarm simulator backend.

Run with:
    python -m uvicorn app.main:app --reload --port 8000

On startup, fetches the heightmap for `DEFAULT_TARGET_LAT/LON` and stashes
a `Heightmap` on `app.state` so route handlers can build sims against it.
Falls back to synthetic terrain if the map server is unreachable.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.terrain import Heightmap, load_heightmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

# Phase-2 hardcoded target; Phase-5 will accept this via POST /simulation.
DEFAULT_TARGET_LAT = 56.579
DEFAULT_TARGET_LON = 14.186


@asynccontextmanager
async def lifespan(app: FastAPI):
    raw = load_heightmap(DEFAULT_TARGET_LAT, DEFAULT_TARGET_LON)
    app.state.heightmap = Heightmap(raw, DEFAULT_TARGET_LAT, DEFAULT_TARGET_LON)
    log.info(
        "heightmap ready: %dx%d, %.1fm/px x %.1fm/px",
        app.state.heightmap.width_px,
        app.state.heightmap.height_px,
        app.state.heightmap.m_per_px_x,
        app.state.heightmap.m_per_px_y,
    )
    yield


app = FastAPI(title="Drone Swarm Simulator", version="0.2.0", lifespan=lifespan)

# Frontends run from arbitrary origins during development (127.0.0.1:8080,
# file://, LAN IPs, etc). Wide-open CORS for now; tighten in Phase 7.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def announce_writes(request: Request, call_next):
    """Print a one-liner for every PUT/POST so incoming client writes are visible.

    Keeps OPTIONS preflights and GETs quiet to avoid drowning out the signal.
    The /scenario/start handler additionally prints the parsed config block.
    """
    if request.method in ("PUT", "POST"):
        client = f"{request.client.host}:{request.client.port}" if request.client else "?"
        print(f">>> {request.method} {request.url.path}  from {client}", flush=True)
    return await call_next(request)


app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "drone-swarm-simulator", "status": "ok"}

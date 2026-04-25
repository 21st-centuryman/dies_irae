"""FastAPI entrypoint — routes, WebSocket stream, and lifespan.

Run with:
    python -m uvicorn app.main:app --reload --port 3000
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.sim import (
    SIM_DT, STREAM_HZ,
    ScenarioConfig, ScenarioSession, SimEngine,
    pack_frame,
)
from app.terrain import Heightmap, load_heightmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_TARGET_LAT = 56.579
DEFAULT_TARGET_LON = 14.186

# ---------------------------------------------------------------------------
# Scenario state
# ---------------------------------------------------------------------------

_active_scenario: ScenarioSession | None = None
_scenario_event  = asyncio.Event()
_scenario_lock   = asyncio.Lock()
_speed: int      = 1

WAIT_FOR_SCENARIO_TIMEOUT_S = 600
_MAP_HALF_EXTENT_M          = 4_500.0   # just inside the 5 km radius (10 km diameter) map

# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def announce_writes(request: Request, call_next):
    if request.method in ("PUT", "POST"):
        client = f"{request.client.host}:{request.client.port}" if request.client else "?"
        print(f">>> {request.method} {request.url.path}  from {client}", flush=True)
    return await call_next(request)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_target_xy(targets: list) -> tuple[float, float]:
    if not targets:
        return (0.0, 0.0)
    first = targets[0]
    if isinstance(first, (list, tuple)):
        return (float(first[0]), float(first[1]))
    return (float(targets[0]), float(targets[1]))


def _bearing_opposite_target(tx: float, ty: float) -> float:
    if tx == 0.0 and ty == 0.0:
        return 0.0
    math_angle_deg = math.degrees(math.atan2(ty, tx))
    return (90.0 - math_angle_deg + 180.0) % 360.0


def _build_engine(heightmap: Heightmap, config: ScenarioConfig) -> SimEngine:
    target_xy = _first_target_xy(config.targets)
    tx, ty    = target_xy
    return SimEngine(
        heightmap=heightmap,
        target_xy=target_xy,
        ring_radius_m=_MAP_HALF_EXTENT_M,
        spawn_bearing_deg=_bearing_opposite_target(tx, ty),
    )


async def _engine_loop(engine: SimEngine) -> None:
    next_tick = time.monotonic()
    try:
        while not engine.finished:
            for _ in range(_speed):
                engine.step()
                if engine.finished:
                    break
            next_tick += SIM_DT
            sleep_for  = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                await asyncio.sleep(0)
                next_tick = time.monotonic()
        log.info("engine loop ended: reason=%s sim_t=%.2fs", engine.finish_reason, engine.t)
    except asyncio.CancelledError:
        log.info("engine loop cancelled at sim_t=%.2fs", engine.t)
        raise


def _clear_if_owns(session: ScenarioSession) -> None:
    global _active_scenario
    if _active_scenario is session:
        _active_scenario = None
        _scenario_event.clear()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "drone-swarm-simulator", "status": "ok"}


async def _start_scenario(config: ScenarioConfig, request: Request) -> dict[str, Any]:
    global _active_scenario

    client = f"{request.client.host}:{request.client.port}" if request.client else "?"
    print("\n" + "=" * 60)
    print(f"  SCENARIO REQUEST  ({request.method} {request.url.path})")
    print(f"  from {client}")
    print("-" * 60)
    for k, v in config.model_dump().items():
        print(f"  {k:>15} : {v}")
    print("=" * 60, flush=True)

    async with _scenario_lock:
        if _active_scenario is not None and _active_scenario.engine_task is not None:
            _active_scenario.engine_task.cancel()

        engine  = _build_engine(request.app.state.heightmap, config)
        session = ScenarioSession(config=config, engine=engine)
        session.engine_task = asyncio.create_task(_engine_loop(engine))
        _active_scenario = session
        _scenario_event.set()

    print(
        f"  -> armed: spawn=({engine.spawn_xyz[0]:.0f}, {engine.spawn_xyz[1]:.0f}, "
        f"{engine.spawn_xyz[2]:.0f})  "
        f"target=({engine.target_xy[0]:.0f}, {engine.target_xy[1]:.0f}, "
        f"{engine.target_z:.1f})  "
        f"ring={engine.ring_radius:.0f}m  type={engine.drone_type}\n",
        flush=True,
    )
    return {
        "status":       "started",
        "started_at":   session.started_at,
        "config":       config.model_dump(),
        "spawn_xyz":    [float(v) for v in engine.spawn_xyz],
        "target_xy":    list(engine.target_xy),
        "target_z":     engine.target_z,
        "max_speed_mps":  engine.max_speed,
        "max_accel_mps2": engine.max_accel,
    }


app.add_api_route("/scenario/start", _start_scenario, methods=["PUT"])


@app.get("/simulation/status")
async def simulation_status() -> dict[str, Any]:
    if _active_scenario is None:
        return {"active": False}
    engine = _active_scenario.engine
    return {
        "active":        True,
        "started_at":    _active_scenario.started_at,
        "elapsed_s":     time.monotonic() - _active_scenario.started_at,
        "sim_t":         engine.t,
        "finished":      engine.finished,
        "finish_reason": engine.finish_reason,
        "drone_pos":     [float(v) for v in engine.state.positions[0]],
        "drone_vel":     [float(v) for v in engine.state.velocities[0]],
        "config":        _active_scenario.config.model_dump(),
    }


@app.put("/simulation/speed")
async def set_speed(body: dict) -> dict[str, Any]:
    global _speed
    _speed = max(1, int(body.get("speed", 1)))
    return {"speed": _speed}


@app.websocket("/simulation/{sim_id}/stream")
async def simulation_stream(websocket: WebSocket, sim_id: str) -> None:
    await websocket.accept()
    log.info("ws open sim=%s", sim_id)

    if _active_scenario is None:
        log.info("ws sim=%s waiting for scenario start", sim_id)
        try:
            await asyncio.wait_for(
                _scenario_event.wait(), timeout=WAIT_FOR_SCENARIO_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            await websocket.close(code=1011, reason="no scenario started")
            return

    session = _active_scenario
    assert session is not None
    engine = session.engine

    dt = 1.0 / STREAM_HZ
    frame_number = 0
    try:
        while True:
            payload = pack_frame(frame_number, engine.t, engine.state.to_records())
            await websocket.send_bytes(payload)
            frame_number += 1

            if engine.finished:
                event = {"type": "event", "events": [{"kind": "scenario_ended",
                    "reason": engine.finish_reason or "unknown",
                    "time": engine.t, "frames": frame_number}]}
                try:
                    await websocket.send_text(json.dumps(event))
                except Exception:
                    pass
                log.info("ws sim=%s scenario ended (%s) at t=%.2fs frames=%d",
                         sim_id, engine.finish_reason, engine.t, frame_number)
                _clear_if_owns(session)
                await asyncio.sleep(0.2)
                await websocket.close(code=1000, reason="scenario_ended")
                return

            await asyncio.sleep(dt)
    except WebSocketDisconnect:
        log.info("ws closed sim=%s (frames=%d)", sim_id, frame_number)
    except Exception:
        log.exception("ws stream error sim=%s", sim_id)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass

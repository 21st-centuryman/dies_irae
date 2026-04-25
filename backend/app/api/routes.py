"""Control API + streaming WebSocket.

Flow:
  1. Frontend PUTs (or POSTs) /scenario/start with the scenario config.
     This builds a SimEngine, spawns one drone at the ring perimeter
     (compass bearing 0° = North) at SPAWN_ALTITUDE_AGL_M, and kicks off
     a 60 Hz background tick task.
  2. Frontend opens /simulation/{id}/stream. The WS handler waits for the
     scenario event if needed, then streams binary position frames at
     STREAM_HZ by reading the engine's current state — it does NOT tick
     the engine itself.
  3. When the engine reports `finished` (drone reached target, or safety
     timeout), the WS sends a final binary frame plus a JSON
     `scenario_ended` event, closes, and clears the active scenario so the
     next PUT starts fresh.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from app.protocol import pack_frame
from app.sim.engine import SIM_DT, STREAM_HZ, SimEngine
from app.sim.session import ScenarioConfig, ScenarioSession

log = logging.getLogger(__name__)

router = APIRouter()

WAIT_FOR_SCENARIO_TIMEOUT_S = 600

_active_scenario: ScenarioSession | None = None
_scenario_event = asyncio.Event()
_scenario_lock = asyncio.Lock()


def _first_target_xy(targets: list) -> tuple[float, float]:
    """Use the first point in `targets` as the drone's destination.

    `targets` is a list of [x, y] points; only `targets[0]` is consumed
    today. Phase-4+ may dispatch each drone to a different index.
    """
    if not targets:
        return (0.0, 0.0)
    first = targets[0]
    if isinstance(first, (list, tuple)):
        return (float(first[0]), float(first[1]))
    # Defensive: if someone sends a flat [x, y], treat it as a single target.
    return (float(targets[0]), float(targets[1]))


def _build_engine(heightmap, config: ScenarioConfig) -> SimEngine:
    target_xy = _first_target_xy(config.targets)
    return SimEngine(
        heightmap=heightmap,
        target_xy=target_xy,
        spawn_bearing_deg=0.0,
    )


async def _engine_loop(engine: SimEngine) -> None:
    """Tick the engine at SIM_HZ until it finishes or the task is cancelled."""
    next_tick = time.monotonic()
    try:
        while not engine.finished:
            engine.step()
            next_tick += SIM_DT
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                # Falling behind real time — yield and keep going. Sim time
                # will lag wall time, but we stay deterministic per-tick.
                await asyncio.sleep(0)
                next_tick = time.monotonic()
        log.info(
            "engine loop ended: reason=%s sim_t=%.2fs", engine.finish_reason, engine.t
        )
    except asyncio.CancelledError:
        log.info("engine loop cancelled at sim_t=%.2fs", engine.t)
        raise


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
        # Cancel any existing engine task — replacing the active scenario.
        if _active_scenario is not None and _active_scenario.engine_task is not None:
            _active_scenario.engine_task.cancel()

        heightmap = request.app.state.heightmap
        engine = _build_engine(heightmap, config)
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
        "status": "started",
        "started_at": session.started_at,
        "config": config.model_dump(),
        "spawn_xyz": [float(v) for v in engine.spawn_xyz],
        "target_xy": list(engine.target_xy),
        "target_z": engine.target_z,
        "max_speed_mps": engine.max_speed,
        "max_accel_mps2": engine.max_accel,
    }


router.add_api_route(
    "/scenario/start",
    _start_scenario,
    methods=["PUT"],
    name="start_scenario",
)


@router.get("/simulation/status")
async def simulation_status() -> dict[str, Any]:
    if _active_scenario is None:
        return {"active": False}
    engine = _active_scenario.engine
    return {
        "active": True,
        "started_at": _active_scenario.started_at,
        "elapsed_s": time.monotonic() - _active_scenario.started_at,
        "sim_t": engine.t,
        "finished": engine.finished,
        "finish_reason": engine.finish_reason,
        "drone_pos": [float(v) for v in engine.state.positions[0]],
        "drone_vel": [float(v) for v in engine.state.velocities[0]],
        "config": _active_scenario.config.model_dump(),
    }


def _clear_if_owns(session: ScenarioSession) -> None:
    global _active_scenario
    if _active_scenario is session:
        _active_scenario = None
        _scenario_event.clear()


@router.websocket("/simulation/{sim_id}/stream")
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
            records = engine.state.to_records()
            payload = pack_frame(frame_number, engine.t, records)
            await websocket.send_bytes(payload)
            frame_number += 1

            if engine.finished:
                event = {
                    "type": "event",
                    "events": [
                        {
                            "kind": "scenario_ended",
                            "reason": engine.finish_reason or "unknown",
                            "time": engine.t,
                            "frames": frame_number,
                        }
                    ],
                }
                try:
                    await websocket.send_text(json.dumps(event))
                except Exception:
                    pass
                log.info(
                    "ws sim=%s scenario ended (%s) at t=%.2fs frames=%d",
                    sim_id, engine.finish_reason, engine.t, frame_number,
                )
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

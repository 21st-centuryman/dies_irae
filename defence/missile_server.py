"""
Defence missile server.

Subscribes to the drone backend WebSocket, simulates SAM missiles intercepting
drones, and broadcasts missile positions on its own WebSocket stream using the
same binary wire format as the drone backend.

Run with:
    python -m uvicorn defence.missile_server:app --port 4000
Or from the repo root:
    uvicorn defence.missile_server:app --port 4000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Wire format  (identical to drone backend so missile.js reuses the decoder)
# ---------------------------------------------------------------------------

DRONE_RECORD_DTYPE = np.dtype(
    [
        ("id", "<u2"),
        ("type", "<u1"),
        ("state", "<u1"),
        ("intent", "<u1"),
        ("_pad", "<u1"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("vx", "<f4"),
        ("vy", "<f4"),
        ("vz", "<f4"),
    ],
    align=False,
)

HEADER_DTYPE = np.dtype(
    [
        ("frame_number", "<u4"),
        ("timestamp", "<f8"),
        ("drone_count", "<u2"),
        ("_pad", "<u2"),
    ],
    align=False,
)

HEADER_SIZE = HEADER_DTYPE.itemsize  # 16
RECORD_SIZE = DRONE_RECORD_DTYPE.itemsize  # 30

assert HEADER_SIZE == 16
assert RECORD_SIZE == 30


def _unpack_frame(data: bytes) -> list[dict]:
    """Decode a drone backend binary frame into a list of drone dicts."""
    if len(data) < HEADER_SIZE:
        return []
    header = np.frombuffer(data[:HEADER_SIZE], dtype=HEADER_DTYPE)[0]
    count = int(header["drone_count"])
    if count == 0:
        return []
    rec_bytes = data[HEADER_SIZE : HEADER_SIZE + count * RECORD_SIZE]
    records = np.frombuffer(rec_bytes, dtype=DRONE_RECORD_DTYPE)
    return [
        {
            "id": int(r["id"]),
            "state": int(r["state"]),
            "x": float(r["x"]),
            "y": float(r["y"]),
            "z": float(r["z"]),
        }
        for r in records
    ]


def _pack_frame(frame_number: int, missiles: dict) -> bytes:
    n = len(missiles)
    header = np.zeros(1, dtype=HEADER_DTYPE)
    header["frame_number"] = frame_number
    header["timestamp"] = time.monotonic()
    header["drone_count"] = n
    if n == 0:
        return header.tobytes()
    records = np.zeros(n, dtype=DRONE_RECORD_DTYPE)
    for i, (mid, m) in enumerate(missiles.items()):
        records[i]["id"] = mid & 0xFFFF
        records[i]["state"] = 0
        records[i]["x"] = m["pos"][0]
        records[i]["y"] = m["pos"][1]
        records[i]["z"] = m["pos"][2]
        records[i]["vx"] = m["vel"][0]
        records[i]["vy"] = m["vel"][1]
        records[i]["vz"] = m["vel"][2]
    return header.tobytes() + records.tobytes()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DRONE_WS_URL = "ws://10.154.139.105:3000/simulation/fake/stream"
MISSILE_SPEED_MPS = 250.0  # m/s — faster than drones
INTERCEPT_RADIUS_M = 40.0  # metres — close enough counts as a hit
SIM_HZ = 60
SIM_DT = 1.0 / SIM_HZ
STREAM_HZ = 20

# SAM site positions (x, y, z) in metres — set via PUT /sam/positions.
_sam_positions: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_drone_positions: dict[int, tuple[float, float, float]] = {}  # drone_id → (x,y,z)
_missiles: dict[int, dict] = {}  # missile_id → {target_id, pos, vel, sam}
_next_missile_id: int = 0
_clients: set[WebSocket] = set()
_frame_number: int = 0

# ---------------------------------------------------------------------------
# Drone subscriber task
# ---------------------------------------------------------------------------


async def _drone_subscriber() -> None:
    """Continuously connect to the drone backend and update _drone_positions."""
    global _drone_positions
    while True:
        try:
            log.info("connecting to drone backend: %s", DRONE_WS_URL)
            async with websockets.connect(DRONE_WS_URL) as ws:
                log.info("connected to drone backend")
                async for message in ws:
                    if not isinstance(message, bytes):
                        continue
                    drones = _unpack_frame(message)
                    alive = {d["id"] for d in drones if d["state"] == 0}
                    _drone_positions = {
                        d["id"]: (d["x"], d["y"], d["z"])
                        for d in drones
                        if d["state"] == 0
                    }
                    # Remove missiles targeting dead/gone drones.
                    for mid in [
                        k for k, m in _missiles.items() if m["target_id"] not in alive
                    ]:
                        del _missiles[mid]
        except Exception as e:
            log.warning("drone backend disconnected (%s), retrying in 2s", e)
            _drone_positions.clear()
            await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Missile simulation task
# ---------------------------------------------------------------------------



async def _missile_sim() -> None:
    global _next_missile_id, _frame_number
    last_tick = time.monotonic()

    while True:
        await asyncio.sleep(SIM_DT)
        now = time.monotonic()
        dt = now - last_tick
        last_tick = now

        # Spawn one missile per SAM site per drone.
        covered = {(m["target_id"], m["sam_idx"]) for m in _missiles.values()}
        for drone_id in list(_drone_positions.keys()):
            for sam_idx, (sx, sy, sz) in enumerate(_sam_positions):
                if (drone_id, sam_idx) in covered:
                    continue
                mid = _next_missile_id
                _next_missile_id += 1
                _missiles[mid] = {
                    "target_id": drone_id,
                    "sam_idx":   sam_idx,
                    "pos": [sx, sy, sz],
                    "vel": [0.0, 0.0, 0.0],
                }
                log.info(
                    "missile %d launched at drone %d from SAM %d (%.0f, %.0f, %.0f)",
                    mid, drone_id, sam_idx, sx, sy, sz,
                )

        # Step each missile toward its target.
        for mid in list(_missiles.keys()):
            m = _missiles[mid]
            target_id = m["target_id"]
            if target_id not in _drone_positions:
                del _missiles[mid]
                continue
            tx, ty, tz = _drone_positions[target_id]
            px, py, pz = m["pos"]
            dx, dy, dz = tx - px, ty - py, tz - pz
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < INTERCEPT_RADIUS_M:
                log.info("missile %d intercepted drone %d", mid, target_id)
                del _missiles[mid]
                continue
            spd = MISSILE_SPEED_MPS
            vx, vy, vz = (dx / dist) * spd, (dy / dist) * spd, (dz / dist) * spd
            m["vel"] = [vx, vy, vz]
            m["pos"] = [px + vx * dt, py + vy * dt, pz + vz * dt]


# ---------------------------------------------------------------------------
# Broadcast task
# ---------------------------------------------------------------------------


async def _broadcaster() -> None:
    global _frame_number
    dt = 1.0 / STREAM_HZ
    while True:
        await asyncio.sleep(dt)
        if not _clients:
            continue
        payload = _pack_frame(_frame_number, _missiles)
        _frame_number += 1
        dead = set()
        for ws in _clients:
            try:
                await ws.send_bytes(payload)
            except Exception:
                dead.add(ws)
        _clients.difference_update(dead)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_drone_subscriber())
    asyncio.create_task(_missile_sim())
    asyncio.create_task(_broadcaster())
    yield


app = FastAPI(title="Missile Defence Server", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"service": "missile-defence", "status": "ok"}


@app.put("/sam/positions")
async def set_sam_positions(body: dict):
    """
    Accept SAM site positions from the frontend.
    Body: { "positions": [[x1, y1, z1], [x2, y2, z2], ...] }
    Coordinates are in server metres (same system as drones).
    """
    global _sam_positions
    raw = body.get("positions", [[0.0, 0.0, 0.0]])
    _sam_positions = [
        (float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 0.0)
        for p in raw if len(p) >= 2
    ]
    log.info("SAM positions updated: %s", _sam_positions)
    return {"sam_positions": _sam_positions}


@app.websocket("/missiles/stream")
async def missile_stream(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    log.info("missile client connected (total=%d)", len(_clients))
    try:
        while True:
            await websocket.receive_text()  # keep alive, ignore input
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)
        log.info("missile client disconnected (total=%d)", len(_clients))

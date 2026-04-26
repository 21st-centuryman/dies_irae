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
import math
import random
import time
from contextlib import asynccontextmanager
from urllib.request import Request, urlopen

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

DRONE_WS_URL = "ws://localhost:3000/simulation/fake/stream"
DRONE_HTTP_URL = "http://localhost:3000"
HEIGHT_SERVER_URL = "http://10.154.6.177:8000"
MISSILE_SPEED_MPS = 125.0  # m/s base speed
INTERCEPT_RADIUS_M = 40.0  # metres — close enough counts as a hit
KILL_PROBABILITY    = 0.4     # LR SAM hit chance
SR_KILL_PROBABILITY = 0.8     # SR SAM hit chance
SR_MAX_RANGE_M      = 2000.0  # SR SAM only engages drones within this distance
SIM_HZ = 60
SIM_DT = 1.0 / SIM_HZ
STREAM_HZ = 20

CLIMB_HEIGHT_M = 40.0  # metres to climb straight up before seeking target

# SAM site positions — (x, y, z, type) where type is "LR" or "SR".
_sam_positions: list[tuple[float, float, float, str]] = [(0.0, 0.0, 0.0, "LR")]

# ---------------------------------------------------------------------------
# Simple heightmap (bilinear, sim coords: X=East Y=North in metres from origin)
# ---------------------------------------------------------------------------

_M_PER_DEG_LAT = 111_320.0


class _SimpleHeightmap:
    """Minimal bilinear-sampled heightmap in sim coords (X=East, Y=North, metres)."""

    def __init__(
        self,
        array: np.ndarray,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        origin_lat: float,
        origin_lon: float,
    ) -> None:
        self.array = array.astype(np.float32)
        self.height_px, self.width_px = array.shape
        m_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(origin_lat))
        bbox_dlon = lon_max - lon_min
        bbox_dlat = lat_max - lat_min
        self.m_per_px_x = (bbox_dlon * m_per_deg_lon) / self.width_px
        self.m_per_px_y = (bbox_dlat * _M_PER_DEG_LAT) / self.height_px
        self.origin_col = ((origin_lon - lon_min) / bbox_dlon) * (self.width_px - 1)
        self.origin_row = ((lat_max - origin_lat) / bbox_dlat) * (self.height_px - 1)

    def height_at(self, x: float, y: float) -> float:
        col = self.origin_col + x / self.m_per_px_x
        row = self.origin_row - y / self.m_per_px_y
        col = max(0.0, min(col, self.width_px - 2.0))
        row = max(0.0, min(row, self.height_px - 2.0))
        c0, r0 = int(col), int(row)
        fc, fr = col - c0, row - r0
        a = self.array
        h = (a[r0, c0] * (1 - fc) + a[r0, c0 + 1] * fc) * (1 - fr) + (
            a[r0 + 1, c0] * (1 - fc) + a[r0 + 1, c0 + 1] * fc
        ) * fr
        return float(h)


_heightmap: _SimpleHeightmap | None = None


def _fetch_heightmap(lat: float, lon: float) -> _SimpleHeightmap | None:
    """Synchronously fetch heightmap from height server. Returns None on failure."""
    try:
        url = f"{HEIGHT_SERVER_URL}/fetch?lat={lat}&lon={lon}"
        log.info("fetching heightmap from %s", url)
        with urlopen(url, timeout=30) as resp:
            headers = resp.headers
            width = int(headers["X-Width"])
            height = int(headers["X-Height"])
            dtype_str = headers["X-Dtype"]
            bbox = [float(s) for s in headers["X-BBox"].split(",")]
            body = resp.read()
        if dtype_str != "float16":
            log.warning("unexpected heightmap dtype: %s", dtype_str)
            return None
        arr = np.frombuffer(body, dtype="<f2").reshape((height, width))
        lon_min, lat_min, lon_max, lat_max = bbox
        log.info(
            "heightmap %dx%d bbox=(%.4f..%.4f, %.4f..%.4f) min=%.1f max=%.1f",
            width,
            height,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            float(arr.min()),
            float(arr.max()),
        )
        return _SimpleHeightmap(arr, lon_min, lat_min, lon_max, lat_max, lat, lon)
    except Exception as e:
        log.warning("failed to fetch heightmap: %s", e)
        return None


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_drone_positions: dict[int, tuple[float, float, float]] = {}  # drone_id → (x,y,z)
_missiles: dict[int, dict] = {}  # missile_id → {target_id, pos, vel, sam, spawn_z}
_next_missile_id: int = 0
_clients: set[WebSocket] = set()
_frame_number: int = 0
# (target_id, sam_idx) pairs blocked after terrain collision — cleared on scenario reset.
_terrain_blocked: set[tuple[int, int]] = set()
_last_launch_time: dict[int, float] = {}  # sam_idx → monotonic time of last launch
LAUNCH_INTERVAL_S: float = 2.0  # minimum seconds between launches per site
_speed_mult: float = 1.0         # mirrors the frontend speed multiplier

SR_MAGAZINE = 10   # total missiles per SR SAM site
LR_MAGAZINE = 24   # total missiles per LR SAM site
_magazine: dict[int, int] = {}   # sam_idx → remaining missiles
_missiles_fired: int = 0         # total launched this scenario

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


async def _kill_drone(drone_id: int) -> None:
    """Fire-and-forget PUT to the backend to mark a drone as destroyed."""
    def _do() -> None:
        req = Request(
            f"{DRONE_HTTP_URL}/drone/{drone_id}/kill",
            method="PUT",
        )
        try:
            with urlopen(req, timeout=5):
                pass
        except Exception as e:
            log.warning("kill request for drone %d failed: %s", drone_id, e)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do)


async def _missile_sim() -> None:
    global _next_missile_id, _frame_number, _missiles_fired
    last_tick = time.monotonic()

    while True:
        await asyncio.sleep(SIM_DT)
        now = time.monotonic()
        dt = now - last_tick
        last_tick = now

        covered = {(m["target_id"], m["sam_idx"]) for m in _missiles.values()}
        launches = []

        # Decide launches first: one per SAM site, same tick
        for sam_idx, (sx, sy, sz, sam_type) in enumerate(_sam_positions):
            if now - _last_launch_time.get(sam_idx, 0.0) < LAUNCH_INTERVAL_S:
                continue
            remaining = _magazine.get(sam_idx, 0)
            if remaining <= 0:
                continue  # site is out of missiles

            sr_range2 = SR_MAX_RANGE_M ** 2  # compare squared distances

            # Pick the closest drone to this SAM site that isn't already targeted.
            chosen_drone_id = None
            best_dist2 = float("inf")
            for drone_id, (ddx, ddy, ddz) in _drone_positions.items():
                if (drone_id, sam_idx) in covered:
                    continue
                if (drone_id, sam_idx) in _terrain_blocked:
                    continue
                d2 = (ddx - sx) ** 2 + (ddy - sy) ** 2 + (ddz - sz) ** 2
                if sam_type == "SR" and d2 > sr_range2:
                    continue  # drone out of SR range
                if d2 < best_dist2:
                    best_dist2 = d2
                    chosen_drone_id = drone_id

            if chosen_drone_id is not None:
                launches.append((sam_idx, chosen_drone_id, sx, sy, sz, sam_type))
                covered.add((chosen_drone_id, sam_idx))

        # Commit launches together
        for sam_idx, drone_id, sx, sy, sz, sam_type in launches:
            _magazine[sam_idx] -= 1
            _missiles_fired += 1
            mid = _next_missile_id
            _next_missile_id += 1
            _missiles[mid] = {
                "target_id": drone_id,
                "sam_idx":   sam_idx,
                "sam_type":  sam_type,
                "pos":     [sx, sy, sz],
                "vel":     [0.0, 0.0, MISSILE_SPEED_MPS],
                "spawn_z": sz,
            }
            _last_launch_time[sam_idx] = now
            left = _magazine[sam_idx]
            log.info(
                "missile %d launched at drone %d from SAM %d (%s) — %d remaining",
                mid, drone_id, sam_idx, sam_type, left,
            )
            if left == 0:
                log.info("SAM %d (%s) magazine exhausted", sam_idx, sam_type)

        # Step each missile.
        for mid in list(_missiles.keys()):
            m = _missiles[mid]
            target_id = m["target_id"]
            if target_id not in _drone_positions:
                del _missiles[mid]
                continue

            px, py, pz = m["pos"]
            tx, ty, tz = _drone_positions[target_id]
            dx, dy, dz = tx - px, ty - py, tz - pz
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < INTERCEPT_RADIUS_M:
                kill_prob = SR_KILL_PROBABILITY if m.get("sam_type") == "SR" else KILL_PROBABILITY
                del _missiles[mid]
                if random.random() < kill_prob:
                    log.info("missile %d KILLED drone %d", mid, target_id)
                    asyncio.create_task(_kill_drone(target_id))
                else:
                    log.info("missile %d missed drone %d (within radius, no kill)", mid, target_id)
                continue

            t = min((pz - m["spawn_z"]) / CLIMB_HEIGHT_M, 1.0)
            inv = 1.0 / max(dist, 1e-6)
            tdx, tdy, tdz = dx * inv, dy * inv, dz * inv

            bx = t * tdx
            by = t * tdy
            bz = (1.0 - t) + t * tdz
            bmag = (bx * bx + by * by + bz * bz) ** 0.5
            bx, by, bz = bx / bmag, by / bmag, bz / bmag

            spd = MISSILE_SPEED_MPS * _speed_mult
            vx, vy, vz = bx * spd, by * spd, bz * spd
            m["vel"] = [vx, vy, vz]
            new_pos = [px + vx * dt, py + vy * dt, pz + vz * dt]
            m["pos"] = new_pos

            if _heightmap is not None:
                ground_z = _heightmap.height_at(new_pos[0], new_pos[1])
                if new_pos[2] < ground_z:
                    log.info(
                        "missile %d hit terrain at (%.0f, %.0f, %.0f)", mid, *new_pos
                    )
                    _terrain_blocked.add((target_id, m["sam_idx"]))
                    del _missiles[mid]


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


@app.put("/simulation/speed")
async def set_speed(body: dict):
    global _speed_mult
    _speed_mult = max(0.0, float(body.get("speed", 1)))
    return {"speed_mult": _speed_mult}


@app.put("/sam/positions")
async def set_sam_positions(body: dict):
    global _sam_positions, _heightmap, _last_launch_time, _missiles_fired
    raw = body.get("positions", [[0.0, 0.0, 0.0]])
    _sam_positions = [
        (float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 0.0,
         str(p[3]) if len(p) >= 4 else "LR")
        for p in raw
        if len(p) >= 2
    ]
    _terrain_blocked.clear()
    _last_launch_time.clear()
    _magazine.clear()
    _missiles_fired = 0
    for idx, (_, _, _, sam_type) in enumerate(_sam_positions):
        _magazine[idx] = SR_MAGAZINE if sam_type == "SR" else LR_MAGAZINE
    log.info("SAM positions updated: %s", _sam_positions)
    log.info("magazines: %s", _magazine)

    lat = body.get("lat")
    lon = body.get("lon")
    if lat is not None and lon is not None:
        loop = asyncio.get_event_loop()
        hmap = await loop.run_in_executor(
            None, _fetch_heightmap, float(lat), float(lon)
        )
        if hmap is not None:
            _heightmap = hmap
            log.info("heightmap loaded for terrain collision")
        else:
            log.warning("could not load heightmap; missile terrain collision disabled")

    return {"sam_positions": _sam_positions}


@app.get("/missiles/stats")
async def missile_stats():
    return {
        "fired":     _missiles_fired,
        "magazine":  {str(k): v for k, v in _magazine.items()},
    }


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

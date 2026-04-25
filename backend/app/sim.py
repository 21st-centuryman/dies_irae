"""Simulation core: wire protocol, drone state, scenario models, and engine."""

from __future__ import annotations

import math
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from app.terrain import Heightmap

# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

DRONE_RECORD_DTYPE = np.dtype(
    [
        ("id",     "<u2"),
        ("type",   "<u1"),
        ("state",  "<u1"),
        ("intent", "<u1"),
        ("_pad",   "<u1"),
        ("x",  "<f4"), ("y",  "<f4"), ("z",  "<f4"),
        ("vx", "<f4"), ("vy", "<f4"), ("vz", "<f4"),
    ],
    align=False,
)

HEADER_DTYPE = np.dtype(
    [
        ("frame_number", "<u4"),
        ("timestamp",    "<f8"),
        ("drone_count",  "<u2"),
        ("_pad",         "<u2"),
    ],
    align=False,
)

DRONE_RECORD_SIZE = DRONE_RECORD_DTYPE.itemsize  # 30
HEADER_SIZE       = HEADER_DTYPE.itemsize        # 16

assert DRONE_RECORD_SIZE == 30
assert HEADER_SIZE       == 16


def pack_frame(frame_number: int, timestamp: float, records: np.ndarray) -> bytes:
    if records.dtype != DRONE_RECORD_DTYPE:
        raise TypeError(f"records dtype {records.dtype} != {DRONE_RECORD_DTYPE}")
    if records.ndim != 1:
        raise ValueError("records must be 1-D")
    header = np.zeros(1, dtype=HEADER_DTYPE)
    header["frame_number"] = frame_number
    header["timestamp"]    = timestamp
    header["drone_count"]  = records.shape[0]
    return header.tobytes() + np.ascontiguousarray(records).tobytes()


# ---------------------------------------------------------------------------
# Drone state enums
# ---------------------------------------------------------------------------

ACTIVE    = 0
DESTROYED = 1
REACHED   = 2

INTENT_SEEKING          = 0
INTENT_AVOIDING         = 1
INTENT_DIVING           = 2
INTENT_ORBITING         = 3
INTENT_TERRAIN_FOLLOWING = 4


# ---------------------------------------------------------------------------
# Drone state store (structure-of-arrays)
# ---------------------------------------------------------------------------

class DroneState:
    def __init__(self, capacity: int) -> None:
        self.capacity  = capacity
        self.n         = 0
        self.positions  = np.zeros((capacity, 3), dtype=np.float32)
        self.velocities = np.zeros((capacity, 3), dtype=np.float32)
        self.types      = np.zeros(capacity, dtype=np.uint8)
        self.states     = np.zeros(capacity, dtype=np.uint8)
        self.intents    = np.zeros(capacity, dtype=np.uint8)
        self.ids        = np.arange(capacity, dtype=np.uint16)
        self._records   = np.zeros(capacity, dtype=DRONE_RECORD_DTYPE)

    def spawn(
        self,
        position,
        velocity=(0.0, 0.0, 0.0),
        type_: int = 0,
        state: int = ACTIVE,
        intent: int = INTENT_SEEKING,
    ) -> int:
        if self.n >= self.capacity:
            raise RuntimeError(f"DroneState full (capacity={self.capacity})")
        i = self.n
        self.positions[i]  = position
        self.velocities[i] = velocity
        self.types[i]      = type_
        self.states[i]     = state
        self.intents[i]    = intent
        self.n += 1
        return i

    def to_records(self) -> np.ndarray:
        n   = self.n
        rec = self._records[:n]
        rec["id"]     = self.ids[:n]
        rec["type"]   = self.types[:n]
        rec["state"]  = self.states[:n]
        rec["intent"] = self.intents[:n]
        rec["x"]  = self.positions[:n, 0]
        rec["y"]  = self.positions[:n, 1]
        rec["z"]  = self.positions[:n, 2]
        rec["vx"] = self.velocities[:n, 0]
        rec["vy"] = self.velocities[:n, 1]
        rec["vz"] = self.velocities[:n, 2]
        return rec


# ---------------------------------------------------------------------------
# Scenario config and session
# ---------------------------------------------------------------------------

class ScenarioConfig(BaseModel):
    drone_count:   int   = Field(ge=0)
    pct_attack:    int   = 50
    pct_recon:     int   = 50
    pct_short:     int   = 50
    pct_long:      int   = 50
    scenario_type: str   = "allatonce"
    spawn_window:  int   = 0
    wave_interval: int   = 5
    lat:   float
    lon:   float
    targets: list = Field(default_factory=lambda: [[0.0, 0.0]])

    model_config = {"extra": "allow"}


@dataclass
class ScenarioSession:
    config:      ScenarioConfig
    engine:      "SimEngine"
    started_at:  float                    = field(default_factory=time.monotonic)
    engine_task: Optional[asyncio.Task]   = None  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Simulation engine constants
# ---------------------------------------------------------------------------

SIM_HZ   = 60
SIM_DT   = 1.0 / SIM_HZ
STREAM_HZ = 20

REACH_THRESHOLD_M    = 30.0
SPAWN_ALTITUDE_AGL_M = 100.0
DEFAULT_RING_RADIUS_M = 5000.0
MIN_AGL_CLEARANCE_M  = 0.0
MAX_SIM_DURATION_S   = 600.0


@dataclass(frozen=True)
class DroneTypeParams:
    name:           str
    max_speed_mps:  float
    max_accel_mps2: float
    max_alt_m:      float


DRONE_TYPES: tuple[DroneTypeParams, ...] = (
    DroneTypeParams("small_fpv",          25.0, 15.0,  500.0),  # 90 km/h
    DroneTypeParams("fpv_fiber",          25.0, 12.0,  400.0),  # 90 km/h
    DroneTypeParams("loitering_munition", 25.0,  5.0, 2000.0),  # 90 km/h
    DroneTypeParams("surveillance",       25.0,  8.0, 1500.0),  # 90 km/h
)


def compass_to_math_angle(bearing_deg: float) -> float:
    return math.pi / 2.0 - math.radians(bearing_deg)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

class SimEngine:
    def __init__(
        self,
        heightmap: Heightmap,
        target_xy: tuple[float, float],
        ring_radius_m: float = DEFAULT_RING_RADIUS_M,
        spawn_bearing_deg: float = 0.0,
        spawn_altitude_agl_m: float = SPAWN_ALTITUDE_AGL_M,
        drone_type: int = 0,
    ) -> None:
        self.heightmap  = heightmap
        self.target_xy  = (float(target_xy[0]), float(target_xy[1]))
        self.ring_radius = float(ring_radius_m)
        self.drone_type  = drone_type

        params          = DRONE_TYPES[drone_type]
        self.max_speed  = params.max_speed_mps
        self.max_accel  = params.max_accel_mps2

        self.state = DroneState(capacity=1)

        theta   = compass_to_math_angle(spawn_bearing_deg)
        spawn_x = self.ring_radius * math.cos(theta)
        spawn_y = self.ring_radius * math.sin(theta)
        spawn_ground = float(heightmap.height_at(spawn_x, spawn_y))
        spawn_z      = spawn_ground + spawn_altitude_agl_m
        self.spawn_xyz = (spawn_x, spawn_y, spawn_z)

        self.state.spawn(
            position=self.spawn_xyz,
            velocity=(0.0, 0.0, 0.0),
            type_=drone_type,
            state=ACTIVE,
            intent=INTENT_SEEKING,
        )

        target_ground    = float(heightmap.height_at(self.target_xy[0], self.target_xy[1]))
        self._target_3d  = np.array(
            [self.target_xy[0], self.target_xy[1], target_ground], dtype=np.float32
        )
        self.target_z    = float(self._target_3d[2])
        self.t           = 0.0
        self.finished    = False
        self.finish_reason: str | None = None

    def step(self) -> None:
        if self.finished:
            return

        n      = self.state.n
        pos    = self.state.positions[:n]
        vel    = self.state.velocities[:n]
        states = self.state.states[:n]

        active = states == ACTIVE
        if not active.any():
            self.finished     = True
            self.finish_reason = "no_active_drones"
            return

        delta   = self._target_3d - pos
        dist_xy = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)  # XY only

        reached_now = active & (dist_xy < REACH_THRESHOLD_M)
        if reached_now.any():
            idx = np.where(reached_now)[0]
            states[idx] = REACHED
            vel[idx]    = 0.0

        seeking = active & ~reached_now
        if seeking.any():
            d     = delta[seeking]
            dist3 = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
            desired_vel   = (d / dist3) * self.max_speed
            desired_accel = (desired_vel - vel[seeking]) / SIM_DT
            accel_mag     = np.linalg.norm(desired_accel, axis=1, keepdims=True)
            accel_scale   = np.minimum(1.0, self.max_accel / np.maximum(accel_mag, 1e-6))
            desired_accel *= accel_scale
            vel[seeking]  += desired_accel * SIM_DT
            speed          = np.linalg.norm(vel[seeking], axis=1, keepdims=True)
            vel[seeking]  *= np.minimum(1.0, self.max_speed / np.maximum(speed, 1e-6))
            pos[seeking]  += vel[seeking] * SIM_DT

        ground = self.heightmap.height_at_batch(pos[:, 0], pos[:, 1])
        min_z  = ground + MIN_AGL_CLEARANCE_M
        below  = pos[:, 2] < min_z
        if below.any():
            pos[below, 2] = min_z[below]
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)

        self.t += SIM_DT

        if (states == REACHED).any():
            self.finished      = True
            self.finish_reason = "reached_target"
        elif self.t > MAX_SIM_DURATION_S:
            self.finished      = True
            self.finish_reason = "max_duration"

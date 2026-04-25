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

SIM_HZ   = 10
SIM_DT   = 1.0 / SIM_HZ
STREAM_HZ = 20

REACH_THRESHOLD_M    = 30.0
SPAWN_ALTITUDE_AGL_M = 0.0
DEFAULT_RING_RADIUS_M = 5000.0
MIN_AGL_CLEARANCE_M  = 0.0

# Boids-style terrain avoidance: directional-corridor search.
# Each step we sweep a fan of candidate headings in the horizontal plane,
# ray-march the heightmap along each, and pick the lowest-cost corridor
# (blockedness + bearing-to-target misalignment + heading-change penalty).
# The chosen corridor produces a single desired velocity, which is then
# steered toward in the usual `(desired - current) / dt` clamped fashion.
TERRAIN_SAFETY_AGL_M  = 50.0
HEADING_FAN_DEG       = (
    0.0, 10.0, -10.0, 20.0, -20.0, 35.0, -35.0,
    55.0, -55.0, 80.0, -80.0, 120.0, -120.0,
)
HORIZON_TIME_S        = 8.0   # how far along each candidate to ray-march
PROBE_COUNT           = 10    # number of samples per ray
NEAR_HORIZON_TIME_S   = 3.0   # window used to set terrain-follow altitude
SOFT_MARGIN_M         = 20.0  # extra clearance band that still scores zero
COST_DEFICIT_WEIGHT   = 10.0  # per meter of clearance deficit
COST_GOAL_WEIGHT      = 50.0  # per radian of bearing-to-target misalignment
COST_TURN_WEIGHT      = 30.0  # per radian of heading change (hysteresis)
MAX_YAW_RATE_DEG_S    = 120.0 # cap on heading change per second
ALT_TRACK_TAU_S       = 1.5   # gap-closing time constant for vertical track


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

    def _choose_desired_velocity(
        self, pos: np.ndarray, vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pick a desired velocity by scoring a fan of candidate headings.

        For each drone:
          1. Build C candidate headings = current_heading + fan_offsets.
          2. Ray-march the heightmap along each candidate out to a horizon
             of `max_speed * HORIZON_TIME_S`. P samples per ray.
          3. Compare required altitude (terrain + safety) against the
             drone's flyable altitude profile (assuming max climb). The
             worst sample on each ray is its blockedness.
          4. Cost = blockedness + |bearing-to-target offset| + |turn|.
             Lowest cost wins.
          5. Apply yaw-rate limit between current heading and chosen heading.
          6. Vertical desired velocity tracks (peak terrain in the near
             portion of the chosen corridor) + safety AGL.

        Returns (desired_v, avoiding_mask). desired_v has shape (n, 3).
        """
        n = pos.shape[0]
        rows = np.arange(n)

        # ── 1. Reference headings ─────────────────────────────────────────
        h_speed = np.linalg.norm(vel[:, :2], axis=1)
        moving  = h_speed > 1.0
        cur_theta = np.zeros(n, dtype=np.float32)
        cur_theta[moving] = np.arctan2(vel[moving, 1], vel[moving, 0])
        # Bearing from each drone to the target (XY).
        to_tgt = self._target_3d[:2] - pos[:, :2]
        tgt_theta = np.arctan2(to_tgt[:, 1], to_tgt[:, 0]).astype(np.float32)
        # Cold start (drone stationary): treat target bearing as current heading
        # so the fan is centered on the goal direction.
        cur_theta = np.where(moving, cur_theta, tgt_theta)

        fan = np.array([math.radians(d) for d in HEADING_FAN_DEG], dtype=np.float32)
        cand_theta = cur_theta[:, None] + fan[None, :]              # (n, C)
        cand_dx    = np.cos(cand_theta).astype(np.float32)
        cand_dy    = np.sin(cand_theta).astype(np.float32)

        # ── 2. Ray-march samples ──────────────────────────────────────────
        horizon_m = self.max_speed * HORIZON_TIME_S
        probe_d   = np.linspace(
            horizon_m / PROBE_COUNT, horizon_m, PROBE_COUNT, dtype=np.float32,
        )                                                            # (P,)

        # Probe positions broadcast to (n, C, P).
        probe_x = pos[:, None, None, 0] + cand_dx[:, :, None] * probe_d[None, None, :]
        probe_y = pos[:, None, None, 1] + cand_dy[:, :, None] * probe_d[None, None, :]
        terrain = self.heightmap.height_at_batch(
            probe_x.reshape(-1), probe_y.reshape(-1),
        ).reshape(probe_x.shape).astype(np.float32)                  # (n, C, P)

        # ── 3. Blockedness per candidate ──────────────────────────────────
        # Drone's flyable altitude over time, assuming max climb from now,
        # capped at the drone's altitude ceiling.
        max_alt   = float(DRONE_TYPES[self.drone_type].max_alt_m)
        probe_t   = (probe_d / self.max_speed).astype(np.float32)    # (P,)
        flyable_z = pos[:, None, None, 2] + self.max_speed * probe_t[None, None, :]
        flyable_z = np.minimum(flyable_z, np.float32(max_alt))       # (n, 1, P) bcast

        deficit       = (terrain + TERRAIN_SAFETY_AGL_M) - flyable_z  # (n, C, P)
        worst_deficit = deficit.max(axis=2)                           # (n, C)

        # ── 4. Cost ───────────────────────────────────────────────────────
        # Wrap angle differences into [-pi, pi].
        def _wrap(a: np.ndarray) -> np.ndarray:
            return (a + np.pi) % (2.0 * np.pi) - np.pi

        goal_diff = _wrap(cand_theta - tgt_theta[:, None])
        turn_diff = _wrap(cand_theta - cur_theta[:, None])

        # Soft-margin band: corridors with < SOFT_MARGIN_M of clearance above
        # the safety buffer still score nonzero, biasing toward roomy paths.
        blockedness = np.maximum(worst_deficit + SOFT_MARGIN_M, 0.0)

        cost = (
            COST_DEFICIT_WEIGHT * blockedness
            + COST_GOAL_WEIGHT  * np.abs(goal_diff)
            + COST_TURN_WEIGHT  * np.abs(turn_diff)
        )                                                             # (n, C)

        best_idx       = np.argmin(cost, axis=1)                      # (n,)
        chosen_theta   = cand_theta[rows, best_idx]
        chosen_deficit = worst_deficit[rows, best_idx]

        # ── 5. Yaw-rate limit ─────────────────────────────────────────────
        max_step  = math.radians(MAX_YAW_RATE_DEG_S) * SIM_DT
        delta     = _wrap(chosen_theta - cur_theta)
        delta     = np.clip(delta, -max_step, max_step)
        new_theta = cur_theta + delta

        desired_vx = np.cos(new_theta).astype(np.float32) * self.max_speed
        desired_vy = np.sin(new_theta).astype(np.float32) * self.max_speed

        # ── 6. Vertical: terrain-follow within near horizon of chosen ray ─
        near_count = max(1, int(round(PROBE_COUNT * NEAR_HORIZON_TIME_S / HORIZON_TIME_S)))
        chosen_terrain = terrain[rows, best_idx, :]                   # (n, P)
        near_peak      = chosen_terrain[:, :near_count].max(axis=1)
        desired_z      = np.minimum(near_peak + TERRAIN_SAFETY_AGL_M, np.float32(max_alt))
        # Also pull toward target altitude when no terrain pressure (so the
        # drone descends into the target rather than cruising high forever).
        # Blend: alt_target = max(terrain_follow, descent_toward_target).
        # Closing rate proportional to remaining XY distance keeps it gentle.
        dist_xy = np.linalg.norm(to_tgt, axis=1)
        glide_z = self._target_3d[2] + np.maximum(dist_xy * 0.05, 0.0)  # ~3° glideslope
        desired_z = np.maximum(desired_z, np.minimum(glide_z, pos[:, 2]))
        # Predictive vz that closes the altitude gap over ALT_TRACK_TAU_S.
        desired_vz = np.clip(
            (desired_z - pos[:, 2]) / ALT_TRACK_TAU_S,
            -self.max_speed, self.max_speed,
        ).astype(np.float32)

        desired_v = np.empty((n, 3), dtype=np.float32)
        desired_v[:, 0] = desired_vx
        desired_v[:, 1] = desired_vy
        desired_v[:, 2] = desired_vz

        # Avoidance is "in effect" whenever the chosen corridor still has any
        # blockage, OR the chosen heading deviates meaningfully from the bee
        # line to the target (we deflected to find a clear path).
        chosen_goal_diff = _wrap(chosen_theta - tgt_theta)
        avoiding = (chosen_deficit > 0.0) | (np.abs(chosen_goal_diff) > math.radians(5.0))
        return desired_v, avoiding

    def step(self) -> None:
        if self.finished:
            return

        n       = self.state.n
        pos     = self.state.positions[:n]
        vel     = self.state.velocities[:n]
        states  = self.state.states[:n]
        intents = self.state.intents[:n]

        active = states == ACTIVE
        if not active.any():
            self.finished     = True
            self.finish_reason = "no_active_drones"
            return

        # Reach check first so we don't steer drones that are already there.
        delta       = self._target_3d - pos
        dist_xy     = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        reached_now = active & (dist_xy < REACH_THRESHOLD_M)
        if reached_now.any():
            idx = np.where(reached_now)[0]
            states[idx] = REACHED
            vel[idx]    = 0.0

        steering = active & ~reached_now
        if steering.any():
            # Single desired velocity from corridor search (seek + avoid fused).
            desired_v, avoiding = self._choose_desired_velocity(pos, vel)

            # Steering accel = (desired − current) / dt, clamped at max_accel.
            accel = (desired_v - vel) / SIM_DT
            mag   = np.linalg.norm(accel, axis=1, keepdims=True)
            accel *= np.minimum(1.0, self.max_accel / np.maximum(mag, 1e-6))

            vel[steering] += accel[steering] * SIM_DT
            speed = np.linalg.norm(vel[steering], axis=1, keepdims=True)
            vel[steering] *= np.minimum(1.0, self.max_speed / np.maximum(speed, 1e-6))
            pos[steering] += vel[steering] * SIM_DT

            intents[steering & avoiding]  = INTENT_AVOIDING
            intents[steering & ~avoiding] = INTENT_SEEKING

        # Hard floor failsafe: avoidance is predictive and can still be beaten
        # by extreme slopes. Never let a drone clip below the ground.
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

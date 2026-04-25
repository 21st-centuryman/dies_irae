"""Phase-3 single-drone simulation engine.

A drone spawns at the ring perimeter (compass bearing 0° = North by default),
seeks a 2D target point at terminal altitude AGL, integrates physics with
kinematic limits from the drone-type table, and terminates when within
REACH_THRESHOLD_M (XY) of the target.

- Internal tick: 60 Hz fixed timestep (ARCHITECTURE §3).
- Streaming layer reads state at STREAM_HZ; the engine ticks itself.
- Vectorized over N drones from the start (Phase-3 N=1) so Phase-4 multi-
  drone work doesn't need a rewrite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.terrain import Heightmap

from .state import (
    ACTIVE,
    INTENT_SEEKING,
    REACHED,
    DroneState,
)

# --- Constants --------------------------------------------------------------

SIM_HZ = 60
SIM_DT = 1.0 / SIM_HZ
STREAM_HZ = 20
SIM_TICKS_PER_FRAME = SIM_HZ // STREAM_HZ  # 3

REACH_THRESHOLD_M = 30.0
SPAWN_ALTITUDE_AGL_M = 100.0
DEFAULT_RING_RADIUS_M = 5000.0
MIN_AGL_CLEARANCE_M = 0.0  # let the drone touch ground for kamikaze impact
MAX_SIM_DURATION_S = 600.0  # safety cap; should never trigger if seek converges


# --- Drone-type kinematic table (ARCHITECTURE "Drone types" section) -------


@dataclass(frozen=True)
class DroneTypeParams:
    name: str
    max_speed_mps: float
    max_accel_mps2: float
    max_alt_m: float


DRONE_TYPES: tuple[DroneTypeParams, ...] = (
    DroneTypeParams("small_fpv",          40.0, 15.0,  500.0),
    DroneTypeParams("fpv_fiber",          30.0, 12.0,  400.0),
    DroneTypeParams("loitering_munition", 20.0,  5.0, 2000.0),
    DroneTypeParams("surveillance",       25.0,  8.0, 1500.0),
)


def compass_to_math_angle(bearing_deg: float) -> float:
    """Compass bearing (0=N, 90=E) → math angle radians from +X (East)."""
    return math.pi / 2.0 - math.radians(bearing_deg)


# --- Engine -----------------------------------------------------------------


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
        self.heightmap = heightmap
        self.target_xy = (float(target_xy[0]), float(target_xy[1]))
        self.ring_radius = float(ring_radius_m)
        self.drone_type = drone_type

        params = DRONE_TYPES[drone_type]
        self.max_speed = params.max_speed_mps
        self.max_accel = params.max_accel_mps2

        self.state = DroneState(capacity=1)

        theta = compass_to_math_angle(spawn_bearing_deg)
        spawn_x = self.ring_radius * math.cos(theta)
        spawn_y = self.ring_radius * math.sin(theta)
        spawn_ground = float(heightmap.height_at(spawn_x, spawn_y))
        spawn_z = spawn_ground + spawn_altitude_agl_m
        self.spawn_xyz = (spawn_x, spawn_y, spawn_z)

        self.state.spawn(
            position=self.spawn_xyz,
            velocity=(0.0, 0.0, 0.0),
            type_=drone_type,
            state=ACTIVE,
            intent=INTENT_SEEKING,
        )

        # Target sits on the ground — Z is whatever the heightmap says at target XY.
        target_ground = float(heightmap.height_at(self.target_xy[0], self.target_xy[1]))
        self._target_3d = np.array(
            [self.target_xy[0], self.target_xy[1], target_ground],
            dtype=np.float32,
        )
        self.target_z = float(self._target_3d[2])

        self.t = 0.0
        self.finished = False
        self.finish_reason: str | None = None

    def step(self) -> None:
        if self.finished:
            return

        n = self.state.n
        pos = self.state.positions[:n]
        vel = self.state.velocities[:n]
        states = self.state.states[:n]

        active = states == ACTIVE
        if not active.any():
            self.finished = True
            self.finish_reason = "no_active_drones"
            return

        delta = self._target_3d - pos                     # (n, 3)
        dist3 = np.linalg.norm(delta, axis=1)             # (n,) — full 3D

        reached_now = active & (dist3 < REACH_THRESHOLD_M)
        if reached_now.any():
            idx = np.where(reached_now)[0]
            states[idx] = REACHED
            vel[idx] = 0.0

        seeking = active & ~reached_now
        if seeking.any():
            d = delta[seeking]
            dist3 = np.linalg.norm(d, axis=1, keepdims=True)
            dist3 = np.maximum(dist3, 1e-6)
            desired_vel = (d / dist3) * self.max_speed
            desired_accel = (desired_vel - vel[seeking]) / SIM_DT
            accel_mag = np.linalg.norm(desired_accel, axis=1, keepdims=True)
            accel_scale = np.minimum(1.0, self.max_accel / np.maximum(accel_mag, 1e-6))
            desired_accel = desired_accel * accel_scale
            vel[seeking] += desired_accel * SIM_DT
            speed = np.linalg.norm(vel[seeking], axis=1, keepdims=True)
            speed_scale = np.minimum(1.0, self.max_speed / np.maximum(speed, 1e-6))
            vel[seeking] *= speed_scale
            pos[seeking] += vel[seeking] * SIM_DT

        # Terrain clearance: refuse to fly underground. Prevents the
        # straight-line trajectory from clipping a hill before Phase-4
        # adds proper terrain avoidance.
        ground = self.heightmap.height_at_batch(pos[:, 0], pos[:, 1])
        min_z = ground + MIN_AGL_CLEARANCE_M
        below = pos[:, 2] < min_z
        if below.any():
            pos[below, 2] = min_z[below]
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)

        self.t += SIM_DT

        if (states == REACHED).any():
            self.finished = True
            self.finish_reason = "reached_target"
        elif self.t > MAX_SIM_DURATION_S:
            self.finished = True
            self.finish_reason = "max_duration"

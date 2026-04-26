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
    # Spawn-direction configuration. Compass azimuths in degrees (0° = N,
    # 90° = E). Empty `dir` list → drones spawn at random positions around
    # the perimeter. `dir_spread` is the half-width of the uniform arc
    # around each supplied direction; 0 → drones spawn exactly on the dirs.
    dir:        list[float] = Field(default_factory=list)
    dir_spread: float       = 0.0
    # Seed for the spawn-RNG (azimuth, jitter, spawn time). `None` → the
    # engine picks a fresh OS-entropy seed each scenario; any int makes
    # the run deterministic given identical other parameters.
    seed:       int | None  = None

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
IMPACT_RADIUS_M       = 5.0   # final-impact override: inside this XY radius
                              # the drone seeks target_3d directly
DESCEND_RANGE_M       = 500.0 # XY distance over which the safety AGL ramps
                              # linearly from full → 0, so the drone glides
                              # smoothly down onto the target instead of
                              # cruising overhead and overshooting.


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
        targets_xy: list[tuple[float, float]],
        ring_radius_m: float = DEFAULT_RING_RADIUS_M,
        spawn_altitude_agl_m: float = SPAWN_ALTITUDE_AGL_M,
        drone_type: int = 0,
        drone_count: int = 1,
        spawn_dirs_deg: list[float] | None = None,
        dir_spread_deg: float = 0.0,
        spawn_window_s: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.heightmap   = heightmap
        self.ring_radius = float(ring_radius_m)
        self.drone_type  = drone_type

        params          = DRONE_TYPES[drone_type]
        self.max_speed  = params.max_speed_mps
        self.max_accel  = params.max_accel_mps2

        # ── Targets ─────────────────────────────────────────────────────
        if not targets_xy:
            targets_xy = [(0.0, 0.0)]
        self._targets_3d = np.array(
            [
                (float(tx), float(ty), float(heightmap.height_at(float(tx), float(ty))))
                for (tx, ty) in targets_xy
            ],
            dtype=np.float32,
        )                                                       # (n_targets, 3)
        n_targets = self._targets_3d.shape[0]

        # Backwards-compat single-target fields (used by /scenario/start
        # response, /simulation/status, and the startup log). They refer to
        # the first target — for multi-target scenarios the truthful list
        # lives on `_targets_3d`.
        self.target_xy = (float(self._targets_3d[0, 0]), float(self._targets_3d[0, 1]))
        self.target_z  = float(self._targets_3d[0, 2])

        # ── Drone slot allocation (round-robin target assignment) ───────
        drone_count = max(1, int(drone_count))
        drone_target_idx = np.array(
            [i % n_targets for i in range(drone_count)], dtype=np.intp,
        )

        # ── Spawn azimuths ──────────────────────────────────────────────
        # Compass azimuth: 0° = north, 90° = east. Three modes:
        #   • no `spawn_dirs_deg`: each drone gets a uniform-random
        #     azimuth in [0, 360);
        #   • dirs supplied, spread = 0: drones snap exactly to the
        #     supplied dirs, round-robin (drone i → dirs[i % k]);
        #   • dirs supplied, spread > 0: round-robin assign each drone
        #     to a dir, then jitter by uniform(-spread, +spread) around
        #     it — so dirs=[15, 89] with spread=5 yields drones uniformly
        #     distributed in the arcs [10, 20] and [84, 94].
        # Seeded RNG drives spawn azimuth (random / dir-jitter modes) and
        # spawn-time draws. With seed=None numpy uses OS entropy; with an
        # int the same scenario parameters reproduce the same scenario.
        rng = np.random.default_rng(seed)
        dirs = list(spawn_dirs_deg) if spawn_dirs_deg else []
        spread = max(0.0, float(dir_spread_deg))

        if not dirs:
            spawn_az_deg = rng.uniform(0.0, 360.0, size=drone_count)
        else:
            base = np.array(
                [dirs[i % len(dirs)] for i in range(drone_count)],
                dtype=np.float64,
            )
            if spread > 0.0:
                spawn_az_deg = base + rng.uniform(-spread, spread, size=drone_count)
            else:
                spawn_az_deg = base

        spawn_positions: list[tuple[float, float, float]] = []
        for az in spawn_az_deg:
            theta = compass_to_math_angle(float(az))
            sx = self.ring_radius * math.cos(theta)
            sy = self.ring_radius * math.sin(theta)
            sg = float(heightmap.height_at(sx, sy))
            sz = sg + spawn_altitude_agl_m
            spawn_positions.append((sx, sy, sz))

        # ── Spawn times: each drone enters at a random t ∈ [0, window] ──
        # Drones with t = 0 spawn immediately; later ones are held back and
        # promoted to ACTIVE inside step() when the sim clock catches up.
        spawn_window = max(0.0, float(spawn_window_s))
        if spawn_window > 0.0:
            spawn_times = rng.uniform(0.0, spawn_window, size=drone_count)
        else:
            spawn_times = np.zeros(drone_count, dtype=np.float64)

        # Sort everything (positions, target idx) by spawn time so we can
        # promote drones in order — the i-th-spawned drone gets the i-th
        # entry of these arrays.
        order = np.argsort(spawn_times, kind="stable")
        self._spawn_times      = spawn_times[order].astype(np.float32)
        self._spawn_positions  = [spawn_positions[int(i)] for i in order]
        self._drone_target_idx = drone_target_idx[order].astype(np.intp)
        self._next_spawn_idx   = 0

        self.state = DroneState(capacity=drone_count)

        self.t              = 0.0
        self.finished       = False
        self.finish_reason: str | None = None

        # Promote any drones whose spawn time is already ≤ 0 (e.g. when
        # spawn_window == 0, every drone enters at t=0).
        self._spawn_due()

        # Representative spawn for the API/log: first-spawned drone's pose.
        self.spawn_xyz = (
            tuple(float(v) for v in self.state.positions[0])
            if self.state.n > 0
            else tuple(float(v) for v in self._spawn_positions[0])
        )

    def _spawn_due(self) -> None:
        """Promote pending drones whose spawn time has arrived to ACTIVE."""
        while (
            self._next_spawn_idx < self.state.capacity
            and self._spawn_times[self._next_spawn_idx] <= self.t
        ):
            sx, sy, sz = self._spawn_positions[self._next_spawn_idx]
            self.state.spawn(
                position=(sx, sy, sz),
                velocity=(0.0, 0.0, 0.0),
                type_=self.drone_type,
                state=ACTIVE,
                intent=INTENT_SEEKING,
            )
            self._next_spawn_idx += 1

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
        # Per-drone target lookup (each drone has its own target via
        # `_drone_target_idx`). Sliced to `n` because pending drones that
        # haven't spawned yet aren't represented in `pos`. Shape (n, 3).
        drone_tgt = self._targets_3d[self._drone_target_idx[:n]]
        # Bearing from each drone to its assigned target (XY).
        to_tgt = drone_tgt[:, :2] - pos[:, :2]
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

        # ── 6. Vertical: terrain-follow that ramps into a dive near target ─
        # The corridor-search `best_idx` can flip between candidate headings
        # tick-to-tick (their costs are close after the soft-margin band).
        # If we used `terrain[rows, best_idx, :]` for the vertical floor,
        # `near_peak` would jump tens of meters whenever the choice flipped
        # — drone chases, overshoots, oscillates.
        #
        # Instead, probe terrain along the drone's *current velocity*
        # direction (yaw-rate limited, so it evolves smoothly even when
        # the lateral corridor choice flips). Vertical and lateral are now
        # decoupled.
        dist_xy = np.linalg.norm(to_tgt, axis=1)

        h_speed_safe = np.maximum(h_speed, 1e-3)
        fwd_x = np.where(moving, vel[:, 0] / h_speed_safe, np.cos(tgt_theta))
        fwd_y = np.where(moving, vel[:, 1] / h_speed_safe, np.sin(tgt_theta))
        fwd_x = fwd_x.astype(np.float32)
        fwd_y = fwd_y.astype(np.float32)

        v_probe_x = pos[:, 0:1] + fwd_x[:, None] * probe_d[None, :]   # (n, P)
        v_probe_y = pos[:, 1:2] + fwd_y[:, None] * probe_d[None, :]
        fwd_terrain = self.heightmap.height_at_batch(
            v_probe_x.reshape(-1), v_probe_y.reshape(-1),
        ).reshape(v_probe_x.shape).astype(np.float32)
        corridor_peak = fwd_terrain.max(axis=1)                       # (n,)

        # Smoothly blend the floor between corridor-peak (far) and the
        # target's own elevation (close), so a hill *past* the target
        # doesn't keep the drone aloft when it's about to impact.
        approach   = np.clip(1.0 - dist_xy / DESCEND_RANGE_M, 0.0, 1.0)
        near_peak  = (1.0 - approach) * corridor_peak + approach * drone_tgt[:, 2]
        safety_agl = TERRAIN_SAFETY_AGL_M * (1.0 - approach)
        desired_z  = np.minimum(near_peak + safety_agl, np.float32(max_alt))

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

        # Impact-radius override: once the drone is within IMPACT_RADIUS_M of
        # its target in XY, ignore terrain clearance and dive straight at
        # it. Without this, the safety AGL keeps the drone cruising ~50 m
        # above the target instead of actually impacting it.
        near_target = dist_xy < IMPACT_RADIUS_M
        if near_target.any():
            delta3 = drone_tgt - pos
            dist3  = np.maximum(np.linalg.norm(delta3, axis=1, keepdims=True), 1e-6)
            seek_v = (delta3 / dist3 * self.max_speed).astype(np.float32)
            mask3  = near_target[:, None]
            desired_v = np.where(mask3, seek_v, desired_v)
            avoiding  = avoiding & ~near_target
        return desired_v, avoiding

    def step(self) -> None:
        if self.finished:
            return

        # Promote any pending drones whose spawn time has arrived.
        self._spawn_due()

        n       = self.state.n
        pos     = self.state.positions[:n]
        vel     = self.state.velocities[:n]
        states  = self.state.states[:n]
        intents = self.state.intents[:n]

        active      = states == ACTIVE if n > 0 else np.zeros(0, dtype=bool)
        all_spawned = self._next_spawn_idx >= self.state.capacity

        # Done condition: every drone has been spawned AND none are active.
        if all_spawned and (n == 0 or not active.any()):
            self.finished      = True
            self.finish_reason = (
                "reached_target"
                if n > 0 and (states == REACHED).any()
                else "no_active_drones"
            )
            return

        # Idle tick: more drones still pending, but nothing to steer right now.
        if n == 0 or not active.any():
            self.t += SIM_DT
            return

        # Reach check first so we don't steer drones that are already there.
        # Each drone is tested against its own assigned target. 3D distance,
        # not XY — otherwise drones get flagged "reached" while still at
        # cruise altitude overhead, before the impact-radius dive can bring
        # them down to target_z.
        drone_tgt   = self._targets_3d[self._drone_target_idx[:n]]
        delta       = drone_tgt - pos
        dist3       = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
        reached_now = active & (dist3 < REACH_THRESHOLD_M)
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

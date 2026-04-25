# Drone Swarm Simulator — Backend Architecture

## Project Summary

Backend for a red-team drone swarm simulator. A topographic map is rendered in 3D in a circle around a user-selected target point. Drones spawn at the ring perimeter and fly toward the target using configurable behaviors. The backend streams drone state to a frontend client that renders the 3D scene.

**Stack:** Python 3.12 + NumPy + FastAPI + uvicorn  
**Why Python:** NumPy vectorization lets us compute steering for hundreds of drones simultaneously using matrix ops. FastAPI gives us WebSocket support and a REST API with minimal boilerplate.

---

## Components

### 1. Terrain Service
- On simulation start, fetch the full heightmap for the playable area from the map server into a NumPy 2D array.
- At 5m resolution over a 10km radius, this is ~16MB in memory. No need for streaming chunks.
- Exposes `height_at(x, y)` via bilinear interpolation (O(1) lookup) and `gradient_at(x, y)` via finite differences.
- Stateless after init. Build and test this first — everything depends on it.
- **Fallback:** If the map server isn't ready, generate synthetic terrain with Perlin noise so development isn't blocked.

### 2. Drone State Store
- **Structure-of-arrays layout**, not array-of-structures. This is critical for performance.
- Separate NumPy arrays indexed by drone ID:
  - `positions: (N, 3)` — x, y, z
  - `velocities: (N, 3)` — vx, vy, vz
  - `types: (N,)` — enum/int for drone type
  - `states: (N,)` — enum/int (active, destroyed, reached_target)
  - `intents: (N,)` — enum/int (seeking, avoiding, diving, orbiting, terrain_following)
- Do NOT make a `Drone` class with attributes. The vectorized layout is what makes the sim fast.
- Double-buffer if needed: write to buffer B while readers consume buffer A, swap atomically.

### 3. Simulation Engine
- Fixed-timestep tick loop at 60Hz internally.
- Each tick: compute steering vectors (vectorized over all drones) → integrate physics → check terminal conditions.
- Pure function of state → state.
- Runs on its own async task (or thread if GIL is a concern, though NumPy releases it during heavy ops).
- Deterministic: accept a seed, use it for all RNG. Same seed = same simulation.

### 4. Streaming Layer
- WebSocket server, decoupled from the sim tick rate.
- Sends binary frames at 20-30Hz (frontend interpolates between frames).
- Does NOT know about steering math. Reads from the state store, packs frames, sends.

### 5. Control API
- REST endpoints:
  - `POST /simulation` — start a new simulation with parameters
  - `DELETE /simulation/{id}` — stop a simulation
  - `GET /simulation/{id}/status` — current status
  - `WS /simulation/{id}/stream` — WebSocket stream of drone positions
- See **Simulation Configuration** section below for the full parameter schema.

---

## Steering Behaviors (Reynolds-style Boids)

Each behavior produces a desired acceleration vector. Sum with weights, clamp to drone's max acceleration.

### Core behaviors
- **Seek:** accelerate toward current waypoint (for kamikaze drones: the target)
- **Separation:** repel from nearby drones (solves inter-drone collision)
- **Cohesion:** steer toward average position of neighbors (makes swarms look like swarms)
- **Alignment:** match velocity with neighbors
- **Terrain avoidance:** repulsive force from ground, stronger as altitude decreases
- **Lookahead obstacle avoidance:** project velocity forward, check terrain height along ray, steer up if collision predicted

### Flight modes as weight presets
| Mode | Seek | Separation | Cohesion | Alignment | Terrain Avoidance | Altitude Target |
|------|------|------------|----------|-----------|-------------------|-----------------|
| Terrain hugging | 1.0 | 1.5 | 0.5 | 0.8 | 3.0 | 15m AGL |
| Top attack | 1.0 | 1.0 | 0.5 | 0.5 | 1.0 | 500m, dive within 200m of target |
| Direct | 1.5 | 1.0 | 0.3 | 0.3 | 1.5 | 80m AGL |
| Circling | 0.0* | 1.0 | 0.5 | 0.8 | 1.5 | 100m AGL |

*Circling uses seek-tangent-to-circle instead of seek-target.

Weights are starting points — tune during development.

### Drone types as kinematic parameter tables
| Type | Max Speed (m/s) | Max Accel (m/s²) | Max Alt (m) | Size | Notes |
|------|----------------|-------------------|-------------|------|-------|
| Small FPV | 40 | 15 | 500 | 0.3m | Fast, agile, short range |
| FPV + fiber optic | 30 | 12 | 400 | 0.4m | Jam-resistant, tethered (soft max-range from spawn) |
| Loitering munition | 20 | 5 | 2000 | 1.2m | Slow, high altitude, long endurance |
| Surveillance | 25 | 8 | 1500 | 0.8m | Circles, doesn't dive |

### Spatial queries
Use a uniform grid rebuilt each tick for neighbor lookups (separation, cohesion, alignment). At hundreds of drones this is plenty fast and simple to implement.

---

## Binary Streaming Protocol

### Position frame (sent at 20-30Hz)
Per-drone record, packed binary (little-endian):

| Field | Type | Bytes | Description |
|-------|------|-------|-------------|
| id | uint16 | 2 | Drone ID |
| type | uint8 | 1 | Drone type enum |
| state | uint8 | 1 | Active/destroyed/reached |
| intent | uint8 | 1 | Current behavior label |
| padding | uint8 | 1 | Alignment |
| x | float32 | 4 | Position X (meters from target) |
| y | float32 | 4 | Position Y |
| z | float32 | 4 | Position Z (altitude) |
| vx | float32 | 4 | Velocity X |
| vy | float32 | 4 | Velocity Y |
| vz | float32 | 4 | Velocity Z |

**Total: 30 bytes per drone.** 500 drones × 30 bytes × 30Hz = 450KB/s. Easily handled by WebSocket.

### Frame envelope
| Field | Type | Bytes |
|-------|------|-------|
| frame_number | uint32 | 4 |
| timestamp | float64 | 8 |
| drone_count | uint16 | 2 |
| padding | uint16 | 2 |
| records | drone[] | 30 × N |

### Event messages (separate message type, sent as needed)
JSON messages for discrete events:
```json
{
  "type": "event",
  "events": [
    {"kind": "drone_destroyed", "id": 42, "reason": "collision", "time": 12.5},
    {"kind": "drone_reached_target", "id": 17, "time": 14.2},
    {"kind": "phase_change", "phase": "terminal_dive", "time": 30.0}
  ]
}
```

Distinguish binary (position frames) from JSON (events) using WebSocket message type (binary vs text).

---

## Coordinate System

- Origin at the target point.
- X = East, Y = North, Z = Up (meters).
- Frontend and backend must agree on this before any integration.

---

## Simulation Configuration

A simulation is configured by a single config object (passed to `POST /simulation`, or loaded from a scenario file). Structure:

```json
{
  "target": {"lat": 59.40, "lon": 17.95},
  "ring_radius": 5000,
  "max_duration_seconds": 180,
  "seed": 42,
  "termination": {
    "stop_when_all_drones_resolved": true,
    "stop_on_max_duration": true
  },
  "waves": [...],
  "failure_model": {...},
  "global_flight_mode_overrides": {}
}
```

### Spawn waves

Spawning is wave-based, not a single flat spawn event. Each wave has its own timing, count, composition, and spawn geometry. This is more expressive than separate "delay" / "direction" / "clustering" parameters and maps naturally to how real layered attacks are structured.

```json
"waves": [
  {
    "name": "recon",
    "start_time": 0,
    "drone_count": 8,
    "composition": {"surveillance": 1.0},
    "flight_mode": "circling",
    "spawn": {
      "directions_deg": [0],
      "arc_width_deg": 360,
      "cluster_spread": 800,
      "altitude_range": [200, 400]
    }
  },
  {
    "name": "first_strike",
    "start_time": 30,
    "spawn_duration": 15,
    "drone_count": 60,
    "composition": {"small_fpv": 0.7, "fpv_fiber": 0.3},
    "flight_mode": "terrain_hugging",
    "spawn": {
      "directions_deg": [45, 90],
      "arc_width_deg": 20,
      "cluster_spread": 150,
      "altitude_range": [20, 50]
    }
  },
  {
    "name": "top_attack_finisher",
    "start_time": 75,
    "drone_count": 6,
    "composition": {"loitering_munition": 1.0},
    "flight_mode": "top_attack",
    "spawn": {
      "directions_deg": [180],
      "arc_width_deg": 10,
      "cluster_spread": 200,
      "altitude_range": [800, 1200]
    }
  }
]
```

### Wave parameters explained

| Parameter | Meaning |
|-----------|---------|
| `start_time` | Seconds from sim start when this wave begins spawning |
| `spawn_duration` | If > 0, drones in this wave spawn linearly over this many seconds. If 0 or omitted, all spawn instantly at `start_time`. |
| `drone_count` | Total drones in this wave |
| `composition` | Map of drone_type → proportion (must sum to 1.0). Counts are randomized per-seed but proportions are honored. |
| `flight_mode` | Behavior preset for drones in this wave (terrain_hugging / top_attack / direct / circling) |
| `spawn.directions_deg` | List of compass bearings (0 = N, 90 = E) where spawn anchors are placed on the ring |
| `spawn.arc_width_deg` | Angular spread around each direction. 0 = pinpoint, 360 = uniform around ring. Lets you express "tight wave from NE" vs "broad sweep from the eastern half." |
| `spawn.cluster_spread` | Spatial spread (meters) around each anchor. Small = tight cluster, large = dispersed. |
| `spawn.altitude_range` | `[min, max]` meters AGL at spawn. Drones spawn uniformly within this range. |

This separates **directional exposure** (`directions_deg` + `arc_width_deg`) from **clustering** (`cluster_spread`) — they're independent axes. You can have one direction with a loose cluster, three directions each tightly clustered, etc.

### Failure model

Real drones fail. Modeling this adds realism and gives you scenario variety.

```json
"failure_model": {
  "rate_per_drone_per_minute": 0.15,
  "mode": "random_mix",
  "mode_weights": {
    "signal_loss": 0.5,
    "mechanical": 0.2,
    "gps_denial": 0.2,
    "intercepted": 0.1
  }
}
```

| Failure mode | Behavior |
|--------------|----------|
| `signal_loss` | Drone holds last velocity vector for N seconds, then drifts to a stop. Visually: drone keeps going, doesn't react. |
| `mechanical` | Drone loses lift, falls under gravity. Visually: drone drops out of swarm. |
| `gps_denial` | Drone's seek behavior disabled; remaining behaviors (separation, terrain) still active. Visually: wanders. |
| `intercepted` | Drone destroyed instantly. Visually: vanishes (frontend can render explosion). |

`mode` accepts: `signal_loss` | `mechanical` | `gps_denial` | `intercepted` | `random_mix` | `none`. With `random_mix`, the per-failure mode is sampled from `mode_weights`.

Failure events are emitted on the event channel so the frontend can render appropriate effects.

### Termination conditions

The simulation stops when EITHER:
- All drones have resolved (reached target, destroyed, or out of bounds), AND `stop_when_all_drones_resolved` is true, OR
- `max_duration_seconds` elapsed, AND `stop_on_max_duration` is true

A `simulation_ended` event is emitted with the reason and summary stats (drones reached target, destroyed, lost).

---

## Pre-parametrized Scenarios

Scenarios are first-class. Stored as JSON files in `config/scenarios/`, loadable by name:

```
POST /simulation
{
  "scenario": "saturation_attack",
  "seed": 42
}
```

The scenario file IS a full simulation config (per the schema above). The seed is provided separately so the same scenario can be re-run with different randomized specifics — same overall structure (waves, directions, counts), different exact drone spawn positions and failure timing.

### Suggested demo scenarios

Build at least these four. Each should look visibly different and tell a story.

| Scenario | Concept | Demonstrates |
|----------|---------|--------------|
| `single_vector_swarm` | One direction, 100 FPV drones, terrain hugging, tight cluster | Pure swarm behavior, terrain following |
| `multi_vector_coordinated` | 3 directions, staggered waves, mixed composition | Realistic coordinated attack, wave timing |
| `top_attack_dive` | Small wave of loitering munitions, high altitude, vertical dive within 200m | Dramatic visual, top-attack flight mode |
| `saturation_with_recon` | Surveillance circling first, then 200-drone saturation strike | Multi-phase attack, lots on screen at once |

Tune each scenario until it looks impressive. Save the seed that looks best — use it for the live demo.

---

## Repository Layout

```
backend/
  app/
    main.py           # FastAPI app, startup
    api/              # REST routes, WebSocket handler
    sim/              # Engine, state store, integrator, tick loop
    behaviors/        # Steering behavior functions (vectorized)
    terrain/          # Heightmap loading, queries, fallback Perlin noise
    protocol/         # Binary frame packing, event schemas
    config/           # Drone type tables, flight mode presets
      scenarios/      # Pre-parametrized scenario JSON files
  tests/              # Sanity checks on math and protocol
  scripts/            # Scenario runners, terrain prep tools
  ARCHITECTURE.md     # This file
  requirements.txt
```

---

## Development Phases

### Phase 1 — End-to-end pipeline (Hours 0-3)
**Goal:** One fake drone streaming positions to the frontend via WebSocket.
- Scaffold FastAPI app with WebSocket endpoint
- Define binary frame format and pack function
- Fake sim loop: one drone moving in a circle, streaming at 20Hz
- Agree on coordinate system and protocol with frontend team
- **Exit criterion:** Frontend renders a moving dot

### Phase 2 — Terrain service (Hours 3-6)
- Load heightmap into NumPy array (or generate Perlin noise fallback)
- Implement `height_at()` and `gradient_at()`
- Fake drone now flies at fixed altitude above terrain
- **Exit criterion:** Drone altitude visibly follows terrain profile

### Phase 3 — Real simulation engine, single drone (Hours 6-12)
- Build state store (structure-of-arrays)
- Implement tick loop with fixed timestep
- Implement seek behavior + physics integration with kinematic limits
- One drone flying from ring edge to target, terrain-aware
- **Exit criterion:** Smooth, physically plausible single-drone flight

### Phase 4 — Multiple drones with steering (Hours 12-20)
- Spawn N drones at ring perimeter based on attack_vectors parameter
- Implement separation (vectorized)
- Implement cohesion + alignment
- Implement terrain avoidance + lookahead
- Build uniform grid for spatial queries
- **Exit criterion:** Swarm of 100+ drones flying to target without collisions

### Phase 5 — Drone types, flight modes, waves, scenarios (Hours 20-28)
- Implement drone type parameter tables
- Implement flight mode weight presets
- Implement wave-based spawn system (start_time, spawn_duration, composition, spawn geometry)
- Implement failure model (signal_loss, mechanical, gps_denial, intercepted)
- Implement scenario loader (JSON files in `config/scenarios/`)
- Add control API endpoints (POST/DELETE/GET)
- Frontend can now configure and launch scenarios by name with a seed
- **Exit criterion:** Visibly different behavior between types, modes, and scenarios; failures emit events

### Phase 6 — Events and polish (Hours 28-36)
- Drone destruction events, reached-target events
- Intent labels in the stream (so frontend can color-code)
- Deterministic seeding
- Edge case handling (drone-drone collision, out-of-bounds)
- **Exit criterion:** Complete, robust simulation

### Phase 7 — Demo prep (Hours 36-44)
- Tune the four pre-built scenarios (see Pre-parametrized Scenarios section)
- For each scenario, try multiple seeds and record the seed that looks best — use that for live demo
- Record backup video of each scenario at its chosen seed
- Load test at target drone count
- **Exit criterion:** Reliable, impressive demo with named scenarios + golden seeds

### Phase 8 — Buffer (Hours 44-48)
Plan to need this time. If you don't, polish or sleep.

---

## Two-Developer Split

**Developer A (Simulation):** Terrain service, state store, engine, steering behaviors. Can develop against print output or matplotlib. Owns `sim/`, `behaviors/`, `terrain/`, `config/`.

**Developer B (Service):** FastAPI app, WebSocket handler, binary protocol, control API, scenario configs. Can develop against a fake state generator. Owns `api/`, `protocol/`, `main.py`.

**Integration contract:** The state store interface. Define it on hour 0, don't change it.

Both devs collaborate on Phase 1 (end-to-end pipeline) to align on the contract early.

---

## Explicitly Deferred

- Multiple concurrent simulations
- Persistence / replay
- Authentication
- Graceful reconnection
- Admin UI
- Realistic aerodynamics (quaternions, thrust vectoring, wind)
- A* or RRT path planning
- Fiber-optic tether physics (just use soft max-range constraint)
- Radio propagation / jamming models (fake it with region-of-effect sphere)

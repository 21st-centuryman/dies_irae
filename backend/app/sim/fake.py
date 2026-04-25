"""Phase-2 fake simulation: a small fleet of drones flying above terrain.

Same record dtype as production so the protocol layer is exercised
end-to-end. Drone Z is now `terrain_height(x, y) + altitude_agl_m`, with
optional sinusoidal AGL wobble — i.e. the drones hug the heightmap
profile as they orbit, fulfilling Phase 2's exit criterion.

Internally uses structure-of-arrays: per-parameter NumPy arrays plus a
shared structured record buffer. Each tick computes positions, velocities,
and terrain samples for all drones in a single vectorized pass — same shape
as the real engine in Phase 3+.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from app.protocol import DRONE_RECORD_DTYPE
from app.terrain import Heightmap


@dataclass(frozen=True)
class OrbitParams:
    radius_m: float
    period_s: float
    altitude_agl_m: float = 100.0    # height above ground level
    z_amplitude_m: float = 0.0       # 0 disables vertical oscillation
    z_period_s: float = 0.0
    phase_rad: float = 0.0


_DEFAULT_FLEET: tuple[OrbitParams, ...] = (
    OrbitParams(radius_m=2000.0, period_s=30.0, altitude_agl_m=80.0),
    OrbitParams(
        radius_m=3000.0,
        period_s=60.0,
        altitude_agl_m=200.0,
        z_amplitude_m=100.0,
        z_period_s=10.0,
        phase_rad=math.pi,           # start on the opposite side of the ring
    ),
)


class FakeOrbitSim:
    """N drones orbiting the target, AGL altitude follows the terrain."""

    def __init__(
        self,
        heightmap: Heightmap,
        drones: tuple[OrbitParams, ...] | None = None,
    ) -> None:
        cfg = drones if drones is not None else _DEFAULT_FLEET
        self.n = len(cfg)
        self.heightmap = heightmap

        self.radius = np.array([d.radius_m for d in cfg], dtype=np.float32)
        self.omega = np.array(
            [2.0 * math.pi / d.period_s for d in cfg], dtype=np.float32
        )
        self.agl = np.array([d.altitude_agl_m for d in cfg], dtype=np.float32)
        self.z_amp = np.array([d.z_amplitude_m for d in cfg], dtype=np.float32)
        self.z_omega = np.array(
            [2.0 * math.pi / d.z_period_s if d.z_period_s > 0 else 0.0 for d in cfg],
            dtype=np.float32,
        )
        self.phase = np.array([d.phase_rad for d in cfg], dtype=np.float32)

        self._records = np.zeros(self.n, dtype=DRONE_RECORD_DTYPE)
        self._records["id"] = np.arange(self.n, dtype=np.uint16)
        self._records["type"] = 0
        self._records["state"] = 0
        self._records["intent"] = 0

        self.t0 = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def sample(self) -> tuple[float, np.ndarray]:
        t = self.elapsed()
        theta = self.omega * t + self.phase
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        z_phase = self.z_omega * t

        x = self.radius * cos_t
        y = self.radius * sin_t
        ground = self.heightmap.height_at_batch(x, y)

        self._records["x"] = x
        self._records["y"] = y
        self._records["z"] = ground + self.agl + self.z_amp * np.sin(z_phase)
        self._records["vx"] = -self.radius * self.omega * sin_t
        self._records["vy"] = self.radius * self.omega * cos_t
        # vz tracks only the AGL wobble; terrain-induced z change is included
        # implicitly each tick (no closed-form derivative without dh/dx,dy here).
        self._records["vz"] = self.z_amp * self.z_omega * np.cos(z_phase)
        return t, self._records

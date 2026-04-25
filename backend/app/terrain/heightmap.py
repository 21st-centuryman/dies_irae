"""Sim-coordinate interface over a RawHeightmap.

Sim coords: X=East, Y=North, in meters from a chosen target lat/lon.
Heightmap rows: row 0 = lat_max (north), so increasing y maps to *decreasing* row.

`height_at` and `gradient_at` are scalar conveniences. `height_at_batch` is
the vectorized form used inside the simulation tick.
"""

from __future__ import annotations

import math

import numpy as np

from .source import RawHeightmap

_M_PER_DEG_LAT = 111_320.0


class Heightmap:
    def __init__(self, raw: RawHeightmap, target_lat: float, target_lon: float) -> None:
        self.array = raw.array
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.lon_min = raw.lon_min
        self.lat_min = raw.lat_min
        self.lon_max = raw.lon_max
        self.lat_max = raw.lat_max

        self.height_px, self.width_px = self.array.shape

        m_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(target_lat))
        bbox_dlon = self.lon_max - self.lon_min
        bbox_dlat = self.lat_max - self.lat_min

        self.m_per_px_x = (bbox_dlon * m_per_deg_lon) / self.width_px
        self.m_per_px_y = (bbox_dlat * _M_PER_DEG_LAT) / self.height_px

        # Pixel coords of the simulation origin (target point). Floats are
        # fine — sample lookups use bilinear interpolation, not integer indexing.
        self.target_col = ((target_lon - self.lon_min) / bbox_dlon) * (self.width_px - 1)
        self.target_row = ((self.lat_max - target_lat) / bbox_dlat) * (self.height_px - 1)

    # --- coordinate transforms ---------------------------------------------

    def _xy_to_px(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        col = self.target_col + np.asarray(x, dtype=np.float32) / self.m_per_px_x
        row = self.target_row - np.asarray(y, dtype=np.float32) / self.m_per_px_y
        return col, row

    # --- sampling ----------------------------------------------------------

    def height_at(self, x: float, y: float) -> float:
        return float(self.height_at_batch(np.array([x], dtype=np.float32),
                                          np.array([y], dtype=np.float32))[0])

    def height_at_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized bilinear sample. Out-of-bounds clamps to edge values."""
        col, row = self._xy_to_px(x, y)
        col = np.clip(col, 0.0, self.width_px - 1.0001)
        row = np.clip(row, 0.0, self.height_px - 1.0001)

        c0 = col.astype(np.intp)
        r0 = row.astype(np.intp)
        fc = col - c0
        fr = row - r0

        a = self.array
        h00 = a[r0, c0]
        h01 = a[r0, c0 + 1]
        h10 = a[r0 + 1, c0]
        h11 = a[r0 + 1, c0 + 1]
        h0 = h00 * (1.0 - fc) + h01 * fc
        h1 = h10 * (1.0 - fc) + h11 * fc
        return h0 * (1.0 - fr) + h1 * fr

    def gradient_at(self, x: float, y: float) -> tuple[float, float]:
        """Slope (∂h/∂x, ∂h/∂y) at (x,y), m per m, by central differences."""
        dx = self.m_per_px_x
        dy = self.m_per_px_y
        gx = (self.height_at(x + dx, y) - self.height_at(x - dx, y)) / (2 * dx)
        gy = (self.height_at(x, y + dy) - self.height_at(x, y - dy)) / (2 * dy)
        return gx, gy

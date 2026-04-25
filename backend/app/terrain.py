"""Heightmap acquisition and sim-coordinate interface.

`fetch_heightmap` / `load_heightmap` obtain a RawHeightmap from the map
server (with synthetic fallback). `Heightmap` wraps one with a
sim-coordinate API: X=East, Y=North in meters from the chosen origin.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from urllib.request import Request, urlopen

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://10.154.6.177:8000"

_M_PER_DEG_LAT = 111_320.0

# Total width/height of the simulation's "world" in sim meters.
# The frontend renders the entire heightmap tile over a plane of size
# WORLD_SIZE * METRES_PER_UNIT = 1000 * 10 = 10_000 m. We therefore treat
# sim coordinates as the frontend's compressed view: regardless of the
# geographic extent the height server actually returns (typically 60 km),
# the heightmap is mapped onto SIM_MAP_EXTENT_M of sim space so that
# backend height samples line up with the on-screen mesh vertices.
SIM_MAP_EXTENT_M = 10_000.0


# ---------------------------------------------------------------------------
# Raw heightmap (geographic coords)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawHeightmap:
    """A 2D height array plus the geographic bounding box that backs it.

    array.shape is (height, width). Row 0 is `lat_max` (north), row H-1 is
    `lat_min` (south). Col 0 is `lon_min` (west). Heights are in meters.
    """

    array: np.ndarray
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape  # type: ignore[return-value]


def fetch_heightmap(
    lat: float,
    lon: float,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
) -> RawHeightmap:
    url = f"{base_url}/fetch?lat={lat}&lon={lon}"
    log.info("fetching heightmap from %s", url)
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        headers = resp.headers
        width = int(headers["X-Width"])
        height = int(headers["X-Height"])
        dtype_str = headers["X-Dtype"]
        bbox = headers["X-BBox"]
        body = resp.read()

    if dtype_str != "float16":
        raise ValueError(f"unsupported X-Dtype {dtype_str!r}")
    expected = width * height * 2
    if len(body) != expected:
        raise ValueError(
            f"payload size {len(body)} != expected {expected} for {width}x{height} float16"
        )

    arr = np.frombuffer(body, dtype="<f2").reshape((height, width)).astype(np.float32)

    parts = [float(s) for s in bbox.split(",")]
    lon_min, lat_min, lon_max, lat_max = parts
    log.info(
        "heightmap %dx%d bbox=(lon %.4f..%.4f, lat %.4f..%.4f) min=%.1f max=%.1f mean=%.1f",
        width, height, lon_min, lon_max, lat_min, lat_max,
        float(arr.min()), float(arr.max()), float(arr.mean()),
    )
    return RawHeightmap(array=arr, lon_min=lon_min, lat_min=lat_min,
                        lon_max=lon_max, lat_max=lat_max)


def synthetic_heightmap(
    lat: float,
    lon: float,
    *,
    side_m: float = 20_000.0,
    cells: int = 4096,
    seed: int = 42,
    base_height: float = 200.0,
) -> RawHeightmap:
    """Sum-of-sines fallback. Same shape/dtype as the real map."""
    rng = np.random.default_rng(seed)
    half = side_m / 2.0
    xs = np.linspace(-half, half, cells, dtype=np.float32)
    ys = np.linspace(half, -half, cells, dtype=np.float32)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    arr = np.full_like(X, base_height)
    for _ in range(8):
        kx = float(rng.uniform(-1.5e-3, 1.5e-3))
        ky = float(rng.uniform(-1.5e-3, 1.5e-3))
        amp = float(rng.uniform(20.0, 80.0))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        arr += amp * np.sin(kx * X + ky * Y + phase)

    m_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(lat))
    dlon = side_m / m_per_deg_lon
    dlat = side_m / _M_PER_DEG_LAT
    return RawHeightmap(
        array=arr,
        lon_min=lon - dlon / 2,
        lat_min=lat - dlat / 2,
        lon_max=lon + dlon / 2,
        lat_max=lat + dlat / 2,
    )


def load_heightmap(
    lat: float,
    lon: float,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> RawHeightmap:
    """Try the map server, fall back to synthetic noise on any failure."""
    try:
        return fetch_heightmap(lat, lon, base_url=base_url)
    except Exception:
        log.exception("map server fetch failed, using synthetic terrain")
        return synthetic_heightmap(lat, lon)


# ---------------------------------------------------------------------------
# Sim-coordinate interface
# ---------------------------------------------------------------------------

class Heightmap:
    """Wraps a RawHeightmap with a sim-coordinate API.

    Sim coords: X=East, Y=North, in meters from a chosen origin lat/lon.
    Heightmap rows: row 0 = lat_max (north), so increasing y maps to
    *decreasing* row index.
    """

    def __init__(
        self,
        raw: RawHeightmap,
        target_lat: float,
        target_lon: float,
        *,
        sim_extent_m: float = SIM_MAP_EXTENT_M,
    ) -> None:
        # Sanitize nodata: the height server returns NaN (and sometimes
        # out-of-realistic-range sentinels like -32768) for cells with no
        # terrain data — water, missing tiles, etc. Bilinear interpolation
        # over even one NaN neighbor yields NaN, which silently propagates
        # into spawn_z / target_z and makes drones invisible in Three.js.
        # Mirror the frontend's behavior (3d.js: isValidElev → fill with the
        # minimum valid value) so height_at() always returns a finite number.
        arr = raw.array
        valid = np.isfinite(arr) & (arr > -500.0) & (arr < 9000.0)
        if not valid.all():
            fill = float(arr[valid].min()) if valid.any() else 0.0
            arr  = np.where(valid, arr, fill).astype(np.float32, copy=False)
            log.info(
                "heightmap: filled %d nodata cells with %.1fm",
                int((~valid).sum()), fill,
            )
        self.array = arr

        self.target_lat = target_lat
        self.target_lon = target_lon
        self.lon_min = raw.lon_min
        self.lat_min = raw.lat_min
        self.lon_max = raw.lon_max
        self.lat_max = raw.lat_max

        self.height_px, self.width_px = self.array.shape

        # Sim-coord pixel scale: the frontend stretches the entire heightmap
        # over `sim_extent_m × sim_extent_m` of display space (currently
        # 10 km × 10 km), regardless of the heightmap's real geographic
        # extent (currently ~60 km from the height server's RADIUS_KM=30).
        # Using the geographic scale here would make `height_at(x, y)`
        # sample real terrain `geographic_extent / sim_extent_m`× further
        # out than the on-screen mesh vertex at the same sim XY, leaving
        # the target marker visually below the rendered ground.
        self.m_per_px_x = sim_extent_m / self.width_px
        self.m_per_px_y = sim_extent_m / self.height_px

        # The target's pixel position is still derived from the bbox so we
        # don't assume the height server centered perfectly on (lat, lon).
        bbox_dlon = self.lon_max - self.lon_min
        bbox_dlat = self.lat_max - self.lat_min
        self.target_col = ((target_lon - self.lon_min) / bbox_dlon) * (self.width_px - 1)
        self.target_row = ((self.lat_max - target_lat) / bbox_dlat) * (self.height_px - 1)

    def _xy_to_px(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        col = self.target_col + np.asarray(x, dtype=np.float32) / self.m_per_px_x
        row = self.target_row - np.asarray(y, dtype=np.float32) / self.m_per_px_y
        return col, row

    def height_at(self, x: float, y: float) -> float:
        return float(self.height_at_batch(np.array([x], dtype=np.float32),
                                          np.array([y], dtype=np.float32))[0])

    def height_at_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized bilinear sample. Out-of-bounds clamps to edge values."""
        col, row = self._xy_to_px(x, y)
        col = np.clip(col, 0.0, self.width_px - 2.0)
        row = np.clip(row, 0.0, self.height_px - 2.0)

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

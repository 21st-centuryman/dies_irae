"""Heightmap acquisition: real fetch from the map server, or synthetic fallback.

The map server returns raw little-endian float16 heights. Shape, dtype, and
bounding box come from response headers (no in-band header in the body).

Synthetic fallback uses sum-of-sines noise so development isn't blocked when
the map server is offline. Generated terrain matches the same dtype/shape
contract.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from urllib.request import Request, urlopen

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://10.154.6.177:8000"


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

    # Promote to float32 immediately — float16 has only ~3 decimal digits of
    # precision, fine for storage but cramped for arithmetic. 32MB → 64MB,
    # still trivial.
    arr = np.frombuffer(body, dtype="<f2").reshape((height, width)).astype(np.float32)

    parts = [float(s) for s in bbox.split(",")]
    lon_min, lat_min, lon_max, lat_max = parts
    log.info(
        "heightmap %dx%d bbox=(lon %.4f..%.4f, lat %.4f..%.4f) min=%.1f max=%.1f mean=%.1f",
        width, height, lon_min, lon_max, lat_min, lat_max,
        float(arr.min()), float(arr.max()), float(arr.mean()),
    )
    return RawHeightmap(
        array=arr,
        lon_min=lon_min,
        lat_min=lat_min,
        lon_max=lon_max,
        lat_max=lat_max,
    )


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

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    dlon = side_m / m_per_deg_lon
    dlat = side_m / m_per_deg_lat
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

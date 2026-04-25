from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed  # ← was missing
import tempfile  # ← was missing
import math
import os
from collections import OrderedDict
import threading

import numpy as np
import requests
import rasterio
import rasterio.warp
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from dotenv import load_dotenv

load_dotenv()

API_KEY = requests.post(
    os.getenv("STAC_TOKEN_URL", ""),
    headers={"Authorization": "Basic " + os.getenv("SECRET", "")},
    data={"grant_type": "client_credentials"},
    verify=False,
).json()["access_token"]
STAC_SEARCH_URL = os.getenv("STAC_SEARCH_URL", "")
RADIUS_KM = float(os.getenv("RADIUS_KM", 30.0))
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", 60))
OUTPUT_SIZE = int(os.getenv("OUTPUT_SIZE", 4096))

DEFAULT_LAT = 56.579
DEFAULT_LON = 14.186
WGS84 = CRS.from_epsg(4326)


def bbox_from_center(lat: float, lon: float):
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    delta_lat = RADIUS_KM / km_per_deg_lat
    delta_lon = RADIUS_KM / km_per_deg_lon
    return (
        lon - delta_lon,  # left  (west)
        lat - delta_lat,  # bottom (south)
        lon + delta_lon,  # right  (east)
        lat + delta_lat,  # top    (north)
    )


def reproject_bbox(bbox_wgs84, dst_crs):
    """Transform a (left, bottom, right, top) WGS84 bbox into dst_crs."""
    left, bottom, right, top = bbox_wgs84
    xs = [left, right, left, right]
    ys = [bottom, bottom, top, top]
    xs_t, ys_t = rasterio.warp.transform(WGS84, dst_crs, xs, ys)
    return (min(xs_t), min(ys_t), max(xs_t), max(ys_t))


def stac_search(lat: float, lon: float):
    bbox = bbox_from_center(lat, lon)
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"bbox": ",".join(map(str, bbox)), "limit": 100}

    features = []
    url = STAC_SEARCH_URL
    page = 0
    while url:
        r = requests.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        batch = data.get("features", [])
        features.extend(batch)
        page += 1
        print(
            f"[stac] page {page}: {len(batch)} features (total so far: {len(features)})"
        )

        # Follow STAC "next" link if present
        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link.get("href")
                break
        url = next_url
        params = {}  # next URL already encodes params

    print(f"[stac] {len(features)} total features across {page} page(s)")
    return {"features": features}, bbox


def extract_asset_urls(feature: dict):
    assets = feature.get("assets", {})
    urls = []
    for _, asset in assets.items():
        href = asset.get("href")
        if not href:
            continue
        asset_type = (asset.get("type") or "").lower()
        roles = [role.lower() for role in asset.get("roles", [])]
        if (
            "geotiff" in asset_type
            or "tiff" in asset_type
            or "image/tiff" in asset_type
            or "data" in roles
            or href.lower().endswith((".tif", ".tiff"))
        ):
            urls.append(href)
    return urls


def _download_one(url):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, headers=headers, stream=True, timeout=180)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".tif")
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return path


def download_files(urls):
    with ThreadPoolExecutor(max_workers=min(len(urls), 8)) as pool:
        futures = {pool.submit(_download_one, url): url for url in urls}
        paths = []
        for future in as_completed(futures):
            paths.append(future.result())
    return paths


_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX = 3  # max cached locations (~256 MB at 4096² float16)


def _cache_key(lat: float, lon: float) -> tuple:
    return (round(lat, 4), round(lon, 4))


def load_height_array(lat: float, lon: float):
    key = _cache_key(lat, lon)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            print(f"[cache] hit for {key}")
            return _cache[key]

    result = _load_height_array_uncached(lat, lon)

    with _cache_lock:
        _cache[key] = result
        _cache.move_to_end(key)
        if len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)

    return result


def _load_height_array_uncached(lat: float, lon: float):
    data, bbox_wgs84 = stac_search(lat, lon)

    asset_urls = []
    for feature in data.get("features", []):
        asset_urls.extend(extract_asset_urls(feature))
    asset_urls = list(dict.fromkeys(asset_urls))
    if not asset_urls:
        raise RuntimeError("No raster assets found for the requested area")

    paths = download_files(asset_urls)
    datasets = []

    try:
        for path in paths:
            datasets.append(rasterio.open(path))

        # Detect the native CRS from the first tile
        native_crs = datasets[0].crs
        print(f"[info] native raster CRS: {native_crs}")

        # Reproject the WGS84 bbox into native CRS so merge() clips correctly
        if native_crs != WGS84:
            merge_bbox = reproject_bbox(bbox_wgs84, native_crs)
        else:
            merge_bbox = bbox_wgs84

        print(f"[info] merge bbox ({native_crs}): {merge_bbox}")

        merged, src_transform = merge(
            datasets,
            bounds=merge_bbox,
            nodata=np.nan,
        )

        raw = merged[0].astype(np.float32, copy=False)
        valid = np.count_nonzero(~np.isnan(raw))
        print(f"[info] merged shape: {raw.shape}, valid cells: {valid}")

        if valid == 0:
            raise RuntimeError(
                "Merged raster contains only NaN — bbox may not overlap any tile"
            )

        # Reproject merged native-CRS raster → WGS84 at OUTPUT_SIZE x OUTPUT_SIZE
        dst_transform = from_bounds(*bbox_wgs84, OUTPUT_SIZE, OUTPUT_SIZE)
        resampled = np.full((OUTPUT_SIZE, OUTPUT_SIZE), np.nan, dtype=np.float32)

        rasterio.warp.reproject(
            source=raw,
            destination=resampled,
            src_transform=src_transform,
            src_crs=native_crs,
            dst_transform=dst_transform,
            dst_crs=WGS84,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        valid_after = np.count_nonzero(~np.isnan(resampled))
        print(f"[info] resampled shape: {resampled.shape}, valid cells: {valid_after}")

        min_h = float(np.nanmin(resampled))
        resampled -= min_h
        print(f"[info] normalised heights: subtracted min={min_h:.2f}m, new range 0–{float(np.nanmax(resampled)):.2f}m")

        height = resampled.astype(np.float16)

        return {
            "array": height,
            "bbox": bbox_wgs84,
            "transform": dst_transform,
        }

    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


def write_height_file(lat: float, lon: float, output_path: str):
    """
    Fetch terrain centered on (lat, lon) with 30km radius,
    resample to 4096x4096 float16, write raw bytes to output_path.
    A sidecar .meta file is written alongside.
    """
    result = load_height_array(lat, lon)
    array = result["array"]  # shape (4096, 4096), float16

    with open(output_path, "wb") as f:
        f.write(array.tobytes(order="C"))

    meta_path = output_path + ".meta"
    bbox = result["bbox"]
    with open(meta_path, "w") as f:
        f.write(f"width={OUTPUT_SIZE}\n")
        f.write(f"height={OUTPUT_SIZE}\n")
        f.write(f"dtype=float16\n")
        f.write(f"order=C\n")
        f.write(f"bbox={','.join(map(str, bbox))}\n")
        f.write(f"lat={lat}\n")
        f.write(f"lon={lon}\n")
        f.write(f"radius_km={RADIUS_KM}\n")

    print(f"Written {output_path}  ({array.nbytes:,} bytes)")
    print(f"Written {meta_path}")
    return result


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/fetch":
            self._handle_fetch(parsed)
        elif parsed.path == "/write":
            self._handle_write(parsed)
        else:
            body = (
                b"Endpoints:\n"
                b"  /fetch?lat=56.579&lon=14.186\n"
                b"  /write?lat=56.579&lon=14.186&out=./terrain.bin\n"
            )
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

    def _handle_fetch(self, parsed):
        try:
            qs = parse_qs(parsed.query)
            lat = float(qs.get("lat", [DEFAULT_LAT])[0])
            lon = float(qs.get("lon", [DEFAULT_LON])[0])

            result = load_height_array(lat, lon)
            payload = result["array"].tobytes(order="C")

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Expose-Headers",
                "X-Width, X-Height, X-Dtype, X-BBox",
            )
            self.send_header("X-Width", str(OUTPUT_SIZE))
            self.send_header("X-Height", str(OUTPUT_SIZE))
            self.send_header("X-Dtype", "float16")
            self.send_header("X-BBox", ",".join(map(str, result["bbox"])))
            self.end_headers()
            self.wfile.write(payload)

        except Exception as e:
            self._error(e)

    def _handle_write(self, parsed):
        try:
            qs = parse_qs(parsed.query)
            lat = float(qs.get("lat", [DEFAULT_LAT])[0])
            lon = float(qs.get("lon", [DEFAULT_LON])[0])
            out = qs.get("out", [f"./terrain_{lat}_{lon}.bin"])[0]

            write_height_file(lat, lon, out)

            body = f"OK: wrote {out} and {out}.meta".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Expose-Headers",
                "X-Width, X-Height, X-Dtype, X-BBox",
            )
            self.end_headers()
            self.wfile.write(body)

        except Exception as e:
            self._error(e)

    def _error(self, e: Exception):
        body = str(e).encode("utf-8", errors="replace")
        try:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        _lat = float(sys.argv[1])
        _lon = float(sys.argv[2])
        _out = sys.argv[3] if len(sys.argv) >= 4 else f"./terrain_{_lat}_{_lon}.bin"
        write_height_file(_lat, _lon, _out)
    else:
        host = "0.0.0.0"
        port = 8000
        server = ThreadingHTTPServer((host, port), Handler)
        print(f"Listening on http://{host}:{port}")
        print("Fetch: http://localhost:8000/fetch?lat=56.579&lon=14.186")
        print(
            "Write: http://localhost:8000/write?lat=56.579&lon=14.186&out=./terrain.bin"
        )
        server.serve_forever()

"""
Inspect a terrain binary from the height server.

Usage:
  # from a saved file:
  python inspect.py terrain.bin

  # fetch live from running server:
  python inspect.py --fetch lat lon
  python inspect.py --fetch 56.579 14.186
"""

import sys
import numpy as np

SIZE = 4096


def load_from_file(path):
    with open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.float16).reshape(SIZE, SIZE).copy()


def load_from_server(lat, lon, host="http://localhost:8000"):
    import requests
    print(f"Fetching {host}/fetch?lat={lat}&lon={lon} ...")
    r = requests.get(f"{host}/fetch", params={"lat": lat, "lon": lon}, timeout=300)
    r.raise_for_status()
    return np.frombuffer(r.content, dtype=np.float16).reshape(SIZE, SIZE).copy()


def report(arr):
    total = arr.size
    valid_mask = ~np.isnan(arr)
    valid = int(np.sum(valid_mask))
    pct = 100 * valid / total

    print(f"\n{'─'*50}")
    print(f"  Shape       : {arr.shape}")
    print(f"  Total cells : {total:,}")
    print(f"  Valid cells : {valid:,}  ({pct:.1f}%)")
    print(f"  NaN cells   : {total - valid:,}  ({100 - pct:.1f}%)")
    if valid:
        print(f"  Min height  : {float(np.nanmin(arr)):.2f} m")
        print(f"  Max height  : {float(np.nanmax(arr)):.2f} m")
        print(f"  Mean height : {float(np.nanmean(arr)):.2f} m")

    print(f"\n  Quadrant coverage (N = raster-north = row 0):")
    h, w = arr.shape
    quadrants = [
        ("NW (top-left)",     arr[: h // 2, : w // 2]),
        ("NE (top-right)",    arr[: h // 2, w // 2 :]),
        ("SW (bottom-left)",  arr[h // 2 :, : w // 2]),
        ("SE (bottom-right)", arr[h // 2 :, w // 2 :]),
    ]
    for name, q in quadrants:
        qv = int(np.sum(~np.isnan(q)))
        qpct = 100 * qv / q.size
        bar = "█" * int(qpct / 5) + "░" * (20 - int(qpct / 5))
        print(f"    {name:20s}  {bar}  {qpct:5.1f}%")

    # ASCII mini-map (32×16 cells)
    print(f"\n  Coverage map (each cell = {SIZE//32}×{SIZE//16} px, █=data ░=NaN):")
    rows, cols = 16, 32
    for ry in range(rows):
        line = "    "
        for rx in range(cols):
            r0, r1 = ry * (h // rows), (ry + 1) * (h // rows)
            c0, c1 = rx * (w // cols), (rx + 1) * (w // cols)
            cell_pct = np.mean(~np.isnan(arr[r0:r1, c0:c1]))
            if cell_pct > 0.66:
                line += "█"
            elif cell_pct > 0.33:
                line += "▒"
            elif cell_pct > 0.0:
                line += "░"
            else:
                line += " "
        print(line)

    print(f"{'─'*50}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    if args[0] == "--fetch":
        if len(args) < 3:
            print("Usage: python inspect.py --fetch <lat> <lon>")
            sys.exit(1)
        arr = load_from_server(float(args[1]), float(args[2]))
    else:
        arr = load_from_file(args[0])

    report(arr)

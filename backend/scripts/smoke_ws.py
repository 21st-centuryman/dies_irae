"""Smoke test: connect to the streaming WebSocket, decode a few frames.

Prints frame header + first drone's position so we can eyeball the fake
drone's circular motion.

Usage: python scripts/smoke_ws.py
"""

from __future__ import annotations

import asyncio
import struct
import sys

import numpy as np
import websockets

from app.protocol import (
    DRONE_RECORD_DTYPE,
    DRONE_RECORD_SIZE,
    HEADER_SIZE,
)


async def main(url: str, frames: int) -> int:
    async with websockets.connect(url) as ws:
        for i in range(frames):
            blob = await ws.recv()
            if not isinstance(blob, (bytes, bytearray)):
                print(f"expected binary frame, got {type(blob)}", file=sys.stderr)
                return 1

            frame_number, timestamp, drone_count, _pad = struct.unpack_from(
                "<IdHH", blob, 0
            )
            expected_len = HEADER_SIZE + drone_count * DRONE_RECORD_SIZE
            if len(blob) != expected_len:
                print(
                    f"bad len: got {len(blob)}, want {expected_len}", file=sys.stderr
                )
                return 1

            records = np.frombuffer(
                blob, dtype=DRONE_RECORD_DTYPE, count=drone_count, offset=HEADER_SIZE
            )
            r0 = records[0]
            print(
                f"frame={frame_number:4d} t={timestamp:6.3f}s n={drone_count} "
                f"id={int(r0['id'])} pos=({r0['x']:8.1f},{r0['y']:8.1f},{r0['z']:6.1f}) "
                f"vel=({r0['vx']:6.1f},{r0['vy']:6.1f},{r0['vz']:5.1f})"
            )
    return 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765/simulation/fake/stream"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sys.exit(asyncio.run(main(url, frames)))

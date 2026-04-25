"""End-to-end Phase-3 verification.

Connects WS, sends PUT /scenario/start, samples a handful of frames at
the start / mid / end of flight, prints the trajectory summary, and
confirms the scenario_ended event arrives.
"""
import asyncio
import json
import struct
import sys
import time
from urllib.request import Request, urlopen

import websockets

HEADER_FMT = "<IdHH"
HEADER_SIZE = 16
RECORD_FMT = "<HBBBxffffff"
RECORD_SIZE = 30


def decode_first_drone(blob: bytes):
    fn, t, n, _ = struct.unpack_from(HEADER_FMT, blob, 0)
    rid, rtype, rstate, rintent, x, y, z, vx, vy, vz = struct.unpack_from(
        RECORD_FMT, blob, HEADER_SIZE
    )
    return {
        "frame": fn, "t": t, "n": n,
        "id": rid, "state": rstate, "intent": rintent,
        "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz,
    }


async def main():
    ws_url = "ws://127.0.0.1:3000/simulation/fake/stream"
    put_url = "http://127.0.0.1:3000/scenario/start"
    payload = json.dumps({
        "drone_count": 1, "pct_attack": 50, "pct_recon": 50,
        "pct_short": 50, "pct_long": 50, "scenario_type": "allatonce",
        "spawn_window": 0, "wave_interval": 5,
        "lat": 56.579, "lon": 14.186, "target": [0, 0],
    }).encode()

    async with websockets.connect(ws_url, max_size=None) as ws:
        # Arm the scenario.
        req = Request(put_url, data=payload, method="PUT",
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as resp:
            arm = json.loads(resp.read())
        print("[ARM]")
        print(f"  spawn_xyz   = {arm['spawn_xyz']}")
        print(f"  target_xy   = {arm['target_xy']}")
        print(f"  target_z    = {arm['target_z']:.1f}")
        print(f"  max_speed   = {arm['max_speed_mps']} m/s")
        print(f"  max_accel   = {arm['max_accel_mps2']} m/s^2")

        speeds = []
        first = None
        last = None
        ended_event = None
        wall_t0 = time.monotonic()

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
            except asyncio.TimeoutError:
                print("TIMEOUT — engine likely stuck")
                return 1

            if isinstance(msg, str):
                ended_event = json.loads(msg)
                break

            d = decode_first_drone(msg)
            if first is None:
                first = d
                print(f"\n[FRAME 0] t={d['t']:.2f}s pos=({d['x']:.1f},{d['y']:.1f},{d['z']:.1f}) "
                      f"vel=({d['vx']:.1f},{d['vy']:.1f},{d['vz']:.1f}) state={d['state']}")
            last = d
            speed = (d['vx']**2 + d['vy']**2 + d['vz']**2)**0.5
            speeds.append(speed)
            if d['frame'] in (10, 50, 100, 200, 400, 800, 1600):
                print(f"[FRAME {d['frame']}] t={d['t']:.2f}s pos=({d['x']:.1f},{d['y']:.1f},{d['z']:.1f}) "
                      f"speed={speed:.2f} m/s state={d['state']}")

        wall_elapsed = time.monotonic() - wall_t0
        print(f"\n[END] t_sim={last['t']:.2f}s wall={wall_elapsed:.2f}s frames={last['frame']+1}")
        print(f"  final pos = ({last['x']:.1f},{last['y']:.1f},{last['z']:.1f}) state={last['state']}")
        print(f"  peak speed = {max(speeds):.2f} m/s (max allowed = {arm['max_speed_mps']})")
        print(f"  event: {ended_event}")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

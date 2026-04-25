"""Verify WS waits for PUT /scenario/start before streaming."""
import asyncio
import json
import sys
import time
from urllib.request import Request, urlopen

import websockets


async def main():
    ws_url = "ws://0.0.0.0:3000/simulation/fake/stream"
    put_url = "http://0.0.0.0:3000/scenario/start"
    payload = json.dumps({
        "drone_count": 10, "pct_attack": 50, "pct_recon": 50,
        "pct_short": 50, "pct_long": 50, "scenario_type": "allatonce",
        "spawn_window": 0, "wave_interval": 5,
        "lat": 56.255, "lon": 12.562, "target": [0, 0],
    }).encode()

    async with websockets.connect(ws_url) as ws:
        print("WS open. Trying to receive a frame within 1.5s (should TIMEOUT)…")
        try:
            blob = await asyncio.wait_for(ws.recv(), timeout=1.5)
            print(f"  UNEXPECTED frame: {len(blob)} bytes — gate is NOT working")
            return 1
        except asyncio.TimeoutError:
            print("  [OK] no frames — WS is correctly waiting")

        print("\nSending PUT /scenario/start…")
        req = Request(put_url, data=payload, method="PUT",
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            print(f"  PUT response: status={body.get('status')}")

        print("\nReceiving 5 frames after arm…")
        for i in range(5):
            blob = await asyncio.wait_for(ws.recv(), timeout=2)
            print(f"  frame {i}: {len(blob)} bytes")
        print("\n[OK] gating works")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

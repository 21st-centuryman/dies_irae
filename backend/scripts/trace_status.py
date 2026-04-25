"""Poll /simulation/status while engine runs — no WS needed."""
import json
import sys
import time
from urllib.request import Request, urlopen

put_url = "http://127.0.0.1:3000/scenario/start"
status_url = "http://127.0.0.1:3000/simulation/status"
payload = json.dumps({
    "drone_count": 1, "pct_attack": 50, "pct_recon": 50,
    "pct_short": 50, "pct_long": 50, "scenario_type": "allatonce",
    "spawn_window": 0, "wave_interval": 5,
    "lat": 56.579, "lon": 14.186, "target": [0, 0],
}).encode()

req = Request(put_url, data=payload, method="PUT",
              headers={"Content-Type": "application/json"})
with urlopen(req, timeout=5) as resp:
    arm = json.loads(resp.read())
print(f"spawn={arm['spawn_xyz']} target_z={arm['target_z']}")

t0 = time.monotonic()
last_pos = None
ticks_no_progress = 0
while True:
    with urlopen(status_url, timeout=5) as resp:
        s = json.loads(resp.read())
    if not s.get("active"):
        print(f"[t_wall={time.monotonic()-t0:.1f}] not active anymore: {s}")
        break
    pos = s["drone_pos"]
    print(f"[sim_t={s['sim_t']:6.1f}s] pos=({pos[0]:7.1f},{pos[1]:7.1f},{pos[2]:6.1f}) "
          f"finished={s['finished']} reason={s['finish_reason']}")
    if s["finished"]:
        break
    if last_pos is not None and abs(pos[0]-last_pos[0])<0.5 and abs(pos[1]-last_pos[1])<0.5:
        ticks_no_progress += 1
        if ticks_no_progress > 5:
            print("WARN: position stuck, drone may be oscillating")
    else:
        ticks_no_progress = 0
    last_pos = pos
    if time.monotonic() - t0 > 180:
        print("WALL TIMEOUT 180s")
        sys.exit(1)
    time.sleep(2)

"""Drone state store — structure-of-arrays layout per ARCHITECTURE.md §2.

A single contiguous block of NumPy arrays indexed by drone slot. The
arrays are sized at construction (`capacity`) and `n` tracks how many
slots are currently populated. `to_records()` produces a view in
DRONE_RECORD_DTYPE for the protocol layer — a packing operation, not a
copy of the arithmetic state.

Phase-3 only spawns N=1, but the same layout scales straight to N drones
for Phase 4 — no rewrite, just bigger arrays.
"""

from __future__ import annotations

import numpy as np

from app.protocol import DRONE_RECORD_DTYPE

# State enum — matches the `state` field on the wire.
ACTIVE = 0
DESTROYED = 1
REACHED = 2

# Intent enum — matches the `intent` field on the wire (behavior label).
INTENT_SEEKING = 0
INTENT_AVOIDING = 1
INTENT_DIVING = 2
INTENT_ORBITING = 3
INTENT_TERRAIN_FOLLOWING = 4


class DroneState:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.n = 0
        self.positions = np.zeros((capacity, 3), dtype=np.float32)
        self.velocities = np.zeros((capacity, 3), dtype=np.float32)
        self.types = np.zeros(capacity, dtype=np.uint8)
        self.states = np.zeros(capacity, dtype=np.uint8)
        self.intents = np.zeros(capacity, dtype=np.uint8)
        self.ids = np.arange(capacity, dtype=np.uint16)
        self._records = np.zeros(capacity, dtype=DRONE_RECORD_DTYPE)

    def spawn(
        self,
        position,
        velocity=(0.0, 0.0, 0.0),
        type_: int = 0,
        state: int = ACTIVE,
        intent: int = INTENT_SEEKING,
    ) -> int:
        if self.n >= self.capacity:
            raise RuntimeError(f"DroneState full (capacity={self.capacity})")
        i = self.n
        self.positions[i] = position
        self.velocities[i] = velocity
        self.types[i] = type_
        self.states[i] = state
        self.intents[i] = intent
        self.n += 1
        return i

    def to_records(self) -> np.ndarray:
        n = self.n
        rec = self._records[:n]
        rec["id"] = self.ids[:n]
        rec["type"] = self.types[:n]
        rec["state"] = self.states[:n]
        rec["intent"] = self.intents[:n]
        rec["x"] = self.positions[:n, 0]
        rec["y"] = self.positions[:n, 1]
        rec["z"] = self.positions[:n, 2]
        rec["vx"] = self.velocities[:n, 0]
        rec["vy"] = self.velocities[:n, 1]
        rec["vz"] = self.velocities[:n, 2]
        return rec

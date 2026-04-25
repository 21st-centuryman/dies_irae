"""Active-scenario session state shared between the REST trigger and the WS stream.

`ScenarioConfig` mirrors the frontend's current PUT body verbatim. Phase-3
uses only `target` (the destination point); the rest is accepted but
ignored. `extra="allow"` means the frontend can grow the payload without
breaking us.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import asyncio

    from app.sim.engine import SimEngine


class ScenarioConfig(BaseModel):
    drone_count: int = Field(ge=0)
    pct_attack: int = 50
    pct_recon: int = 50
    pct_short: int = 50
    pct_long: int = 50
    scenario_type: str = "allatonce"
    spawn_window: int = 0
    wave_interval: int = 5
    lat: float
    lon: float
    targets: list = Field(default_factory=lambda: [[0.0, 0.0]])

    model_config = {"extra": "allow"}


@dataclass
class ScenarioSession:
    config: ScenarioConfig
    engine: "SimEngine"
    started_at: float = field(default_factory=time.monotonic)
    engine_task: Optional["asyncio.Task"] = None

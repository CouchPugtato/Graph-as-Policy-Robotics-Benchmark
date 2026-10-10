"""Small, durable progress markers for bounded diagnostics; no simulator imports."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class DiagnosticProgress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.monotonic()

    def mark(self, stage: str, **details: object) -> None:
        payload = {
            "stage": stage,
            "elapsed_wall_s": round(time.monotonic() - self.started, 3),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "commands_applied": 0,
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        print(f"[feasibility {payload['elapsed_wall_s']:.3f}s] {stage} {details}", flush=True)

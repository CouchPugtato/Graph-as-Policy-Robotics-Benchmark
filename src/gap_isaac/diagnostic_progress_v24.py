"""Bounded Windows sharing retries for diagnostic metadata; no control actions."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from gap_isaac.diagnostic_progress import DiagnosticProgress as BaselineProgress


class DiagnosticProgress(BaselineProgress):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.write_retry_count = 0

    def mark(self, stage: str, **details: object) -> None:
        payload = {"stage": stage, "elapsed_wall_s": round(time.monotonic() - self.started, 3),
                   "updated_at": datetime.now(timezone.utc).isoformat(), "details": details,
                   "commands_applied": 0, "progress_write_retry_count": self.write_retry_count}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Windows readers/indexers can temporarily deny replace sharing. Ten
        # attempts consume at most 225 ms; persistent failures still propagate.
        for attempt in range(10):
            try:
                temporary.replace(self.path)
                break
            except PermissionError as error:
                if getattr(error, "winerror", None) not in (5, 32) or attempt == 9:
                    raise
                self.write_retry_count += 1
                print(f"[feasibility] progress write sharing retry {attempt + 1}: {stage}",
                      flush=True)
                time.sleep(.025)
        print(f"[feasibility {payload['elapsed_wall_s']:.3f}s] {stage} {details}", flush=True)

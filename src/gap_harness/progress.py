"""Lightweight terminal progress without extra requests or response streaming."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from contextlib import contextmanager


class Progress:
    def __init__(self, stream=None, heartbeat_s=15, clock=time.monotonic):
        if not math.isfinite(heartbeat_s) or not 0 < heartbeat_s <= 60:
            raise ValueError("heartbeat interval must be positive and at most sixty seconds")
        self.stream = stream if stream is not None else sys.stderr
        self.heartbeat_s = heartbeat_s
        self.clock = clock
        self.started = clock()
        self.lock = threading.Lock()

    def event(self, message):
        elapsed = self.clock() - self.started
        with self.lock:
            print(f"[{elapsed:7.1f}s] {message}", file=self.stream, flush=True)

    @contextmanager
    def waiting(self, label, details=""):
        started = self.clock()
        finished = threading.Event()
        self.event(f"{label}: started" + (f" | {details}" if details else ""))

        def heartbeat():
            while not finished.wait(self.heartbeat_s):
                self.event(f"{label}: still waiting ({self.clock()-started:.0f}s elapsed)")

        # this thread only prints status while the original operation stays synchronous
        thread = threading.Thread(target=heartbeat, name="harness-progress", daemon=True)
        thread.start()
        try:
            yield started
        finally:
            finished.set()
            thread.join(timeout=1)


def session_usage(session):
    requests = list(session.glob("v*/calls/*/request.json"))
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    missing = 0
    for request in requests:
        response_path = request.parent / "response.json"
        if not response_path.is_file():
            missing += 1
            continue
        response = json.loads(response_path.read_text(encoding="utf-8"))
        usage = dict(response.get("usage") or {})
        if (usage.get("total_tokens") is None
                and all(type(usage.get(key)) is int and usage[key] >= 0
                        for key in ("input_tokens", "output_tokens"))):
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        if not all(type(usage.get(key)) is int and usage[key] >= 0 for key in totals):
            missing += 1
        for key in totals:
            if type(usage.get(key)) is int and usage[key] >= 0:
                totals[key] += usage[key]
    return {**totals, "attempts": len(requests), "incomplete_usage_calls": missing}


def usage_summary(session, max_calls):
    usage = session_usage(session)
    message = (f"calls {usage['attempts']}/{max_calls} | reported session tokens "
               f"{usage['total_tokens']:,} (in {usage['input_tokens']:,}, "
               f"out {usage['output_tokens']:,})")
    if usage["incomplete_usage_calls"]:
        message += f" | {usage['incomplete_usage_calls']} call(s) with incomplete usage"
    return message

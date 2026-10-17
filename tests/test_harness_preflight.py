import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gap_harness.models import Task
from gap_harness.preflight import check_runner

ROOT = Path(__file__).resolve().parents[1]


def harness():
    path = ROOT / "harness/tasks/libero-packing-local-qwen.json"
    return SimpleNamespace(task=Task.model_validate_json(path.read_text()), task_path=path,
                           progress=SimpleNamespace(event=lambda message: None))


@pytest.mark.parametrize("ready", [True, False])
def test_local_service_check_is_bounded_and_never_starts_simulation(monkeypatch, ready):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0 if ready else 1,
            stdout=json.dumps({"ok": ready, "detail": "local VLM unreachable"}))

    monkeypatch.setattr("gap_harness.preflight.subprocess.run", run)
    if ready:
        check_runner(harness())
    else:
        with pytest.raises(RuntimeError, match="before generation.*unreachable"):
            check_runner(harness())
    assert calls[0][0][-3:] == ["-m", "gap_libero.cli", "preflight"]
    assert calls[0][1]["timeout"] == 45
    assert "--enable-motion" not in calls[0][0]


def test_no_runtime_probe_for_tasks_without_a_runner(monkeypatch):
    candidate = harness()
    candidate.task.runner = None
    monkeypatch.setattr("gap_harness.preflight.subprocess.run", lambda *a, **kw: pytest.fail())
    check_runner(candidate)

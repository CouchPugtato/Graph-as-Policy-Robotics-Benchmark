import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def runner_module():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "cylinder_trial_runner", root / "scripts/run_cylinder_center_trials.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trial(module, folder):
    instance = object.__new__(module.CylinderTrial)
    instance.baseline = False
    instance.seed = 3
    instance.directory = folder
    instance.state = {"artifacts": {}, "steps": [], "infrastructure_retries": 0}
    return instance


def test_passive_watchdog_retry_retains_both_attempts(tmp_path, monkeypatch):
    module = runner_module()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"ok": False, "error_type": "DiagnosticTimeout",
                                 "commands_applied": 0}))
    second.write_text(json.dumps({"ok": True}))
    calls = []

    class Process:
        def __init__(self, args, **kwargs):
            calls.append(args)
            self.pid = len(calls)
            self.returncode = 1 if self.pid == 1 else 0
            result = first if self.pid == 1 else second
            kwargs["stdout"].write(f"Saved diagnostic: {result}\n")

        def wait(self, timeout):
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    instance = trial(module, tmp_path)
    instance.run("feasibility", "test-perceived-feasibility.ps1", [], 110)
    assert instance.state["infrastructure_retries"] == 1
    assert [s["name"] for s in instance.state["steps"]] == ["feasibility", "feasibility-retry"]
    assert [s["result"] for s in instance.state["steps"]] == [str(first), str(second)]
    assert first.is_file() and second.is_file()


def test_motion_watchdog_stops_owned_process_without_retry(tmp_path, monkeypatch):
    module = runner_module()
    stopped = []

    class Process:
        pid = 42
        returncode = None

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("owned simulator", timeout)

        def poll(self):
            return self.returncode

    def stop(process):
        stopped.append(process.pid)
        process.returncode = 1

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module, "stop_owned_tree", stop)
    instance = trial(module, tmp_path)
    try:
        instance.run("grasp", "test-graph-pick-place.ps1", [], 120, allow_failed_result=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Motion watchdog must fail the trial")
    assert stopped == [42]
    assert len(instance.state["steps"]) == 1
    assert instance.state["steps"][0]["timed_out"]
    assert instance.state["infrastructure_retries"] == 0

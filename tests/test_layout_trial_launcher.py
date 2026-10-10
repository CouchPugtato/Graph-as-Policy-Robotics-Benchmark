import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def load_launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/run_layout_trials.py"
    spec = importlib.util.spec_from_file_location("layout_trials", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_powershell_named_parameters_and_literal_values(tmp_path):
    module = load_launcher()
    module.ROOT = tmp_path
    directory = tmp_path / "scripts/windows"
    directory.mkdir(parents=True)
    (directory / "probe.ps1").write_text(
        "param([int]$Seed, [string]$Stage, [switch]$Headless)\n"
        "@{seed=$Seed;stage=$Stage;headless=[bool]$Headless}|ConvertTo-Json\n",
        encoding="utf-8",
    )
    value = "C:/test's folder/$literal.usd"
    result = subprocess.run(
        module.command("probe.ps1", ["-Seed", 3, "-Stage", value, "-Headless"]),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    import json

    assert json.loads(result.stdout) == {"seed": 3, "stage": value, "headless": True}
    invalid = subprocess.run(
        module.command("probe.ps1", ["-Seed", "invalid"]),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert invalid.returncode != 0


def test_stage_receipt_parser_ignores_startup_dictionaries():
    module = load_launcher()
    assert (
        module.payload(
            'startup {invalid}\n{"nested": {"ok": false}}\n'
            '{"ok": true, "result": "artifact.json"}\n',
            "result",
        )
        == "artifact.json"
    )


def test_trial_uses_explicit_graph_and_records_its_hash(tmp_path, monkeypatch):
    module = load_launcher()
    graph = tmp_path / "new-generation" / "accepted.json"
    observed = []

    def verify(path):
        observed.append(path)
        return {"graph_sha256": "new-graph-hash"}

    monkeypatch.setattr(module, "verify_accepted", verify)
    trial = module.Trial(0, tmp_path, graph)
    assert observed == [graph.resolve()]
    assert trial.graph == graph.resolve()
    assert trial.state["graph"] == str(graph.resolve())
    assert trial.state["graph_sha256"] == "new-graph-hash"


def test_saved_input_retry_rejects_different_graph(tmp_path, monkeypatch):
    module = load_launcher()
    original = tmp_path / "trial.json"
    original.write_text(
        json.dumps({"status": "failed", "seed": 0, "graph_sha256": "original-hash"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "verify_accepted", lambda path: {"graph_sha256": "other-hash"})
    monkeypatch.setattr(module.sys, "argv", ["runner", "--resume-trial", str(original)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert not (tmp_path / "results").exists()


def test_saved_input_retry_does_not_repeat_perception(tmp_path, monkeypatch):
    module = load_launcher()
    module.ROOT = tmp_path
    original = tmp_path / "trial.json"
    artifacts = {
        "stage": "saved.usd",
        "alphabet_soup_geometry": "target.json",
        "basket_geometry": "basket.json",
        "grasp": "failed-grasp.json",
    }
    original.write_text(
        json.dumps(
            {"status": "failed", "seed": 0, "graph_sha256": "same-hash", "artifacts": artifacts}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "verify_accepted", lambda path: {"graph_sha256": "same-hash"})
    monkeypatch.setattr(module.sys, "argv", ["runner", "--resume-trial", str(original)])
    calls = []

    def motion(trial, stage, geometries):
        assert "grasp" not in trial.state["artifacts"]
        calls.append((stage, geometries))
        trial.state["status"] = "succeeded"
        trial.save()

    monkeypatch.setattr(module.Trial, "execute_motion", motion)
    monkeypatch.setattr(module.Trial, "execute", lambda trial: pytest.fail("Perception repeated"))
    assert module.main() == 0
    assert calls == [(Path("saved.usd"), {"alphabet_soup": "target.json", "basket": "basket.json"})]
    assert json.loads(original.read_text())["status"] == "failed"

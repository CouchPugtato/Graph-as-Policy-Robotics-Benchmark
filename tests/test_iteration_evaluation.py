import importlib.util
import json
from pathlib import Path

import pytest


def load_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("iteration", scripts / "evaluate_iteration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(seed=0):
    return {
        "ok": True,
        "run_revision": "test-v1",
        "graph_sha256": "hash",
        "evaluation": {
            "seed": seed,
            "status": "succeeded",
            "disallowed_collisions": [],
            "checkpoints": dict.fromkeys(
                ["target_grasped", "target_lifted", "target_released", "target_contained"], True
            ),
        },
        "graph": {
            "status": "succeeded",
            "terminal": "SUCCEEDED",
            "trace": [{"outcome": "succeeded"}] * 10,
        },
    }


@pytest.mark.parametrize("problem", ["revision", "hash", "empty-checkpoints", "collision", "trace"])
def test_receipt_gate_rejects_false_success(monkeypatch, problem):
    module = load_module(monkeypatch)
    data = result()
    if problem == "revision":
        data["run_revision"] = "wrong"
    elif problem == "hash":
        data["graph_sha256"] = "wrong"
    elif problem == "empty-checkpoints":
        data["evaluation"]["checkpoints"] = {}
    elif problem == "collision":
        data["evaluation"]["disallowed_collisions"] = ["collision"]
    else:
        data["graph"]["trace"] = []
    with pytest.raises(ValueError):
        module.validate_result(data, "test-v1", "hash", 0)


def setup_main(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    module.ROOT = tmp_path
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate.json").write_text("{}", encoding="utf-8")
    (bundle / "accepted.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "verify_package", lambda path: {})
    monkeypatch.setattr(module, "validate_generated_graph", lambda data: None)
    monkeypatch.setattr(module, "verify_accepted", lambda path: {"graph_sha256": "hash"})
    monkeypatch.setattr(module, "revision", lambda root: "test-v1")
    monkeypatch.setattr(module, "source_hashes", lambda root: {"source.py": "frozen"})
    monkeypatch.setattr(module, "run_tests", lambda directory: None)
    monkeypatch.setattr(module, "check_resources", lambda directory, seed: None)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["evaluation", "--bundle", str(bundle), "--expected-revision", "test-v1"],
    )
    return module


def test_validate_only_never_runs_trials(monkeypatch, tmp_path):
    module = setup_main(monkeypatch, tmp_path)
    module.sys.argv.append("--validate-only")
    monkeypatch.setattr(module, "Trial", lambda *args: pytest.fail("Simulation started"))
    assert module.main() == 0
    summary = json.loads(next((tmp_path / "results").rglob("summary.json")).read_text())
    assert summary["status"] == "validated-not-run"
    assert summary["seeds"] == []


def test_all_ten_continue_after_seed_failure(monkeypatch, tmp_path):
    module = setup_main(monkeypatch, tmp_path)
    calls = []

    class FakeTrial:
        def __init__(self, seed, directory, graph):
            self.seed = seed
            self.directory = directory / f"seed-{seed}"
            self.directory.mkdir()
            self.state = {"status": "running", "artifacts": {}}

        def execute(self):
            calls.append(self.seed)
            if self.seed == 3:
                raise ValueError("grasp failed")
            artifact = self.directory / "graph.json"
            artifact.write_text(json.dumps(result(self.seed)), encoding="utf-8")
            self.state.update(status="succeeded", artifacts={"graph": str(artifact)})
            self.save()

        def save(self):
            (self.directory / "trial.json").write_text(json.dumps(self.state), encoding="utf-8")

    monkeypatch.setattr(module, "Trial", FakeTrial)
    assert module.main() == 1
    assert calls == list(range(10))
    summary = json.loads(next((tmp_path / "results").rglob("summary.json")).read_text())
    assert summary["status"] == "complete"
    assert len(summary["seeds"]) == 10
    assert sum(row["status"] == "succeeded" for row in summary["seeds"]) == 9


def test_revision_mismatch_stops_before_tests_or_trials(monkeypatch, tmp_path):
    module = setup_main(monkeypatch, tmp_path)
    module.sys.argv[-1] = "incorrect"
    monkeypatch.setattr(
        module, "run_tests", lambda directory: pytest.fail("Tests should not start")
    )
    assert module.main() == 2


def test_source_changes_stop_before_any_trial(monkeypatch, tmp_path):
    module = setup_main(monkeypatch, tmp_path)
    hashes = iter(
        [
            {"source.py": "tested"},
            {"source.py": "tested"},
            {"source.py": "frozen"},
            {"source.py": "changed"},
        ]
    )
    monkeypatch.setattr(module, "source_hashes", lambda root: next(hashes))
    monkeypatch.setattr(module, "Trial", lambda *args: pytest.fail("Changed source ran"))
    assert module.main() == 2
    summary = json.loads(next((tmp_path / "results").rglob("summary.json")).read_text())
    assert summary["status"] == "blocked"
    assert summary["seeds"] == []


def test_failed_local_tests_block_simulation(monkeypatch, tmp_path):
    module = setup_main(monkeypatch, tmp_path)

    def fail(directory):
        raise ValueError("Local tests failed")

    monkeypatch.setattr(module, "run_tests", fail)
    monkeypatch.setattr(module, "Trial", lambda *args: pytest.fail("Untested source ran"))
    assert module.main() == 2

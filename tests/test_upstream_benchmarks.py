import sys
from types import SimpleNamespace

import pytest
import yaml

from gap_libero.benchmarks.catalog import BENCHMARKS, profile
from gap_libero.benchmarks.runtime import activate, configure


def make_source(tmp_path, name):
    selected = profile(name, tmp_path)
    upstream = selected["source"] / "libero/libero"
    for folder in ("assets", "bddl_files", "init_files"):
        (upstream / folder).mkdir(parents=True)
    (upstream / "bddl_files" / selected["suite"]).mkdir()
    return selected, upstream


def test_catalog_separates_custom_and_upstream(tmp_path):
    assert "packing" not in BENCHMARKS
    for name in BENCHMARKS:
        selected = profile(name, tmp_path)
        assert selected["output_root"].is_relative_to(tmp_path / "results/benchmarks")
        assert selected["config"].is_relative_to(tmp_path / "cache/benchmarks")
    assert profile("libero-pro-object")["suite"] == "libero_object_object"
    assert profile("libero-pro-swap")["suite"] == "libero_object_swap"


def test_prepare_does_not_change_custom_config(tmp_path, monkeypatch):
    selected, upstream = make_source(tmp_path, "libero-object")
    monkeypatch.setattr("gap_libero.benchmarks.runtime.subprocess.check_output",
                        lambda *a, **kw: selected["revision"])
    monkeypatch.setenv("LIBERO_CONFIG_PATH", "custom-config")
    configure("libero-object", tmp_path)
    content = yaml.safe_load((selected["config"] / "config.yaml").read_text())
    assert content["assets"] == str((upstream / "assets").resolve())
    import os
    assert os.environ["LIBERO_CONFIG_PATH"] == "custom-config"
    configure("libero-object", tmp_path)


def test_prepare_refuses_unexpected_revision(tmp_path, monkeypatch):
    make_source(tmp_path, "libero-object")
    monkeypatch.setattr("gap_libero.benchmarks.runtime.subprocess.check_output",
                        lambda *a, **kw: "other")
    with pytest.raises(ValueError, match="pinned revision"):
        configure("libero-object", tmp_path)


def test_pro_uses_separate_evaluation_data(tmp_path, monkeypatch):
    selected, upstream = make_source(tmp_path, "libero-pro-object")
    for folder in ("bddl_files", "init_files"):
        (selected["data"] / folder / selected["suite"]).mkdir(parents=True)
    monkeypatch.setattr("gap_libero.benchmarks.runtime.subprocess.check_output",
                        lambda *a, **kw: selected["revision"])
    configure("libero-pro-object", tmp_path)
    content = yaml.safe_load((selected["config"] / "config.yaml").read_text())
    assert content["bddl_files"] == str((selected["data"] / "bddl_files").resolve())
    assert content["assets"] == str((upstream / "assets").resolve())


def test_activate_rejects_mixed_imports(monkeypatch):
    monkeypatch.setitem(sys.modules, "libero.libero", object())
    with pytest.raises(RuntimeError, match="fresh worker"):
        activate("libero-object")


def test_custom_selection_cannot_enter_native_catalog():
    with pytest.raises(ValueError, match="unknown upstream"):
        profile("fulfillment")


def test_worker_rejects_invalid_task_before_simulator(monkeypatch):
    from gap_libero.benchmarks import cli

    monkeypatch.setattr(cli, "tasks", lambda selected: [])
    with pytest.raises(ValueError, match="outside"):
        cli.worker(SimpleNamespace(mode="capture", task_id=0), {})


def test_worker_refuses_missing_official_files(monkeypatch):
    from gap_libero.benchmarks import cli

    monkeypatch.setattr(cli, "tasks", lambda selected: [{"files_present": False}])
    with pytest.raises(FileNotFoundError, match="official task"):
        cli.worker(SimpleNamespace(mode="capture", task_id=0), {})


def test_cli_prevents_output_into_custom_results(tmp_path, monkeypatch):
    from gap_libero.benchmarks import cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmarks", "capture", "--worker", "--output",
                                     str(tmp_path / "results/harness/native")])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not (tmp_path / "results/harness/native").exists()


def test_cli_no_motion_in_capture(monkeypatch):
    from gap_libero.benchmarks import cli

    monkeypatch.setattr(sys, "argv", ["benchmarks", "capture", "--enable-motion"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_pro_plan_suites_supported():
    from gap_libero.actions import RunPlan

    for suite in ("libero_object_object", "libero_object_swap"):
        plan = RunPlan(suite=suite, task_id=0, target_object="alphabet_soup_1",
                       workspace_m=((-1, -1, -1), (2, 2, 2)), actions={"wait": {"kind": "hold"}})
        assert plan.suite == suite


@pytest.mark.parametrize("mode", ["capture", "run"])
def test_benchmarks_delegate_to_existing_runner(tmp_path, monkeypatch, mode):
    from gap_libero import cli as shared
    from gap_libero.benchmarks import cli

    bddl = tmp_path / "task.bddl"
    states = tmp_path / "task.pruned_init"
    bddl.write_bytes(b"official task")
    states.write_bytes(b"official states")
    monkeypatch.setattr(cli, "tasks", lambda selected: [
        {"files_present": True, "bddl": str(bddl), "initial_states": str(states)}])
    args = SimpleNamespace(mode=mode, task_id=0, init_state_id=2, seed=3)
    calls = []

    def execute(received):
        calls.append(received)
        assert received.expected_selection == ("libero_object_object", 0, 2, 3)
        assert received.capture_initial_observation
        return {"ok": True, "plan_sha256": "shared runner evidence"}

    monkeypatch.setattr(shared, "execute", execute)
    result = cli.worker(args, {"suite": "libero_object_object"})
    assert calls == [args]
    assert result["plan_sha256"] == "shared runner evidence"
    assert result["simulation_evaluated"] == (mode == "run")
    assert result["task_definition_sha256"]


def test_existing_cli_dispatches_benchmark_selection(monkeypatch):
    from gap_libero import cli
    from gap_libero.benchmarks import cli as benchmarks

    calls = []
    monkeypatch.setattr(benchmarks, "main", lambda argv: calls.append(argv))
    monkeypatch.setattr(sys, "argv", ["gap-libero", "benchmark", "list", "--benchmark",
                                     "libero-pro-object"])
    cli.main()
    assert calls == [["list", "--benchmark", "libero-pro-object"]]

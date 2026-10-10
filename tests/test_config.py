from pathlib import Path

from gap_isaac.config import load_benchmark, load_task, validate_project

ROOT = Path(__file__).resolve().parents[1]


def test_project_configs_validate() -> None:
    summaries = validate_project(ROOT)
    assert len(summaries) == 3
    assert "benchmark demonstration_10.yaml: 10 trials" in summaries


def test_windows_platform_selects_only_the_rtx_5080() -> None:
    from gap_isaac.config import load_platform

    platform = load_platform(ROOT / "configs/platform/windows.yaml")
    assert platform.gpu.index == 0
    assert platform.gpu.expected_name == "NVIDIA GeForce RTX 5080"
    assert platform.gpu.exclusive is True


def test_demonstration_grid_has_exactly_10_trials() -> None:
    # keep the active benchmark limited to the agreed ten layouts
    config = load_benchmark(ROOT / "configs/benchmarks/demonstration_10.yaml")
    assert config.trial_count == 10
    assert config.seeds == tuple(range(10))
    assert config.task_ids == ("alphabet_soup",)
    assert config.workers == 1


def test_task_references_exist() -> None:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    ids = {item.id for item in task.objects}
    assert set(task.target_ids) <= ids
    assert task.container_id in ids

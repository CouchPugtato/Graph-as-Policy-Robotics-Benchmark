"""Typed project configuration and cross-file validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Range2D(StrictModel):
    x: tuple[float, float]
    y: tuple[float, float]

    @model_validator(mode="after")
    def ordered_bounds(self) -> Range2D:
        if self.x[0] >= self.x[1] or self.y[0] >= self.y[1]:
            raise ValueError("range lower bounds must be smaller than upper bounds")
        return self


class PoseVariation(StrictModel):
    xy_m: Range2D
    yaw_deg: tuple[float, float] = (-180.0, 180.0)

    @model_validator(mode="after")
    def ordered_yaw(self) -> PoseVariation:
        if self.yaw_deg[0] > self.yaw_deg[1]:
            raise ValueError("yaw lower bound must not exceed upper bound")
        return self


class ObjectSpec(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    semantic_label: str
    asset: str
    footprint_radius_m: float = Field(gt=0)
    target_description: str | None = None


class EvaluationConfig(StrictModel):
    lift_clearance_m: float = Field(gt=0)
    containment_margin_m: float = Field(ge=0)
    max_release_speed_m_s: float = Field(ge=0)
    collision_impulse_threshold_ns: float = Field(ge=0)
    basket_wall_thickness_m: float = Field(gt=0)
    basket_base_thickness_m: float = Field(gt=0)
    grasp_contact_frames: int = Field(gt=0, le=120)
    release_contact_frames: int = Field(gt=0, le=120)


class TaskConfig(StrictModel):
    schema_version: Literal[1]
    task_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    instruction_template: str
    robot: str
    target_ids: tuple[str, ...] = Field(min_length=1)
    container_id: str
    objects: tuple[ObjectSpec, ...] = Field(min_length=2)
    spawn: PoseVariation
    minimum_clearance_m: float = Field(ge=0)
    max_sampling_attempts: int = Field(gt=0, le=100_000)
    settle_time_s: float = Field(gt=0)
    timeout_s: float = Field(gt=0)
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def valid_references(self) -> TaskConfig:
        ids = [item.id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object ids must be unique")
        unknown = set(self.target_ids) | {self.container_id}
        unknown -= set(ids)
        if unknown:
            raise ValueError(f"unknown object references: {sorted(unknown)}")
        if self.container_id in self.target_ids:
            raise ValueError("container cannot also be a target")
        return self


class GpuConfig(StrictModel):
    index: int = Field(ge=0)
    expected_name: str
    exclusive: bool = True


class PlatformConfig(StrictModel):
    schema_version: Literal[1]
    role: Literal["simulator", "runtime"]
    operating_system: str
    isaac_sim_version: str | None = None
    isaac_lab_version: str | None = None
    ros_distribution: str | None = None
    ros_domain_id: int | None = Field(default=None, ge=0, le=232)
    gpu: GpuConfig
    required_environment: tuple[str, ...] = ()


class BenchmarkConfig(StrictModel):
    schema_version: Literal[1]
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    task_config: Path
    task_ids: tuple[str, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    workers: int = Field(gt=0)
    record_video: bool
    enable_tracing: bool
    checkpoint_mode: Literal["warn", "raise"]
    success_gate: float = Field(ge=0, le=1)
    output_dir: Path

    @model_validator(mode="after")
    def unique_trials(self) -> BenchmarkConfig:
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task ids must be unique")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be unique")
        return self

    @property
    def trial_count(self) -> int:
        return len(self.task_ids) * len(self.seeds)


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return data


def load_task(path: Path) -> TaskConfig:
    return TaskConfig.model_validate(_read_yaml(path))


def load_platform(path: Path) -> PlatformConfig:
    return PlatformConfig.model_validate(_read_yaml(path))


def load_benchmark(path: Path) -> BenchmarkConfig:
    data = _read_yaml(path)
    if "task_config" in data:
        data["task_config"] = (path.parent / data["task_config"]).resolve()
    if "output_dir" in data:
        data["output_dir"] = (path.parent / data["output_dir"]).resolve()
    return BenchmarkConfig.model_validate(data)


def validate_project(root: Path) -> list[str]:
    """Validate every checked-in configuration and return human-readable summaries."""

    summaries: list[str] = []
    for path in sorted((root / "configs" / "platform").glob("*.yaml")):
        config = load_platform(path)
        summaries.append(f"platform {path.name}: {config.role} / {config.operating_system}")

    task_by_path: dict[Path, TaskConfig] = {}
    for path in sorted((root / "configs" / "tasks").glob("*.yaml")):
        task = load_task(path)
        task_by_path[path.resolve()] = task
        summaries.append(f"task {path.name}: {len(task.target_ids)} targets")

    for path in sorted((root / "configs" / "benchmarks").glob("*.yaml")):
        benchmark = load_benchmark(path)
        task = task_by_path.get(benchmark.task_config)
        if task is None:
            raise ValueError(f"{path}: task_config is not a checked-in task config")
        unknown = set(benchmark.task_ids) - set(task.target_ids)
        if unknown:
            raise ValueError(f"{path}: unknown task ids: {sorted(unknown)}")
        summaries.append(f"benchmark {path.name}: {benchmark.trial_count} trials")
    return summaries

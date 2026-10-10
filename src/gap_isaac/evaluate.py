"""Privileged, simulator-side evaluation for grocery-fulfillment episodes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

from gap_isaac.config import EvaluationConfig, TaskConfig
from gap_isaac.grocery_scene import GEOMETRY, WORK_SURFACE_Z_M
from gap_isaac.reset import EpisodeResetPlan


class EpisodeStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TerminationReason(str, Enum):
    SUCCESS = "success"
    WRONG_OBJECT = "wrong_object"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    GRAPH_ERROR = "graph_error"


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    position_m: tuple[float, float, float]
    yaw_deg: float = 0.0
    linear_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CollisionEvent:
    body_a: str
    body_b: str
    impulse_ns: float
    allowed: bool = False


@dataclass(frozen=True)
class EpisodeFrame:
    elapsed_s: float
    objects: tuple[ObjectState, ...]
    held_object_id: str | None = None
    collisions: tuple[CollisionEvent, ...] = ()
    graph_error: str | None = None


@dataclass(frozen=True)
class EpisodeCheckpoints:
    target_grasped: bool = False
    target_lifted: bool = False
    target_released: bool = False
    target_contained: bool = False


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    target_id: str
    container_id: str
    status: EpisodeStatus
    termination_reason: TerminationReason | None
    elapsed_s: float
    checkpoints: EpisodeCheckpoints
    disallowed_collisions: tuple[CollisionEvent, ...]
    detail: str = ""

    @property
    def terminal(self) -> bool:
        return self.status is not EpisodeStatus.RUNNING

    def as_dict(self) -> dict:
        return asdict(self)


def _speed(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _rotate_xy(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (cosine * x - sine * y, sine * x + cosine * y)


def _object_corners_in_container(
    target: ObjectState,
    target_size_m: tuple[float, float, float],
    container: ObjectState,
) -> tuple[tuple[float, float], ...]:
    half_x, half_y = target_size_m[0] / 2, target_size_m[1] / 2
    corners: list[tuple[float, float]] = []
    for local_x, local_y in (
        (-half_x, -half_y),
        (-half_x, half_y),
        (half_x, -half_y),
        (half_x, half_y),
    ):
        world_offset = _rotate_xy(local_x, local_y, target.yaw_deg)
        relative_x = target.position_m[0] + world_offset[0] - container.position_m[0]
        relative_y = target.position_m[1] + world_offset[1] - container.position_m[1]
        corners.append(_rotate_xy(relative_x, relative_y, -container.yaw_deg))
    return tuple(corners)


def target_fits_basket_opening(
    target: ObjectState,
    container: ObjectState,
    evaluation: EvaluationConfig,
) -> bool:
    """Check the full horizontal footprint, without declaring vertical containment."""

    target_geometry = GEOMETRY[target.object_id]
    container_geometry = GEOMETRY[container.object_id]
    inner_half_x = (
        container_geometry.size_m[0] / 2
        - evaluation.basket_wall_thickness_m
        - evaluation.containment_margin_m
    )
    inner_half_y = (
        container_geometry.size_m[1] / 2
        - evaluation.basket_wall_thickness_m
        - evaluation.containment_margin_m
    )
    if inner_half_x <= 0 or inner_half_y <= 0:
        raise ValueError("basket evaluation dimensions leave no interior volume")
    return not any(
        abs(x) > inner_half_x or abs(y) > inner_half_y
        for x, y in _object_corners_in_container(
            target,
            target_geometry.size_m,
            container,
        )
    )


def target_is_contained(
    target: ObjectState,
    container: ObjectState,
    evaluation: EvaluationConfig,
) -> bool:
    """Return whether the full target proxy is within the basket's inner volume."""
    if not target_fits_basket_opening(target, container, evaluation):
        return False
    target_geometry = GEOMETRY[target.object_id]
    container_geometry = GEOMETRY[container.object_id]

    # success requires the whole object to fit rather than just its center
    target_half_height = target_geometry.size_m[2] / 2
    target_bottom = target.position_m[2] - target_half_height
    target_top = target.position_m[2] + target_half_height
    interior_bottom = container.position_m[2] + evaluation.basket_base_thickness_m
    interior_top = (
        container.position_m[2] + evaluation.basket_base_thickness_m + container_geometry.size_m[2]
    )
    tolerance_m = 0.005
    return (
        target_bottom >= interior_bottom - tolerance_m and target_top <= interior_top + tolerance_m
    )


class PrivilegedEpisodeEvaluator:
    """Stateful checkpoint evaluator that must never be exposed to the policy."""

    def __init__(self, task: TaskConfig, reset: EpisodeResetPlan) -> None:
        if reset.target_id not in task.target_ids:
            raise ValueError(f"reset target is not configured: {reset.target_id}")
        if reset.container_id != task.container_id:
            raise ValueError("reset container does not match task configuration")
        self._task = task
        self._reset = reset
        self._checkpoints = EpisodeCheckpoints()
        self._collisions: list[CollisionEvent] = []
        self._previous_held_object_id: str | None = None
        self._last_elapsed_s = -1.0
        self._result: EpisodeResult | None = None

    @property
    def result(self) -> EpisodeResult | None:
        return self._result

    def _finish(
        self,
        status: EpisodeStatus,
        reason: TerminationReason,
        elapsed_s: float,
        detail: str = "",
    ) -> EpisodeResult:
        self._result = EpisodeResult(
            seed=self._reset.seed,
            target_id=self._reset.target_id,
            container_id=self._reset.container_id,
            status=status,
            termination_reason=reason,
            elapsed_s=elapsed_s,
            checkpoints=self._checkpoints,
            disallowed_collisions=tuple(self._collisions),
            detail=detail,
        )
        return self._result

    def update(self, frame: EpisodeFrame) -> EpisodeResult:
        if self._result is not None:
            return self._result
        if frame.elapsed_s < 0:
            raise ValueError("elapsed time must not be negative")
        if frame.elapsed_s < self._last_elapsed_s:
            raise ValueError("episode frame time must be monotonic")
        self._last_elapsed_s = frame.elapsed_s
        objects = {item.object_id: item for item in frame.objects}
        if len(objects) != len(frame.objects):
            raise ValueError("episode frame contains duplicate object ids")
        try:
            target = objects[self._reset.target_id]
            container = objects[self._reset.container_id]
        except KeyError as error:
            raise ValueError(f"episode frame is missing object: {error.args[0]}") from error

        disallowed = tuple(
            collision
            for collision in frame.collisions
            if not collision.allowed
            and collision.impulse_ns >= self._task.evaluation.collision_impulse_threshold_ns
        )
        self._collisions.extend(disallowed)
        if disallowed:
            first = disallowed[0]
            return self._finish(
                EpisodeStatus.FAILED,
                TerminationReason.COLLISION,
                frame.elapsed_s,
                f"{first.body_a} collided with {first.body_b}",
            )
        if frame.held_object_id is not None and frame.held_object_id != self._reset.target_id:
            return self._finish(
                EpisodeStatus.FAILED,
                TerminationReason.WRONG_OBJECT,
                frame.elapsed_s,
                f"held {frame.held_object_id} instead of {self._reset.target_id}",
            )
        if frame.graph_error:
            return self._finish(
                EpisodeStatus.FAILED,
                TerminationReason.GRAPH_ERROR,
                frame.elapsed_s,
                frame.graph_error,
            )

        grasped = self._checkpoints.target_grasped or (
            frame.held_object_id == self._reset.target_id
        )
        target_bottom = target.position_m[2] - GEOMETRY[target.object_id].size_m[2] / 2
        lifted = self._checkpoints.target_lifted or (
            grasped and target_bottom >= WORK_SURFACE_Z_M + self._task.evaluation.lift_clearance_m
        )
        released_now = (
            self._previous_held_object_id == self._reset.target_id and frame.held_object_id is None
        )
        released = self._checkpoints.target_released or released_now
        contained = (
            lifted
            and released
            and frame.held_object_id is None
            and _speed(target.linear_velocity_m_s) <= self._task.evaluation.max_release_speed_m_s
            and target_is_contained(target, container, self._task.evaluation)
        )
        self._checkpoints = EpisodeCheckpoints(
            target_grasped=grasped,
            target_lifted=lifted,
            target_released=released,
            target_contained=self._checkpoints.target_contained or contained,
        )
        self._previous_held_object_id = frame.held_object_id

        if self._checkpoints.target_contained:
            return self._finish(
                EpisodeStatus.SUCCEEDED,
                TerminationReason.SUCCESS,
                frame.elapsed_s,
            )
        if frame.elapsed_s >= self._reset.timeout_s:
            return self._finish(
                EpisodeStatus.FAILED,
                TerminationReason.TIMEOUT,
                frame.elapsed_s,
            )
        return EpisodeResult(
            seed=self._reset.seed,
            target_id=self._reset.target_id,
            container_id=self._reset.container_id,
            status=EpisodeStatus.RUNNING,
            termination_reason=None,
            elapsed_s=frame.elapsed_s,
            checkpoints=self._checkpoints,
            disallowed_collisions=tuple(self._collisions),
        )

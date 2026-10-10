from pathlib import Path

import pytest

from gap_isaac.config import load_task
from gap_isaac.evaluate import (
    CollisionEvent,
    EpisodeFrame,
    EpisodeStatus,
    ObjectState,
    PrivilegedEpisodeEvaluator,
    TerminationReason,
    target_fits_basket_opening,
    target_is_contained,
)
from gap_isaac.grocery_scene import GEOMETRY, WORK_SURFACE_Z_M
from gap_isaac.reset import build_episode_reset_plan

ROOT = Path(__file__).resolve().parents[1]


def setup_episode() -> tuple:
    task = load_task(ROOT / "configs/tasks/grocery_fulfillment.yaml")
    reset = build_episode_reset_plan(task, seed=0, target_id="alphabet_soup")
    initial = {item.object_id: item for item in reset.objects}
    basket_reset = initial[reset.container_id]
    target_reset = initial[reset.target_id]
    basket = ObjectState("basket", basket_reset.position_m, basket_reset.yaw_deg)
    target = ObjectState("alphabet_soup", target_reset.position_m, target_reset.yaw_deg)
    return task, reset, basket, target


def frame(
    elapsed_s: float,
    target: ObjectState,
    basket: ObjectState,
    held: str | None = None,
    collisions: tuple[CollisionEvent, ...] = (),
    graph_error: str | None = None,
) -> EpisodeFrame:
    distractor = ObjectState("cream_cheese", (0.5, 0.2, 0.75))
    return EpisodeFrame(
        elapsed_s=elapsed_s,
        objects=(target, basket, distractor),
        held_object_id=held,
        collisions=collisions,
        graph_error=graph_error,
    )


def contained_target(basket: ObjectState, speed: float = 0.0) -> ObjectState:
    height = GEOMETRY["alphabet_soup"].size_m[2]
    base = 0.020
    return ObjectState(
        "alphabet_soup",
        (
            basket.position_m[0],
            basket.position_m[1],
            basket.position_m[2] + base + height / 2,
        ),
        yaw_deg=basket.yaw_deg,
        linear_velocity_m_s=(speed, 0.0, 0.0),
    )


def test_opening_fit_does_not_declare_a_suspended_object_contained() -> None:
    task, _reset, basket, _target = setup_episode()
    above = ObjectState("alphabet_soup", (basket.position_m[0], basket.position_m[1], 1.0))
    assert target_fits_basket_opening(above, basket, task.evaluation)
    assert not target_is_contained(above, basket, task.evaluation)
    outside = ObjectState("alphabet_soup", (basket.position_m[0] + 0.20, basket.position_m[1], 1.0))
    assert not target_fits_basket_opening(outside, basket, task.evaluation)


def test_containment_uses_rotated_basket_interior() -> None:
    task, _, _, _ = setup_episode()
    basket = ObjectState("basket", (0.6, 0.0, WORK_SURFACE_Z_M), yaw_deg=90.0)
    inside = contained_target(basket)
    outside = ObjectState(
        "alphabet_soup",
        (basket.position_m[0] + 0.15, basket.position_m[1], inside.position_m[2]),
    )

    assert target_is_contained(inside, basket, task.evaluation)
    assert not target_is_contained(outside, basket, task.evaluation)


def test_success_requires_grasp_lift_release_containment_and_settle() -> None:
    task, reset, basket, target = setup_episode()
    evaluator = PrivilegedEpisodeEvaluator(task, reset)
    height = GEOMETRY[target.object_id].size_m[2]
    lifted = ObjectState(
        target.object_id,
        (target.position_m[0], target.position_m[1], WORK_SURFACE_Z_M + height / 2 + 0.08),
    )

    grasp_result = evaluator.update(frame(1.0, target, basket, held=target.object_id))
    lift_result = evaluator.update(frame(2.0, lifted, basket, held=target.object_id))
    moving_result = evaluator.update(frame(3.0, contained_target(basket, speed=0.2), basket))
    success = evaluator.update(frame(4.0, contained_target(basket), basket))

    assert grasp_result.status is EpisodeStatus.RUNNING
    assert lift_result.checkpoints.target_lifted
    assert moving_result.status is EpisodeStatus.RUNNING
    assert success.status is EpisodeStatus.SUCCEEDED
    assert success.termination_reason is TerminationReason.SUCCESS
    assert success.checkpoints.target_contained
    assert evaluator.update(frame(5.0, target, basket)) is success


def test_wrong_object_is_terminal_failure() -> None:
    task, reset, basket, target = setup_episode()
    result = PrivilegedEpisodeEvaluator(task, reset).update(
        frame(1.0, target, basket, held="cream_cheese")
    )

    assert result.status is EpisodeStatus.FAILED
    assert result.termination_reason is TerminationReason.WRONG_OBJECT


def test_disallowed_collision_above_threshold_is_terminal() -> None:
    task, reset, basket, target = setup_episode()
    event = CollisionEvent("panda_link4", "basket", impulse_ns=2.0)
    result = PrivilegedEpisodeEvaluator(task, reset).update(
        frame(1.0, target, basket, collisions=(event,))
    )

    assert result.termination_reason is TerminationReason.COLLISION
    assert result.disallowed_collisions == (event,)


def test_allowed_and_low_impulse_contacts_do_not_fail() -> None:
    task, reset, basket, target = setup_episode()
    events = (
        CollisionEvent("finger", "alphabet_soup", impulse_ns=5.0, allowed=True),
        CollisionEvent("panda_link4", "table", impulse_ns=0.5),
    )
    result = PrivilegedEpisodeEvaluator(task, reset).update(
        frame(1.0, target, basket, collisions=events)
    )

    assert result.status is EpisodeStatus.RUNNING
    assert not result.disallowed_collisions


def test_timeout_and_graph_error_are_distinct_failures() -> None:
    task, reset, basket, target = setup_episode()
    timeout = PrivilegedEpisodeEvaluator(task, reset).update(frame(reset.timeout_s, target, basket))
    graph_error = PrivilegedEpisodeEvaluator(task, reset).update(
        frame(1.0, target, basket, graph_error="planner failed")
    )

    assert timeout.termination_reason is TerminationReason.TIMEOUT
    assert graph_error.termination_reason is TerminationReason.GRAPH_ERROR
    assert graph_error.detail == "planner failed"


def test_frame_time_must_be_monotonic() -> None:
    task, reset, basket, target = setup_episode()
    evaluator = PrivilegedEpisodeEvaluator(task, reset)
    evaluator.update(frame(2.0, target, basket))

    with pytest.raises(ValueError, match="monotonic"):
        evaluator.update(frame(1.0, target, basket))

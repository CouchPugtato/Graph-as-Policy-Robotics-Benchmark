"""Deterministic custom LIBERO scenario contracts without simulator imports."""

import math
import random
from dataclasses import replace

import numpy as np

from gap_isaac.config import load_task
from gap_isaac.grocery_scene import (
    GEOMETRY,
    GroceryScenePlan,
    PlannedGroceryObject,
    build_grocery_scene_plan,
)

SCENARIOS = ("fulfillment", "packing", "popcorn")
VARIATIONS = ("xy", "basket-swap", "permutation", "mixed")
PACKING_BASKET_SIZE = (0.32, 0.26, 0.15)
ITEMS = ("alphabet_soup", "tomato_sauce", "chocolate_pudding", "cream_cheese")
DESCRIPTIONS = {
    "alphabet_soup": "the blue upright cylinder",
    "tomato_sauce": "the red upright cylinder",
    "chocolate_pudding": "the brown upright rectangular box",
    "cream_cheese": "the flat yellow rectangular box",
    "basket": "the green open basket",
    "pan": "the blue rectangular handle attached to the gray pan",
    "knob": "the green rectangular stove knob",
    "burner": "the purple square burner on the stove",
}
PRIORS = {name: (spec.shape, spec.size_m) for name, spec in GEOMETRY.items()}
PRIORS.update(
    pan=("box", (0.12, 0.028, 0.024)),
    knob=("box", (0.05, 0.024, 0.05)),
    burner=("box", (0.16, 0.16, 0.01)),
)


def packing_items(seed):
    pairs = (
        ("alphabet_soup", "tomato_sauce"),
        ("chocolate_pudding", "cream_cheese"),
        ("alphabet_soup", "cream_cheese"),
        ("tomato_sauce", "chocolate_pudding"),
    )
    return pairs[seed % len(pairs)]


def oriented_containment(
    target_position,
    target_rotation,
    object_id,
    basket_position,
    basket_rotation,
    *,
    wall=0.015,
    base=0.02,
    margin=0.005,
    basket_size=None,
):
    shape, size = PRIORS[object_id]
    rotation = np.asarray(basket_rotation).T @ np.asarray(target_rotation)
    center = np.asarray(basket_rotation).T @ (
        np.asarray(target_position) - np.asarray(basket_position)
    )
    half = np.asarray(size) / 2
    if shape == "cylinder":
        axis = rotation[:, 2]
        extent = half[0] * np.sqrt(np.maximum(0, 1 - axis * axis)) + half[2] * np.abs(axis)
    else:
        extent = np.abs(rotation) @ half
    basket_size = PRIORS["basket"][1] if basket_size is None else basket_size
    opening = np.asarray(basket_size[:2]) / 2 - wall - margin
    return bool(
        np.all(np.abs(center[:2]) + extent[:2] <= opening)
        and center[2] - extent[2] >= base - 0.005
        and center[2] + extent[2] <= base + basket_size[2] + 0.005
    )


def validate_case(scenario, seed, variation):
    if (
        scenario not in SCENARIOS
        or variation not in VARIATIONS
        or type(seed) is not int
        or not 0 <= seed < 10
    ):
        raise ValueError("select a deployed scenario variation and seed from zero to nine")


def grocery_layout(root, seed, variation="mixed", basket_size=None, requested_ids=None):
    validate_case("fulfillment", seed, variation)
    task = load_task(root / "configs/tasks/grocery_fulfillment.yaml")
    target = requested_ids[0] if requested_ids else ITEMS[seed % len(ITEMS)]
    for attempt in range(100):
        plan = build_grocery_scene_plan(task, seed + attempt * 1009, target)
        items = list(plan.objects)
        positions = {item.object_id: item.position_m[:2] for item in items}
        if variation in ("basket-swap", "mixed"):
            positions[target], positions["basket"] = positions["basket"], positions[target]
        if variation in ("permutation", "mixed"):
            order = [positions[name] for name in ITEMS]
            random.Random(seed + attempt).shuffle(order)
            positions.update(zip(ITEMS, order, strict=True))
        items = [
            replace(
                item,
                position_m=(*positions[item.object_id], item.position_m[2]),
                size_m=basket_size if item.object_id == "basket" and basket_size else item.size_m,
            )
            for item in items
        ]
        if requested_ids:
            items = [item for item in items if item.object_id in (*requested_ids, "basket")]
        if all(
            math.dist(a.position_m[:2], b.position_m[:2])
            >= math.hypot(*a.size_m[:2]) / 2
            + math.hypot(*b.size_m[:2]) / 2
            + (0.06 if requested_ids and (a.is_container or b.is_container) else 0.008)
            for i, a in enumerate(items)
            for b in items[i + 1 :]
        ):
            return GroceryScenePlan(seed, target, tuple(items))
    raise ValueError("could not sample a separated grocery layout within the bounded budget")


def popcorn_layout(seed):
    validate_case("popcorn", seed, "xy")
    rng = random.Random(seed)

    def item(name, shape, size, position, color, mass=0, yaw=0):
        return PlannedGroceryObject(name, name, shape, size, mass, color, position, yaw, False)

    pan_yaw = rng.uniform(-15, 15)
    objects = (
        item("stand", "fixture", (0.26, 0.23, 0.10), (0.59, 0.24, 0.05), (0.45, 0.45, 0.45)),
        item(
            "pan",
            "pan",
            (0.25, 0.15, 0.06),
            (0.59 + rng.uniform(-0.01, 0.01), 0.24, 0.121),
            (0.12, 0.32, 0.9),
            0.2,
            pan_yaw,
        ),
        item("stove", "fixture", (0.28, 0.26, 0.11), (0.48, -0.21, 0.055), (0.4, 0.4, 0.4)),
        item("burner", "fixture", (0.16, 0.16, 0.01), (0.48, -0.19, 0.115), (0.65, 0.1, 0.85)),
        item("knob", "knob", (0.05, 0.024, 0.05), (0.48, -0.46, 0.05), (0.1, 0.7, 0.2)),
    )
    return GroceryScenePlan(seed, "pan", objects)

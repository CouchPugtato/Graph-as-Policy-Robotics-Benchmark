"""Pure-data plan for the deterministic Isaac grocery proxy scene."""

from __future__ import annotations

from dataclasses import dataclass

from gap_isaac.config import TaskConfig
from gap_isaac.sampling import sample_instance

WORK_SURFACE_Z_M = 0.0
ROBOT_BASE_POSITION_M = (0.0, 0.0, WORK_SURFACE_Z_M)
GROUND_PRIM_PATH = "/World/GroundPlane"
GROUND_POSITION_M = (0.45, 0.0, -0.05)
GROUND_SIZE_M = (6.0, 6.0, 0.10)
SCENE_LAYOUT = "ground-v1"


def validate_ground_scene(stage) -> None:
    """Reject old tabletop stages before applying ground-level reset/motion plans."""
    if (
        not stage.GetPrimAtPath(GROUND_PRIM_PATH).IsValid()
        or stage.GetPrimAtPath("/World/Task/TableTop").IsValid()
    ):
        raise RuntimeError(
            "Ground-level scene required. Build a new stage with create-grocery-scene.ps1; "
            "the previous tabletop stage cannot use ground-level reset plans."
        )
    evaluation = stage.GetPrimAtPath("/World/Evaluation")
    if not evaluation.IsValid() or evaluation.GetAttribute("gap:sceneLayout").Get() != SCENE_LAYOUT:
        raise RuntimeError("Stage is missing ground-v1 layout metadata; rebuild the grocery scene")


@dataclass(frozen=True)
class GroceryGeometry:
    shape: str
    size_m: tuple[float, float, float]
    mass_kg: float
    color_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class PlannedGroceryObject:
    object_id: str
    semantic_label: str
    shape: str
    size_m: tuple[float, float, float]
    mass_kg: float
    color_rgb: tuple[float, float, float]
    position_m: tuple[float, float, float]
    yaw_deg: float
    is_container: bool


@dataclass(frozen=True)
class GroceryScenePlan:
    seed: int
    target_id: str
    objects: tuple[PlannedGroceryObject, ...]


GEOMETRY: dict[str, GroceryGeometry] = {
    "alphabet_soup": GroceryGeometry("cylinder", (0.070, 0.070, 0.110), 0.35, (0.10, 0.35, 0.90)),
    "chocolate_pudding": GroceryGeometry("box", (0.070, 0.050, 0.105), 0.25, (0.35, 0.12, 0.05)),
    "tomato_sauce": GroceryGeometry("cylinder", (0.068, 0.068, 0.105), 0.34, (0.85, 0.08, 0.05)),
    "cream_cheese": GroceryGeometry("box", (0.115, 0.055, 0.035), 0.20, (0.90, 0.82, 0.35)),
    "basket": GroceryGeometry("basket", (0.240, 0.200, 0.150), 0.0, (0.15, 0.65, 0.25)),
}


def build_grocery_scene_plan(
    task: TaskConfig,
    seed: int,
    target_id: str | None = None,
) -> GroceryScenePlan:
    """Convert a sampled task instance into simulator-independent geometry."""

    configured_ids = {item.id for item in task.objects}
    missing = configured_ids - GEOMETRY.keys()
    if missing:
        raise ValueError(f"missing proxy geometry for: {sorted(missing)}")
    instance = sample_instance(task, seed=seed, target_id=target_id)
    specs = {item.id: item for item in task.objects}
    planned: list[PlannedGroceryObject] = []
    for pose in instance.poses:
        geometry = GEOMETRY[pose.object_id]
        is_container = pose.object_id == task.container_id
        z = WORK_SURFACE_Z_M if is_container else WORK_SURFACE_Z_M + geometry.size_m[2] / 2 + 0.003
        planned.append(
            PlannedGroceryObject(
                object_id=pose.object_id,
                semantic_label=specs[pose.object_id].semantic_label,
                shape=geometry.shape,
                size_m=geometry.size_m,
                mass_kg=geometry.mass_kg,
                color_rgb=geometry.color_rgb,
                position_m=(pose.x_m, pose.y_m, z),
                yaw_deg=pose.yaw_deg,
                is_container=is_container,
            )
        )
    return GroceryScenePlan(seed=seed, target_id=instance.target_id, objects=tuple(planned))

"""Read and validate the Franka finger actuator topology without changing drives."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

FINGER_NAMES = ("panda_finger_joint1", "panda_finger_joint2")


@dataclass(frozen=True)
class FingerDrive:
    name: str
    index: int
    joint_path: str
    stiffness_n_m: float
    damping_n_s_m: float
    effort_limit_n: float | None
    mimic_reference: str | None = None
    mimic_gearing: float | None = None
    mimic_offset_m: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def read_finger_drives(stage: Any, robot: Any, controller: Any) -> tuple[FingerDrive, ...]:
    """Inspect USD mimic schemas and live physics gains after world initialization."""

    names = tuple(str(name) for name in robot.dof_names)
    kps, kds = controller.get_gains()
    efforts = controller.get_max_efforts()
    joint_prims = {}
    for prim in stage.Traverse():
        if (
            str(prim.GetPath()).startswith("/World/Franka/")
            and prim.GetName() in FINGER_NAMES
            and prim.GetTypeName() == "PhysicsPrismaticJoint"
        ):
            if prim.GetName() in joint_prims:
                raise RuntimeError(f"ambiguous finger joint: {prim.GetName()}")
            joint_prims[prim.GetName()] = prim

    records = []
    for name in FINGER_NAMES:
        if name not in joint_prims:
            raise RuntimeError(f"USD finger joint not found: {name}")
        prim = joint_prims[name]
        index = names.index(name)
        schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
        mimic_schemas = [schema for schema in schemas if "MimicJointAPI" in schema]
        reference, gearing, offset = None, None, None
        if mimic_schemas:
            if len(mimic_schemas) != 1:
                raise RuntimeError(f"multiple mimic constraints on {name}: {mimic_schemas}")
            axis = mimic_schemas[0].split(":")[-1]
            prefix = f"physxMimicJoint:{axis}"
            targets = prim.GetRelationship(f"{prefix}:referenceJoint").GetTargets()
            if len(targets) != 1:
                raise RuntimeError(f"mimic finger must reference exactly one joint: {name}")
            reference = str(targets[0])
            gearing = prim.GetAttribute(f"{prefix}:gearing").Get()
            offset = prim.GetAttribute(f"{prefix}:offset").Get()
        if any(schema.startswith("NewtonMimicAPI") for schema in schemas):
            raise RuntimeError("this validator requires the PhysX Franka asset")
        records.append(
            FingerDrive(
                name,
                index,
                str(prim.GetPath()),
                float(kps[index]),
                float(kds[index]),
                float(efforts[index]) if efforts is not None else None,
                reference,
                float(gearing) if gearing is not None else None,
                float(offset) if offset is not None else None,
            )
        )
    return tuple(records)


def select_finger_actuators(drives: tuple[FingerDrive, ...]) -> tuple[FingerDrive, ...]:
    """Accept two position drives or a position-driven leader with a 1:1 follower."""

    if {drive.name for drive in drives} != set(FINGER_NAMES) or len(drives) != 2:
        raise ValueError("expected exactly two Franka finger joints")
    for drive in drives:
        if (
            not math.isfinite(drive.stiffness_n_m)
            or drive.stiffness_n_m < 0
            or not math.isfinite(drive.damping_n_s_m)
            or drive.damping_n_s_m < 0
        ):
            raise ValueError(f"invalid drive gains on {drive.name}")
    active = tuple(drive for drive in drives if drive.mimic_reference is None)
    if not active or any(drive.stiffness_n_m <= 0 for drive in active):
        raise ValueError("active finger position drive is missing; inspect finger_drives")
    for follower in drives:
        if follower.mimic_reference is None:
            continue
        if len(active) != 1 or follower.mimic_reference != active[0].joint_path:
            raise ValueError("mimic follower does not reference the active finger")
        if (
            follower.mimic_gearing is None
            or not math.isclose(follower.mimic_gearing, -1.0, abs_tol=1e-6)
            or follower.mimic_offset_m is None
            or not math.isclose(follower.mimic_offset_m, 0.0, abs_tol=1e-6)
        ):
            raise ValueError("only symmetric 1:1 Franka finger mimic constraints are supported")
        if follower.stiffness_n_m != 0:
            raise ValueError("mimic follower has a competing position drive")
    return active


def plan_finger_preload(
    drives: tuple[FingerDrive, ...], *, force_per_finger_n: float, maximum_deflection_m: float
) -> tuple[FingerDrive, ...]:
    """Size active spring stiffness for the preload without raising effort limits."""

    if any(
        not math.isfinite(value) or value <= 0
        for value in (force_per_finger_n, maximum_deflection_m)
    ):
        raise ValueError("preload force and deflection must be finite and positive")
    active = select_finger_actuators(drives)
    actuator_force_n = force_per_finger_n * (2.0 if len(active) == 1 else 1.0)
    for drive in active:
        if (
            drive.effort_limit_n is None
            or not math.isfinite(drive.effort_limit_n)
            or drive.effort_limit_n < actuator_force_n
        ):
            raise ValueError(
                f"{drive.name} effort limit {drive.effort_limit_n} N cannot support "
                f"the {actuator_force_n} N nominal actuator preload"
            )
    return tuple(
        replace(
            drive, stiffness_n_m=max(drive.stiffness_n_m, actuator_force_n / maximum_deflection_m)
        )
        for drive in active
    )

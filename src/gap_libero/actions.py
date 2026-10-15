"""Map existing graph action names to bounded LIBERO OSC commands."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gap_isaac.graph_policy import GraphPolicy, GraphPolicyExecutor, NodeResult
from gap_libero.backend import LiberoBackend, vector


class ActionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["move", "open", "close", "hold", "verify"]
    position_m: tuple[float, float, float] | None = None
    max_steps: int = Field(default=60, ge=1, le=200, strict=True)
    tolerance_m: float = Field(default=0.005, gt=0, le=0.01)
    require_grasp: bool = False

    @model_validator(mode="after")
    def move_requires_position(self) -> ActionSpec:
        if (self.kind == "move") != (self.position_m is not None):
            raise ValueError("only move actions require a position")
        return self


class RunPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    suite: Literal["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
    task_id: int = Field(ge=0, strict=True)
    init_state_id: int = Field(default=0, ge=0, strict=True)
    seed: int = Field(default=0, ge=0, strict=True)
    target_object: str = Field(min_length=1)
    workspace_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    max_episode_steps: int = Field(default=400, ge=1, le=1000, strict=True)
    maximum_slip_m: float = Field(default=0.015, gt=0, le=0.02)
    forbidden_geom_pairs: list[tuple[str, str]] = Field(default_factory=list)
    actions: dict[str, ActionSpec]

    @model_validator(mode="after")
    def check_workspace(self) -> RunPlan:
        low, high = np.asarray(self.workspace_m)
        if not (low < high).all():
            raise ValueError("workspace corners must be ordered")
        for action in self.actions.values():
            if action.position_m is not None:
                position = np.asarray(action.position_m)
                if (position < low).any() or (position > high).any():
                    raise ValueError("planned position is outside the workspace")
        if not self.actions:
            raise ValueError("plan must declare actions")
        return self


class LiberoGraphActions:
    def __init__(self, backend: LiberoBackend, plan: RunPlan) -> None:
        self.backend, self.plan = backend, plan
        self.gripper = -1.0
        self.hold_reference: np.ndarray | None = None
        self.missing_contact_frames = 0
        backend.target_object(plan.target_object)
        controller = backend.env.robots[0].controller
        if not controller.use_delta:
            raise ValueError("LIBERO adapter requires a delta OSC controller")
        low = vector(controller.input_min, 6, "controller input minimum")
        high = vector(controller.input_max, 6, "controller input maximum")
        output_low = vector(controller.output_min, 6, "controller output minimum")
        output_high = vector(controller.output_max, 6, "controller output maximum")
        if not np.allclose(low, -1) or not np.allclose(high, 1):
            raise ValueError("OSC controller must accept normalized inputs")
        if not np.allclose(output_low, -output_high) or not (output_high > 0).all():
            raise ValueError("OSC controller outputs must be symmetric and positive")
        self.position_scale = output_high[:3]

    def check_hold(self) -> None:
        if self.hold_reference is None:
            raise RuntimeError("no verified grasp has been established")
        self.missing_contact_frames = (0 if self.backend.grasped(self.plan.target_object)
                                       else self.missing_contact_frames + 1)
        if self.missing_contact_frames >= 3:
            raise RuntimeError("target contact was lost")
        slip = np.linalg.norm(
            self.backend.target_in_gripper(self.plan.target_object) - self.hold_reference)
        if not np.isfinite(slip) or slip > self.plan.maximum_slip_m:
            raise RuntimeError(f"target slipped relative to the gripper: {slip:.6f} m")

    def command(self, translation: Any = (0, 0, 0), *, require_grasp: bool = False) -> None:
        if require_grasp:
            self.check_hold()
        # zero rotation retains the current wrist orientation rather than using lula
        self.backend.step([*translation, 0, 0, 0, self.gripper])
        if require_grasp:
            self.check_hold()

    def execute_action(self, name: str, _state: dict[str, Any]) -> NodeResult:
        spec = self.plan.actions[name]
        if self.backend.done:
            if self.backend.task_success():
                return NodeResult(detail="LIBERO task terminal reached; action skipped")
            raise RuntimeError("LIBERO episode ended before task success")
        if spec.kind == "verify":
            success = self.backend.task_success()
            return NodeResult("succeeded" if success else "failed", "LIBERO task success predicate")
        if spec.kind == "move":
            self.backend.check_workspace(spec.position_m)
            for _ in range(spec.max_steps):
                error = np.asarray(spec.position_m) - self.backend.eef_position()
                if np.linalg.norm(error) <= spec.tolerance_m:
                    if spec.require_grasp:
                        self.check_hold()
                    return NodeResult(detail="Cartesian position reached within tolerance")
                translation = np.clip(error / self.position_scale, -0.15, 0.15)
                self.command(translation, require_grasp=spec.require_grasp)
                if self.backend.done:
                    return NodeResult("succeeded" if self.backend.task_success() else "failed",
                                      "LIBERO episode terminal during motion")
            return NodeResult("failed", "Cartesian move exhausted its step budget")
        if spec.kind in ("open", "close"):
            self.gripper = -1.0 if spec.kind == "open" else 1.0
            stable_contacts = 0
            for _ in range(spec.max_steps):
                self.command()
                if self.backend.done:
                    return NodeResult("succeeded" if self.backend.task_success() else "failed",
                                      "LIBERO episode terminal during gripper action")
                if spec.kind == "close":
                    stable_contacts = (stable_contacts + 1
                                       if self.backend.grasped(self.plan.target_object) else 0)
                    if stable_contacts >= 3:
                        self.hold_reference = self.backend.target_in_gripper(
                            self.plan.target_object)
                        self.missing_contact_frames = 0
                        return NodeResult(detail="two finger target contact verified")
            if spec.kind == "close":
                return NodeResult("failed", "grasp contact not verified before timeout")
            self.hold_reference = None
            outcome = "failed" if self.backend.grasped(self.plan.target_object) else "succeeded"
            return NodeResult(outcome, "target release contact check")
        for _ in range(spec.max_steps):
            self.command(require_grasp=spec.require_grasp)
            if self.backend.done:
                return NodeResult("succeeded" if self.backend.task_success() else "failed",
                                  "LIBERO episode terminal during hold")
        return NodeResult(detail="bounded hold completed")

    def registry(self) -> dict[str, Any]:
        return {name: (lambda state, key=name: self.execute_action(key, state))
                for name in self.plan.actions}


def run_graph(backend: LiberoBackend, policy: GraphPolicy, plan: RunPlan) -> dict[str, Any]:
    if not backend.motion_enabled:
        raise RuntimeError("enable motion explicitly before running a LIBERO graph")
    if backend.task_success():
        raise RuntimeError("task is already successful at reset; refusing a trivial success")
    actions = LiberoGraphActions(backend, plan)
    result = GraphPolicyExecutor(policy, actions.registry(), max_steps=64).execute()
    success = backend.task_success()
    return {"ok": result.status == "succeeded" and success, "backend": "libero",
            "graph": result.as_dict(), "steps": backend.steps,
            "evaluation": {"status": "succeeded" if success else "failed",
                           "task_success": success,
                           "collision_check": "configured geometry pairs only",
                           "grasp_check": "two finger contact and gripper relative slip"}}

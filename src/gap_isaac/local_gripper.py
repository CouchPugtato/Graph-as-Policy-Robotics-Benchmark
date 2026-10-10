"""Bounded simulator-local Franka gripper control."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gap_isaac.contact_telemetry import ContactTelemetry
from gap_isaac.protocol import CommandResult


@dataclass(frozen=True)
class GripperSafetyPolicy:
    open_position_m: float = 0.04
    closed_position_m: float = 0.0
    max_finger_speed_m_s: float = 0.02
    timeout_s: float = 5.0
    position_tolerance_m: float = 0.0005
    collision_impulse_threshold_ns: float = 1.0
    grasp_settle_s: float = 0.25
    grasp_force_n: float = 0.0
    max_grasp_deflection_m: float = 0.0005
    grasp_force_multiplier: float = 1.0


class LocalGripperExecutor:
    """Move both finger joints symmetrically and stop on verified contact."""

    def __init__(
        self,
        robot: Any,
        *,
        physics_dt_s: float,
        step: Callable[[], None],
        action_factory: Callable[[tuple[float, ...], tuple[int, ...]], Any],
        contact_capture: Callable[[], ContactTelemetry],
        policy: GripperSafetyPolicy | None = None,
        finger_stiffness_n_m: tuple[float, ...] | None = None,
        controlled_finger_names: tuple[str, ...] | None = None,
    ) -> None:
        if physics_dt_s <= 0:
            raise ValueError("physics time step must be positive")
        names = getattr(robot, "dof_names", None)
        if names is None:
            raise RuntimeError("robot articulation does not expose DOF names")
        self._robot = robot
        self._finger_indices = (
            tuple(str(name) for name in names).index("panda_finger_joint1"),
            tuple(str(name) for name in names).index("panda_finger_joint2"),
        )
        active_names = controlled_finger_names or ("panda_finger_joint1", "panda_finger_joint2")
        if len(set(active_names)) != len(active_names) or not set(active_names) <= {
            "panda_finger_joint1",
            "panda_finger_joint2",
        }:
            raise ValueError("controlled joints must be distinct Franka fingers")
        self._indices = tuple(
            tuple(str(name) for name in names).index(name) for name in active_names
        )
        self._physics_dt_s = physics_dt_s
        self._step = step
        self._action_factory = action_factory
        self._contact_capture = contact_capture
        self._policy = policy or GripperSafetyPolicy()
        if self._policy.grasp_settle_s < 0:
            raise ValueError("grasp settle time must not be negative")
        if not math.isfinite(self._policy.grasp_force_n) or self._policy.grasp_force_n < 0:
            raise ValueError("grasp force must be finite and nonnegative")
        if (
            not math.isfinite(self._policy.max_grasp_deflection_m)
            or self._policy.max_grasp_deflection_m < 0
        ):
            raise ValueError("maximum grasp deflection must be finite and nonnegative")
        if self._policy.grasp_force_n > 0 and (
            finger_stiffness_n_m is None
            or len(finger_stiffness_n_m) != len(self._indices)
            or any(not math.isfinite(value) or value <= 0 for value in finger_stiffness_n_m)
        ):
            raise ValueError("force preload requires finite positive active finger stiffnesses")
        if not math.isfinite(
            self._policy.grasp_force_multiplier
        ) or self._policy.grasp_force_multiplier not in (1.0, 2.0):
            raise ValueError("grasp force multiplier must be 1 or 2")
        self._finger_stiffness_n_m = finger_stiffness_n_m
        self._stop = threading.Event()
        self.preload_diagnostics: dict = {}

    def request_stop(self) -> None:
        self._stop.set()

    def _positions(self) -> tuple[float, ...]:
        positions = self._robot.get_joint_positions()
        return tuple(float(positions[index]) for index in self._indices)

    def _hold(self) -> None:
        self._robot.apply_action(self._action_factory(self._positions(), self._indices))

    def _stabilize_grasp(
        self, command_id: str, object_id: str, *, initial_commanded: tuple[float, ...]
    ) -> CommandResult:
        """Verify contact, optionally retaining a force-calibrated spring preload.

        Preload uses F / stiffness, capped by a bounded command offset per finger.
        This is a drive-target offset, not allowed collision penetration.
        The resulting joint target stays fixed; it never chases a slipping object.
        This is a nominal spring force, not a measured contact-force guarantee.
        """

        settle_steps = math.ceil(self._policy.grasp_settle_s / self._physics_dt_s)
        preload_targets = None
        # contact can stop the fingers before the commanded closed position
        contact_positions = self._positions()
        # Preserve the closing drive's current load. Restarting from actual
        # positions would remove its existing spring force at contact.
        preload_commanded = initial_commanded
        preload_steps = 0
        if self._policy.grasp_force_n > 0:
            assert self._finger_stiffness_n_m is not None
            deflections = tuple(
                min(
                    self._policy.grasp_force_n * self._policy.grasp_force_multiplier / stiffness,
                    self._policy.max_grasp_deflection_m,
                )
                for stiffness in self._finger_stiffness_n_m
            )
            # preload changes the drive target rather than attaching the object
            preload_targets = tuple(
                max(self._policy.closed_position_m, value - deflection)
                for value, deflection in zip(contact_positions, deflections, strict=True)
            )
            max_delta = self._policy.max_finger_speed_m_s * self._physics_dt_s
            preload_steps = math.ceil(
                max(
                    abs(target - value)
                    for target, value in zip(preload_targets, preload_commanded, strict=True)
                )
                / max_delta
            )
        self.preload_diagnostics = {
            "contact_positions_m": list(contact_positions),
            "incoming_command_positions_m": list(initial_commanded),
            "fixed_preload_targets_m": list(preload_targets) if preload_targets else None,
            "stiffness_n_m": list(self._finger_stiffness_n_m or ()),
        }
        telemetry = None
        for _ in range(preload_steps + settle_steps):
            if self._stop.is_set():
                self._hold()
                return CommandResult(command_id=command_id, status="canceled")
            if preload_targets is not None:
                preload_commanded = tuple(
                    value + max(-max_delta, min(max_delta, target - value))
                    for value, target in zip(preload_commanded, preload_targets, strict=True)
                )
                self._robot.apply_action(self._action_factory(preload_commanded, self._indices))
            self._step()
            telemetry = self._contact_capture()
            self.preload_diagnostics.update(
                last_command_positions_m=list(preload_commanded),
                last_actual_positions_m=list(self._positions()),
                held_object_id=telemetry.held_object_id,
            )
            disallowed = [
                event
                for event in telemetry.collisions
                if not event.allowed
                and event.impulse_ns >= self._policy.collision_impulse_threshold_ns
            ]
            if disallowed:
                self._hold()
                strongest = max(disallowed, key=lambda event: event.impulse_ns)
                return CommandResult(
                    command_id=command_id,
                    status="failed",
                    detail=f"disallowed collision: {strongest.body_a} with {strongest.body_b}",
                )
            if telemetry.held_object_id not in (None, object_id):
                self._hold()
                return CommandResult(
                    command_id=command_id,
                    status="failed",
                    detail=f"holding unexpected object {telemetry.held_object_id}",
                )
            if preload_targets is not None and telemetry.held_object_id is None:
                self._hold()
                return CommandResult(
                    command_id=command_id,
                    status="failed",
                    detail=f"lost {object_id} during force-calibrated preload",
                )
        if telemetry is None or telemetry.held_object_id != object_id:
            self._hold()
            return CommandResult(
                command_id=command_id,
                status="failed",
                detail=f"lost {object_id} while stabilizing verified contact",
            )
        # Leave the last verified finger target active. Arm-only actions do not
        # overwrite the finger targets, so the physical hold persists.
        return CommandResult(
            command_id=command_id,
            status="succeeded",
            detail=(
                f"holding {object_id} with force-calibrated preload "
                f"(nominal up to {self._policy.grasp_force_n:.3f} N per finger, "
                f"deflection cap {self._policy.max_grasp_deflection_m:.6f} m)"
                if preload_targets is not None
                else f"holding {object_id} at stabilized contact"
            ),
        )

    def move(
        self,
        command_id: str,
        *,
        open_gripper: bool,
        require_held_object_id: str | None = None,
    ) -> CommandResult:
        if self._stop.is_set():
            return CommandResult(command_id=command_id, status="canceled", detail="stop is latched")
        if not open_gripper:
            self.preload_diagnostics = {}
        target = self._policy.open_position_m if open_gripper else self._policy.closed_position_m
        elapsed_s = 0.0
        max_delta = self._policy.max_finger_speed_m_s * self._physics_dt_s
        commanded = self._positions()
        while elapsed_s < self._policy.timeout_s:
            if self._stop.is_set():
                self._hold()
                return CommandResult(command_id=command_id, status="canceled")
            next_positions = tuple(
                value + max(-max_delta, min(max_delta, target - value)) for value in commanded
            )
            commanded = next_positions
            self._robot.apply_action(self._action_factory(next_positions, self._indices))
            self._step()
            elapsed_s += self._physics_dt_s
            telemetry = self._contact_capture()
            disallowed = [
                event
                for event in telemetry.collisions
                if not event.allowed
                and event.impulse_ns >= self._policy.collision_impulse_threshold_ns
            ]
            if disallowed:
                self._hold()
                strongest = max(disallowed, key=lambda event: event.impulse_ns)
                return CommandResult(
                    command_id=command_id,
                    status="failed",
                    detail=f"disallowed collision: {strongest.body_a} with {strongest.body_b}",
                )
            if require_held_object_id is not None:
                if telemetry.held_object_id == require_held_object_id:
                    return self._stabilize_grasp(
                        command_id, require_held_object_id, initial_commanded=commanded
                    )
                if telemetry.held_object_id is not None:
                    self._hold()
                    return CommandResult(
                        command_id=command_id,
                        status="failed",
                        detail=f"holding unexpected object {telemetry.held_object_id}",
                    )
            error = max(abs(value - target) for value in self._positions())
            if error <= self._policy.position_tolerance_m:
                if require_held_object_id is not None:
                    return CommandResult(
                        command_id=command_id,
                        status="failed",
                        detail="gripper closed without verifying a two-finger hold",
                    )
                return CommandResult(command_id=command_id, status="succeeded")
        actual = self._positions()
        if require_held_object_id is not None:
            timeout_detail = (
                f"two-finger hold of {require_held_object_id} was not verified before timeout; "
                f"active_targets_m={commanded}, active_positions_m={actual}, "
                f"separation_m={self.separation_m():.6f}, "
                f"held_object={telemetry.held_object_id or 'none'}, "
                f"raw_contacts={telemetry.raw_contact_count}"
            )
        else:
            timeout_detail = f"gripper did not reach {target:.3f} m before timeout"
        self._hold()
        return CommandResult(
            command_id=command_id,
            status="timed_out",
            detail=timeout_detail,
        )

    def separation_m(self) -> float:
        positions = self._robot.get_joint_positions()
        return math.fsum(float(positions[index]) for index in self._finger_indices)

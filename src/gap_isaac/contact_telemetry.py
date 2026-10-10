"""Interpret privileged Isaac contact records and infer gripper holds."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gap_isaac.evaluate import CollisionEvent
from gap_isaac.grocery_scene import GROUND_PRIM_PATH

FRANKA_CONTACT_LINKS = tuple(
    [f"/World/Franka/panda_link{index}" for index in range(8)]
    + [
        "/World/Franka/panda_hand",
        "/World/Franka/panda_leftfinger",
        "/World/Franka/panda_rightfinger",
    ]
)
FRANKA_LEFT_FINGER = "/World/Franka/panda_leftfinger"
FRANKA_RIGHT_FINGER = "/World/Franka/panda_rightfinger"
MINIMUM_HOLD_IMPULSE_NS = 1e-9
# Only the fixed mount may rest on the floor; moving links/fingers remain monitored.
FRANKA_GROUND_MOUNT_CONTACTS = (("/World/Franka/panda_link0", GROUND_PRIM_PATH),)


def scale_contact_frame_count(configured_frames: int, physics_hz: int) -> int:
    """Preserve the configured 60 Hz contact-debounce duration at higher rates."""
    if configured_frames < 1 or physics_hz not in (60, 120):
        raise ValueError("contact debounce needs positive frames and 60 or 120 Hz physics")
    # faster physics should not shorten the time needed to verify a hold
    return math.ceil(configured_frames * physics_hz / 60)


@dataclass(frozen=True)
class ContactSample:
    body_a: str
    body_b: str
    impulse_xyz_ns: tuple[float, float, float]

    @property
    def impulse_ns(self) -> float:
        return math.sqrt(sum(component * component for component in self.impulse_xyz_ns))


@dataclass(frozen=True)
class ContactTelemetry:
    held_object_id: str | None
    collisions: tuple[CollisionEvent, ...]
    raw_contact_count: int


def _is_at_or_below(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _matching_root(path: str, roots: Mapping[str, str]) -> str | None:
    matches = (
        (object_id, root) for object_id, root in roots.items() if _is_at_or_below(path, root)
    )
    try:
        return max(matches, key=lambda item: len(item[1]))[0]
    except ValueError:
        return None


class ContactInterpreter:
    """Classify robot contacts and debounce a two-finger hold signal."""

    def __init__(
        self,
        *,
        object_roots: Mapping[str, str],
        container_id: str,
        robot_root: str,
        left_finger_root: str,
        right_finger_root: str,
        grasp_contact_frames: int,
        release_contact_frames: int,
        stationary_mount_contacts: Sequence[tuple[str, str]] = (),
    ) -> None:
        if grasp_contact_frames < 1 or release_contact_frames < 1:
            raise ValueError("contact debounce frame counts must be positive")
        if container_id not in object_roots:
            raise ValueError("container is missing from object roots")
        self._object_roots = dict(object_roots)
        self._graspable_ids = set(object_roots) - {container_id}
        self._robot_root = robot_root
        self._left_finger_root = left_finger_root
        self._right_finger_root = right_finger_root
        self._grasp_contact_frames = grasp_contact_frames
        self._release_contact_frames = release_contact_frames
        self._stationary_mount_contacts = tuple(stationary_mount_contacts)
        self.reset()

    def reset(self) -> None:
        self._held_object_id: str | None = None
        self._candidate_id: str | None = None
        self._candidate_frames = 0
        self._lost_frames = 0

    def _finger_object(self, sample: ContactSample, finger_root: str) -> str | None:
        if _is_at_or_below(sample.body_a, finger_root):
            candidate = _matching_root(sample.body_b, self._object_roots)
        elif _is_at_or_below(sample.body_b, finger_root):
            candidate = _matching_root(sample.body_a, self._object_roots)
        else:
            return None
        return candidate if candidate in self._graspable_ids else None

    def _held_candidate(self, samples: Sequence[ContactSample]) -> str | None:
        left_strength: dict[str, float] = {}
        right_strength: dict[str, float] = {}
        for sample in samples:
            # PhysX may emit zero-impulse proximity pairs inside its contact
            # margin. They are useful collision telemetry, but they do not
            # constitute force-bearing finger contact or a verified hold.
            if sample.impulse_ns < MINIMUM_HOLD_IMPULSE_NS:
                continue
            left = self._finger_object(sample, self._left_finger_root)
            right = self._finger_object(sample, self._right_finger_root)
            if left is not None:
                left_strength[left] = left_strength.get(left, 0.0) + sample.impulse_ns
            if right is not None:
                right_strength[right] = right_strength.get(right, 0.0) + sample.impulse_ns
        common = left_strength.keys() & right_strength.keys()
        if not common:
            return None
        return max(
            common,
            key=lambda object_id: (left_strength[object_id] + right_strength[object_id], object_id),
        )

    def _update_hold(self, candidate: str | None) -> str | None:
        if self._held_object_id is not None:
            if candidate == self._held_object_id:
                self._lost_frames = 0
            else:
                self._lost_frames += 1
                if self._lost_frames >= self._release_contact_frames:
                    self._held_object_id = None
                    self._candidate_id = candidate
                    self._candidate_frames = 1 if candidate is not None else 0
                    self._lost_frames = 0
            return self._held_object_id

        if candidate is None:
            self._candidate_id = None
            self._candidate_frames = 0
        elif candidate == self._candidate_id:
            self._candidate_frames += 1
        else:
            self._candidate_id = candidate
            self._candidate_frames = 1
        if self._candidate_frames >= self._grasp_contact_frames:
            self._held_object_id = self._candidate_id
            self._lost_frames = 0
        return self._held_object_id

    def _collision_events(self, samples: Sequence[ContactSample]) -> tuple[CollisionEvent, ...]:
        impulses: dict[tuple[str, str, bool], float] = {}
        for sample in samples:
            a_is_robot = _is_at_or_below(sample.body_a, self._robot_root)
            b_is_robot = _is_at_or_below(sample.body_b, self._robot_root)
            if a_is_robot == b_is_robot:
                continue
            robot_body = sample.body_a if a_is_robot else sample.body_b
            external_body = sample.body_b if a_is_robot else sample.body_a
            object_id = _matching_root(external_body, self._object_roots)
            is_finger = _is_at_or_below(robot_body, self._left_finger_root) or _is_at_or_below(
                robot_body, self._right_finger_root
            )
            mount_support = any(
                _is_at_or_below(robot_body, mount) and _is_at_or_below(external_body, support)
                for mount, support in self._stationary_mount_contacts
            )
            allowed = mount_support or (is_finger and object_id in self._graspable_ids)
            key = (robot_body, external_body, allowed)
            impulses[key] = impulses.get(key, 0.0) + sample.impulse_ns
        return tuple(
            CollisionEvent(body_a=key[0], body_b=key[1], impulse_ns=impulse, allowed=key[2])
            for key, impulse in sorted(impulses.items())
        )

    def update(self, samples: Sequence[ContactSample]) -> ContactTelemetry:
        candidate = self._held_candidate(samples)
        return ContactTelemetry(
            held_object_id=self._update_hold(candidate),
            collisions=self._collision_events(samples),
            raw_contact_count=len(samples),
        )


class IsaacContactCollector:
    """Normalize raw records from native Isaac contact sensors."""

    def __init__(
        self,
        sensors: Mapping[str, Any],
        decode_body_path: Callable[[int], str],
        interpreter: ContactInterpreter,
    ) -> None:
        self._sensors = dict(sensors)
        self._decode_body_path = decode_body_path
        self._interpreter = interpreter

    def reset(self) -> None:
        self._interpreter.reset()

    def _path(self, value: Any) -> str:
        return value if isinstance(value, str) else self._decode_body_path(int(value))

    @staticmethod
    def _impulse(value: Any) -> tuple[float, float, float]:
        if isinstance(value, Mapping):
            return (float(value["x"]), float(value["y"]), float(value["z"]))
        return tuple(float(component) for component in value)

    def capture(self) -> ContactTelemetry:
        samples: list[ContactSample] = []
        seen: set[
            tuple[str, str, tuple[float, float, float], tuple[float, float, float] | None]
        ] = set()
        for sensor in self._sensors.values():
            for record in sensor.get_raw_data():
                body_a = self._path(record["body0"])
                body_b = self._path(record["body1"])
                impulse = self._impulse(record["impulse"])
                position = self._impulse(record["position"]) if "position" in record else None
                key = (body_a, body_b, impulse, position)
                reverse_key = (body_b, body_a, impulse, position)
                if key in seen or reverse_key in seen:
                    continue
                seen.add(key)
                samples.append(ContactSample(body_a, body_b, impulse))
        return self._interpreter.update(samples)

from unittest.mock import Mock

from gap_isaac.contact_telemetry import (
    FRANKA_GROUND_MOUNT_CONTACTS,
    ContactInterpreter,
    ContactSample,
    IsaacContactCollector,
)

OBJECT_ROOTS = {
    "alphabet_soup": "/World/Task/alphabet_soup",
    "cream_cheese": "/World/Task/cream_cheese",
    "basket": "/World/Task/basket",
}
LEFT = "/World/Franka/panda_leftfinger"
RIGHT = "/World/Franka/panda_rightfinger"


def interpreter(grasp_frames: int = 3, release_frames: int = 2) -> ContactInterpreter:
    return ContactInterpreter(
        object_roots=OBJECT_ROOTS,
        container_id="basket",
        robot_root="/World/Franka",
        left_finger_root=LEFT,
        right_finger_root=RIGHT,
        grasp_contact_frames=grasp_frames,
        release_contact_frames=release_frames,
    )


def grip_samples(object_id: str = "alphabet_soup") -> tuple[ContactSample, ...]:
    root = OBJECT_ROOTS[object_id]
    return (
        ContactSample(f"{LEFT}/collision", root, (0.2, 0.0, 0.0)),
        ContactSample(f"{RIGHT}/collision", root, (-0.2, 0.0, 0.0)),
    )


def test_two_finger_hold_is_debounced_and_release_has_hysteresis() -> None:
    detector = interpreter()

    assert detector.update(grip_samples()).held_object_id is None
    assert detector.update(grip_samples()).held_object_id is None
    assert detector.update(grip_samples()).held_object_id == "alphabet_soup"
    assert detector.update(()).held_object_id == "alphabet_soup"
    assert detector.update(()).held_object_id is None


def test_one_finger_contact_does_not_count_as_a_hold() -> None:
    detector = interpreter(grasp_frames=1)

    telemetry = detector.update(grip_samples()[:1])

    assert telemetry.held_object_id is None
    assert telemetry.collisions[0].allowed


def test_zero_impulse_proximity_pairs_do_not_count_as_a_hold() -> None:
    detector = interpreter(grasp_frames=1)
    root = OBJECT_ROOTS["alphabet_soup"]
    samples = (
        ContactSample(f"{LEFT}/collision", root, (0.0, 0.0, 0.0)),
        ContactSample(f"{RIGHT}/collision", root, (0.0, 0.0, 0.0)),
    )

    telemetry = detector.update(samples)

    assert telemetry.held_object_id is None
    assert telemetry.raw_contact_count == 2


def test_robot_collision_is_disallowed_but_grasp_contact_is_allowed() -> None:
    samples = grip_samples() + (
        ContactSample(
            "/World/Franka/panda_link4/collisions",
            "/World/Task/TableTop",
            (0.0, 0.0, 2.0),
        ),
    )

    telemetry = interpreter(grasp_frames=1).update(samples)

    assert telemetry.held_object_id == "alphabet_soup"
    assert len(telemetry.collisions) == 3
    assert sum(not event.allowed for event in telemetry.collisions) == 1


def test_container_contact_is_not_a_valid_hold() -> None:
    telemetry = interpreter(grasp_frames=1).update(grip_samples("basket"))

    assert telemetry.held_object_id is None
    assert all(not event.allowed for event in telemetry.collisions)


def test_ground_support_exception_is_only_for_the_fixed_mount() -> None:
    detector = ContactInterpreter(
        object_roots=OBJECT_ROOTS,
        container_id="basket",
        robot_root="/World/Franka",
        left_finger_root=LEFT,
        right_finger_root=RIGHT,
        grasp_contact_frames=1,
        release_contact_frames=1,
        stationary_mount_contacts=FRANKA_GROUND_MOUNT_CONTACTS,
    )
    samples = tuple(
        ContactSample(robot, external, (0.0, 0.0, 2.0))
        for robot, external in (
            ("/World/Franka/panda_link0/collision", "/World/GroundPlane"),
            ("/World/Franka/panda_link1", "/World/GroundPlane"),
            (LEFT, "/World/GroundPlane"),
            ("/World/Franka/panda_link0", "/World/Task/basket/base"),
            ("/World/Franka/panda_link0_extra", "/World/GroundPlane"),
        )
    )
    telemetry = detector.update(samples)
    assert telemetry.held_object_id is None
    assert sum(event.allowed for event in telemetry.collisions) == 1
    assert next(event for event in telemetry.collisions if event.allowed).body_a.endswith(
        "panda_link0/collision"
    )


def test_collector_normalizes_paths_impulses_and_duplicate_sensor_records() -> None:
    record = {
        "body0": 1,
        "body1": 2,
        "impulse": {"x": 0.2, "y": 0.0, "z": 0.0},
    }
    left_sensor = Mock()
    right_sensor = Mock()
    left_sensor.get_raw_data.return_value = [record]
    right_sensor.get_raw_data.return_value = [record]
    paths = {1: f"{LEFT}/collision", 2: OBJECT_ROOTS["alphabet_soup"]}
    collector = IsaacContactCollector(
        {"left": left_sensor, "right": right_sensor},
        lambda body_id: paths[body_id],
        interpreter(),
    )

    telemetry = collector.capture()

    assert telemetry.raw_contact_count == 1
    assert len(telemetry.collisions) == 1

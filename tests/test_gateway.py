from fastapi.testclient import TestClient

from gap_isaac.fake_backend import FakeBackend
from gap_isaac.gateway import create_app


def test_gateway_requires_authentication() -> None:
    client = TestClient(create_app(FakeBackend(), "secret"))
    assert client.get("/v1/health").status_code == 401
    response = client.get("/v1/health", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json()["protocol_version"] == "1.0"

    observation = client.get(
        "/v1/observations/latest", headers={"Authorization": "Bearer secret"}
    ).json()
    asset = client.get(observation["rgb_url"], headers={"Authorization": "Bearer secret"})
    assert asset.status_code == 200
    assert asset.content == b"fake-rgb"


def test_gateway_rejects_motion_when_disabled_but_allows_stop() -> None:
    backend = FakeBackend(motion_enabled=False)
    client = TestClient(create_app(backend, "secret"))
    headers = {"Authorization": "Bearer secret"}
    trajectory = {
        "protocol_version": "1.0",
        "command_id": "move-1",
        "joint_names": ["panda_joint1"],
        "points": [{"time_from_start_s": 1.0, "positions_rad": [0.0]}],
        "timeout_s": 10,
    }
    assert (
        client.post("/v1/commands/trajectory", headers=headers, json=trajectory).status_code == 409
    )

    stop = client.post(
        "/v1/commands/stop",
        headers=headers,
        json={"protocol_version": "1.0", "command_id": "stop-1"},
    )
    assert stop.status_code == 200
    assert backend.stopped is True


def test_gateway_executes_motion_only_when_enabled() -> None:
    backend = FakeBackend(motion_enabled=True)
    client = TestClient(create_app(backend, "secret"))
    response = client.post(
        "/v1/commands/gripper",
        headers={"Authorization": "Bearer secret"},
        json={
            "protocol_version": "1.0",
            "command_id": "grip-1",
            "action": "close",
            "force_n": 20,
        },
    )
    assert response.status_code == 200
    assert backend.commands == ["grip-1"]

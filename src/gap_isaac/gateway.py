"""Authenticated connector gateway and simulator-backend contract."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import Response

from gap_isaac.protocol import (
    CommandResult,
    ExecuteTrajectoryRequest,
    GripperRequest,
    HealthResponse,
    ObservationDescriptor,
    StopRequest,
)


@dataclass(frozen=True)
class AssetPayload:
    body: bytes
    media_type: str = "application/octet-stream"


class ConnectorBackend(Protocol):
    def health(self) -> HealthResponse: ...

    def observation(self) -> ObservationDescriptor: ...

    def asset(self, sequence: int, kind: str) -> AssetPayload: ...

    def execute_trajectory(self, request: ExecuteTrajectoryRequest) -> CommandResult: ...

    def control_gripper(self, request: GripperRequest) -> CommandResult: ...

    def stop(self, request: StopRequest) -> CommandResult: ...


def create_app(backend: ConnectorBackend, token: str) -> FastAPI:
    """Create the Windows-side API without importing Isaac or ROS packages."""

    if not token:
        raise ValueError("connector token must not be empty")

    app = FastAPI(
        title="Graph-as-Policy Isaac Connector",
        version="1.0",
        docs_url=None,
        redoc_url=None,
    )

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        scheme, _, credential = (authorization or "").partition(" ")
        valid = scheme.lower() == "bearer" and secrets.compare_digest(credential, token)
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid connector credential",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # observation access requires the same credential as command access
    authenticated = [Depends(authenticate)]

    @app.get("/v1/health", response_model=HealthResponse, dependencies=authenticated)
    def health() -> HealthResponse:
        return backend.health()

    @app.get(
        "/v1/observations/latest",
        response_model=ObservationDescriptor,
        dependencies=authenticated,
    )
    def observation() -> ObservationDescriptor:
        return backend.observation()

    @app.get("/v1/assets/{sequence}/{kind}", dependencies=authenticated)
    def asset(sequence: int, kind: str) -> Response:
        try:
            payload = backend.asset(sequence, kind)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=404, detail="asset not found") from error
        return Response(content=payload.body, media_type=payload.media_type)

    @app.post(
        "/v1/commands/trajectory",
        response_model=CommandResult,
        dependencies=authenticated,
    )
    def execute_trajectory(request: ExecuteTrajectoryRequest) -> CommandResult:
        health_state = backend.health()
        if not health_state.motion_enabled:
            raise HTTPException(status_code=409, detail="motion is disabled")
        return backend.execute_trajectory(request)

    @app.post(
        "/v1/commands/gripper",
        response_model=CommandResult,
        dependencies=authenticated,
    )
    def control_gripper(request: GripperRequest) -> CommandResult:
        health_state = backend.health()
        if not health_state.motion_enabled:
            raise HTTPException(status_code=409, detail="motion is disabled")
        return backend.control_gripper(request)

    @app.post("/v1/commands/stop", response_model=CommandResult, dependencies=authenticated)
    def stop(request: StopRequest) -> CommandResult:
        return backend.stop(request)

    return app

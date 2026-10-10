"""Client for the native Windows Isaac connector gateway."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from gap_isaac.protocol import (
    CommandResult,
    ExecuteTrajectoryRequest,
    GripperRequest,
    HealthResponse,
    ObservationDescriptor,
    StopRequest,
)


@dataclass
class ConnectorClient:
    base_url: str
    token: str
    timeout_s: float = 30.0

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _request(self, method: str, path: str, payload: dict | None = None) -> httpx.Response:
        response = httpx.request(
            method,
            f"{self.base_url.rstrip('/')}{path}",
            headers=self._headers,
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response

    def health(self) -> HealthResponse:
        return HealthResponse.model_validate(self._request("GET", "/v1/health").json())

    def observation(self) -> ObservationDescriptor:
        response = self._request("GET", "/v1/observations/latest")
        return ObservationDescriptor.model_validate(response.json())

    def execute_trajectory(self, request: ExecuteTrajectoryRequest) -> CommandResult:
        response = self._request("POST", "/v1/commands/trajectory", request.model_dump(mode="json"))
        return CommandResult.model_validate(response.json())

    def control_gripper(self, request: GripperRequest) -> CommandResult:
        response = self._request("POST", "/v1/commands/gripper", request.model_dump(mode="json"))
        return CommandResult.model_validate(response.json())

    def stop(self, request: StopRequest) -> CommandResult:
        response = self._request("POST", "/v1/commands/stop", request.model_dump(mode="json"))
        return CommandResult.model_validate(response.json())

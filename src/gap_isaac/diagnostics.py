"""Non-mutating host diagnostics for Windows and WSL development."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else None


def run_checks() -> list[Check]:
    checks: list[Check] = []
    system = platform.system()
    in_wsl = bool(os.environ.get("WSL_DISTRO_NAME"))
    checks.append(Check("host", "ok", f"{system}; WSL={in_wsl}"))

    gpu_index = os.environ.get("GAP_ISAAC_GPU_INDEX", "0")
    nvidia = _version(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    checks.append(
        Check("nvidia", "ok" if nvidia else "missing", nvidia or "nvidia-smi unavailable")
    )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_visibility = visible_devices == gpu_index
    checks.append(
        Check(
            "gpu_isolation",
            "ok" if expected_visibility else "warning",
            visible_devices or f"set CUDA_VISIBLE_DEVICES={gpu_index}",
        )
    )

    if in_wsl:
        connector_url = os.environ.get("GAP_ISAAC_CONNECTOR_URL")
        connector_token = os.environ.get("GAP_ISAAC_CONNECTOR_TOKEN")
        checks.append(
            Check(
                "connector_url",
                "ok" if connector_url else "missing",
                connector_url or "GAP_ISAAC_CONNECTOR_URL not set",
            )
        )
        checks.append(
            Check(
                "connector_token",
                "ok" if connector_token else "missing",
                "configured" if connector_token else "GAP_ISAAC_CONNECTOR_TOKEN not set",
            )
        )
    else:
        isaac_path = os.environ.get("ISAAC_SIM_PATH")
        checks.append(
            Check(
                "isaac_sim",
                "ok" if isaac_path else "warning",
                isaac_path or "ISAAC_SIM_PATH not set",
            )
        )

    token = os.environ.get("HF_TOKEN")
    checks.append(
        Check(
            "model_access",
            "ok" if token else "warning",
            "configured" if token else "HF_TOKEN not set",
        )
    )

    skill_path = os.environ.get("GAP_SKILLS_PATH")
    checks.append(
        Check(
            "skill_registry",
            "ok" if skill_path else "warning",
            skill_path or "GAP_SKILLS_PATH not set; sibling discovery may still work",
        )
    )
    return checks

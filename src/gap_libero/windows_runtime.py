"""Prepare pinned robosuite Windows paths without importing the simulator."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path

PRIVATE_MACROS = (
    "# keep windows rendering on glfw rather than linux egl\n"
    "MUJOCO_GPU_RENDERING = False\n"
)


def prepare_windows_runtime() -> dict:
    if os.name != "nt":
        return {"ok": True, "windows_changes_required": False}
    if Path(sys.prefix).name != ".venv-libero" or sys.prefix == sys.base_prefix:
        raise RuntimeError("Windows runtime preparation requires the isolated .venv-libero")
    robosuite = importlib.metadata.distribution("robosuite")
    mujoco = importlib.metadata.distribution("mujoco")
    if robosuite.version != "1.4.1" or mujoco.version != "2.3.7":
        raise RuntimeError("Windows workaround only supports the pinned robosuite and mujoco")
    root = Path(robosuite.locate_file("robosuite")).resolve()
    source = Path(mujoco.locate_file("mujoco/mujoco.dll")).resolve()
    destination = root / "utils" / "mujoco.dll"
    private = root / "macros_private.py"
    environment = Path(sys.prefix).resolve()
    if not all(path.is_relative_to(environment) for path in (source, destination, private)):
        raise RuntimeError("refusing runtime changes outside the isolated LIBERO environment")
    if not source.is_file():
        raise FileNotFoundError("installed MuJoCo Windows DLL was not found")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError("existing robosuite MuJoCo DLL differs and was not overwritten")
    else:
        shutil.copyfile(source, destination)
    if private.exists():
        if private.read_text(encoding="utf-8") != PRIVATE_MACROS:
            raise RuntimeError("existing robosuite private macros were not overwritten")
    else:
        with private.open("x", encoding="utf-8") as handle:
            handle.write(PRIVATE_MACROS)
    return {"ok": True, "windows_changes_required": True, "dll_sha256": expected_hash,
            "render_backend": "glfw", "simulator_started": False}

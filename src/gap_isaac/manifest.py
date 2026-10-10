"""Immutable experiment provenance manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_provenance(root: Path) -> dict:
    commit = _command(["git", "rev-parse", "HEAD"], root)
    status = _command(["git", "status", "--porcelain"], root)
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def build_manifest(root: Path, files: Iterable[Path], metadata: dict | None = None) -> dict:
    resolved_root = root.resolve()
    hashes: dict[str, str] = {}
    for file_path in sorted((path.resolve() for path in files), key=str):
        try:
            label = str(file_path.relative_to(resolved_root))
        except ValueError:
            label = str(file_path)
        hashes[label.replace("\\", "/")] = sha256_file(file_path)

    gpu_index = os.environ.get("GAP_ISAAC_GPU_INDEX", "0")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": git_provenance(resolved_root),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable_name": Path(sys.executable).name,
            "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
            "gpu": _command(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "files": hashes,
        "metadata": metadata or {},
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

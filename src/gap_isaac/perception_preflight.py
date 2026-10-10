"""Metadata-only perception readiness check; never imports model packages.

Runnable with plain Python 3.12 before installing the project or upstream stack.
No network calls, downloads, credential values, CUDA initialization, or weights.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import ipaddress
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

SKILLS_REVISION = "752dfbd96968d4d08379f8111765efbf6457800b"
CORE_REVISION = "c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea"
SAM3_REVISION = "b26a5f330e05d321afb39d01d3d4881f258f65ff"
PACKAGES = (
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "transformers",
    "pillow",
    "opencv-python-headless",
    "open3d",
    "scikit-learn",
    "sam3",
    "decord2",
    "pycocotools",
    "einops",
    "psutil",
    "huggingface-hub",
    "httpx",
    "graph-as-policy-core",
    "graph-as-policy",
    "triton-windows",
)


def is_loopback_endpoint(value: str) -> bool:
    """Accept explicit loopback API bases only; never resolve DNS or make a request."""
    try:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.path.rstrip("/") != "/v1"
        ):
            return False
        # Accessing port also validates malformed/out-of-range port numbers.
        if url.port is not None and url.port == 0:
            return False
        return url.hostname == "localhost" or ipaddress.ip_address(url.hostname).is_loopback
    except (ValueError, TypeError):
        return False


def _revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_preflight(root: Path, skills_root: Path | None = None) -> dict:
    """Report missing preparation, not proof that native inference works."""
    upstream = root / "cache/upstream"
    skills_root = skills_root or Path(
        os.environ.get("GAP_SKILLS_PATH") or upstream / "open-robot-skills"
    )
    core_root = skills_root.parent / "graph-as-policy"
    package_versions = {}
    for name in PACKAGES:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    checks = {
        "python_312": sys.version_info[:2] == (3, 12),
        "isolated_environment": sys.prefix != sys.base_prefix,
        "windows_x64": platform.system() == "Windows" and sys.maxsize > 2**32,
        "packages_present": all(package_versions.values()),
        "skills_pinned": _revision(skills_root) == SKILLS_REVISION,
        "core_pinned": _revision(core_root) == CORE_REVISION,
        "skill_script_present": (
            skills_root / "skills/perceiving-objects/scripts/perceive_dino_vlm.py"
        ).is_file(),
    }
    credentials = {
        "hf_token_environment": bool(os.environ.get("HF_TOKEN")),
        "alternate_vlm_key_environment": bool(os.environ.get("GAP_VLM_API_KEY")),
    }
    local_vlm = {
        "explicit_loopback_endpoint": is_loopback_endpoint(os.environ.get("GAP_VLM_BASE_URL", "")),
        "explicit_model": bool(os.environ.get("GAP_VLM_MODEL", "").strip()),
        # Upstream names its OpenAI-compatible transport "openrouter" even locally.
        "compatible_transport": os.environ.get("GAP_VLM_PROVIDER") == "openrouter",
    }
    return {
        "schema_version": 1,
        "check_kind": "metadata-only-no-inference",
        "inference_policy": "local-only",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "checks": checks,
        "packages": package_versions,
        "missing_packages": [name for name, version in package_versions.items() if version is None],
        "credentials_presence_only": credentials,
        "credentials_validated": False,
        "local_vlm_configuration": local_vlm,
        "local_vlm_configuration_ready": all(local_vlm.values()),
        "local_server_validated": False,
        "graph_generator_configuration_validated": False,
        "sam3_gated_access_validated": False,
        "weights_validated": False,
        "models_loaded": False,
        "network_requests": 0,
        "hf_cli_available": shutil.which("hf") is not None,
        "ready_for_import_probe": all(checks.values()),
        "native_inference_validated": False,
        "wsl_required": "not-established-native-import-probe-pending",
        "upstream": {
            "skills_root": str(skills_root),
            "skills_revision": SKILLS_REVISION,
            "core_root": str(core_root),
            "core_revision": CORE_REVISION,
            "sam3_revision": SAM3_REVISION,
        },
        "notes": [
            "Package metadata presence does not prove dependency, CUDA, or import compatibility.",
            "SAM3 image-loader imports Triton transitively; disabling video does not avoid it.",
            "Native Windows candidate uses triton-windows with a matching PyTorch minor version.",
            "Cached HF authentication may exist without HF_TOKEN; never print stored tokens.",
            "Local VLM configuration requires explicit loopback URL and model; no cloud fallback.",
            "This diagnostic does not enforce upstream networking or validate the local server.",
            "Do not load models while Isaac is running. Test imports before weights or inference.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--skills-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(run_preflight(args.root.resolve(), args.skills_root), indent=2, sort_keys=True)
    )
    # Diagnostic report, not an installation or inference command.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

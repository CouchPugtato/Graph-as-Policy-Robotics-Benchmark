"""Noninteractive configuration before importing LIBERO."""

import os
from pathlib import Path


def prepare_runtime_config():
    default = Path(__file__).resolve().parents[2] / "cache/libero-config"
    directory = Path(os.environ.get("LIBERO_CONFIG_PATH", str(default))).resolve()
    if not (directory / "config.yaml").is_file():
        raise RuntimeError(
            f"LIBERO config missing at {directory / 'config.yaml'}; "
            "run scripts/windows/prepare-libero.ps1 first or set LIBERO_CONFIG_PATH "
            "to a prepared config directory")
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    return directory

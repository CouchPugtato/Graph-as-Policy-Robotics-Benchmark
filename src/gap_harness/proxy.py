"""Start or stop the pinned localhost Codex compatibility server."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from gap_harness.config import ROOT

PACKAGE = "openai-api-server-via-codex==0.2.2"


def stop_windows(environment):
    pid_path = ROOT / "cache/codex-proxy/run/server-127.0.0.1-18080.pid"
    if not pid_path.is_file():
        print("Codex proxy is not running under this project")
        return 0
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    if pid <= 0:
        raise ValueError("invalid project proxy process record")
    # verify process identity before stopping only this daemon and its direct worker
    script = """
$ErrorActionPreference = 'Stop'
$ownedPid = [int]$env:GAP_OWNED_PROXY_PID
$supervisor = Get-CimInstance Win32_Process -Filter "ProcessId = $ownedPid"
if ($null -eq $supervisor) { exit 0 }
if ($supervisor.Name -ne 'openai-api-server-via-codex.exe') {
    throw 'Project proxy process record belongs to a different application'
}
$children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $ownedPid" |
    Where-Object { $_.Name -eq $supervisor.Name -and
                   $_.ExecutablePath -eq $supervisor.ExecutablePath })
Stop-Process -Id $ownedPid -ErrorAction Stop
foreach ($child in $children) {
    Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
}
"""
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        cwd=ROOT, env={**environment, "GAP_OWNED_PROXY_PID": str(pid)},
        check=False, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode == 0:
        pid_path.unlink(missing_ok=True)
        print("Stopped project Codex proxy")
    return result.returncode


def command(mode):
    executable = Path(sys.executable).parent / ("uvx.exe" if os.name == "nt" else "uvx")
    uvx = str(executable) if executable.is_file() else shutil.which("uvx")
    if not uvx:
        raise RuntimeError("install harness tools with python -m pip install -e .[harness-tools]")
    return [uvx, "--python", sys.executable, PACKAGE, mode, "--config",
            str(ROOT / "harness/codex-proxy.toml")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["start", "status", "stop"])
    args = parser.parse_args()
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN",
                                  "HUGGING_FACE_HUB_TOKEN", "GAP_CODEX_PROXY_KEY"}}
    environment["UV_PYTHON_DOWNLOADS"] = "never"
    try:
        if os.name == "nt" and args.mode == "stop":
            returncode = stop_windows(environment)
        else:
            result = subprocess.run(command(args.mode), cwd=ROOT, env=environment,
                                    check=False, timeout=30)
            returncode = result.returncode
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Codex proxy stopped: {type(error).__name__}: {error}\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()

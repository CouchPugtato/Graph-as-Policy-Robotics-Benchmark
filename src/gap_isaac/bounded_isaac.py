"""Watch one owned diagnostic process tree; stop it at a wall-clock deadline."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path


def powershell_literal(value: str) -> str:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError("launcher arguments cannot contain control characters")
    return "'" + value.replace("'", "''") + "'"


def diagnostic_command(launcher: Path, script: Path, arguments: list[str]) -> list[str]:
    # Native Windows Python launcher is a batch file. A hidden, dedicated PowerShell
    # process preserves its environment setup without string-building cmd.exe commands.
    invocation = "& " + " ".join(powershell_literal(str(value))
                                  for value in (launcher, script, *arguments))
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
            "-Command", invocation + "; exit $LASTEXITCODE"]


def stop_owned_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    # Exact PID created by this watchdog only; never kill by process name.
    subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                   capture_output=True, timeout=10, check=False)
    process.wait(timeout=5)


def run_bounded(command: list[str], result_file: Path, timeout_s: float) -> int:
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    log_file = result_file.with_suffix(".log")
    progress_file = result_file.with_suffix(".progress.json")
    watchdog_file = result_file.with_suffix(".watchdog.json")
    started = time.monotonic()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    timed_out = False
    interrupted = False
    cleanup_error = None

    def forward_output() -> None:
        with log_file.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

    reader = threading.Thread(target=forward_output, daemon=True)
    reader.start()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(f"Feasibility deadline exceeded ({timeout_s:g}s); stopping owned process tree.",
              flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Diagnostic interrupted; stopping owned process tree.", flush=True)
    finally:
        if process.poll() is None:
            try:
                stop_owned_tree(process)
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_error = str(error)
        reader.join(timeout=1)
        progress = None
        if progress_file.is_file():
            with suppress(OSError, ValueError):
                progress = json.loads(progress_file.read_text(encoding="utf-8"))
        receipt = {
            "timed_out": timed_out, "interrupted": interrupted,
            "timeout_s": timeout_s, "elapsed_wall_s": round(time.monotonic() - started, 3),
            "owned_pid": process.pid, "process_exit_code": process.poll(),
            "cleanup_error": cleanup_error, "last_progress": progress,
            "log": str(log_file), "result": str(result_file),
        }
        watchdog_file.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(f"Saved watchdog diagnostic: {watchdog_file}", flush=True)
        if timed_out or interrupted or cleanup_error:
            # Preserve any partial or completed result, including shutdown-hang evidence.
            if not result_file.exists():
                result_file.write_text(json.dumps({
                    "ok": False, "error_type": "DiagnosticTimeout" if timed_out
                    else "DiagnosticInterrupted", "error": cleanup_error or
                    "Bounded diagnostic stopped before completion", "last_progress": progress,
                    "commands_applied": 0, "graph_motion_ready": False,
                }, indent=2), encoding="utf-8")
            print(json.dumps(receipt, indent=2), flush=True)
    if timed_out or interrupted or cleanup_error:
        return 1
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--target-geometry", type=Path, required=True)
    parser.add_argument("--basket-geometry", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=90)
    args = parser.parse_args()
    if not 15 <= args.timeout_s <= 300:
        parser.error("timeout must be between 15 and 300 seconds")
    arguments = ["--stage", str(args.stage), "--target-geometry", str(args.target_geometry),
                 "--basket-geometry", str(args.basket_geometry),
                 "--result-file", str(args.result_file)]
    return run_bounded(diagnostic_command(args.launcher, args.script, arguments),
                       args.result_file, args.timeout_s)


if __name__ == "__main__":
    raise SystemExit(main())

import json
import sys

import pytest

from gap_isaac.bounded_isaac import diagnostic_command, powershell_literal, run_bounded
from gap_isaac.diagnostic_progress import DiagnosticProgress


def test_progress_marker_is_saved_and_printed(tmp_path, capsys):
    path = tmp_path / "check.progress.json"
    progress = DiagnosticProgress(path)
    progress.mark("ik_start", waypoint="grasp")
    payload = json.loads(path.read_text())
    assert payload["stage"] == "ik_start"
    assert payload["details"] == {"waypoint": "grasp"}
    assert payload["commands_applied"] == 0
    assert "ik_start" in capsys.readouterr().out
    assert not path.with_suffix(".tmp").exists()
    progress.mark("ik_done", solved=True)
    assert json.loads(path.read_text())["stage"] == "ik_done"


def test_launcher_quotes_literal_arguments_and_hides_window(tmp_path):
    command = diagnostic_command(tmp_path / "Isaac path/python.bat",
                                 tmp_path / "runner.py", ["--stage", "a'b & c.usd"])
    assert command[:6] == ["powershell.exe", "-NoProfile", "-NonInteractive",
                           "-WindowStyle", "Hidden", "-Command"]
    assert "'a''b & c.usd'" in command[-1]
    assert command[-1].endswith("exit $LASTEXITCODE")
    with pytest.raises(ValueError):
        powershell_literal("bad\nargument")


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows process ownership test")
def test_completed_tiny_process_is_logged_without_isaac(tmp_path):
    result = tmp_path / "finished.json"
    code = run_bounded([sys.executable, "-c", "print('tiny watchdog test')"], result, 5)
    assert code == 0
    receipt = json.loads(result.with_suffix(".watchdog.json").read_text())
    assert receipt["timed_out"] is False
    assert receipt["process_exit_code"] == 0
    assert "tiny watchdog test" in result.with_suffix(".log").read_text()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows process ownership test")
def test_deadline_stops_only_owned_tiny_process_and_fails(tmp_path):
    result = tmp_path / "timeout.json"
    DiagnosticProgress(result.with_suffix(".progress.json")).mark("synthetic_stall")
    code = run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], result, .2)
    assert code == 1
    receipt = json.loads(result.with_suffix(".watchdog.json").read_text())
    assert receipt["timed_out"] is True
    assert receipt["cleanup_error"] is None
    assert receipt["process_exit_code"] is not None
    assert receipt["last_progress"]["stage"] == "synthetic_stall"
    diagnostic = json.loads(result.read_text())
    assert diagnostic["ok"] is False
    assert diagnostic["commands_applied"] == 0
    assert diagnostic["graph_motion_ready"] is False


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows process ownership test")
def test_shutdown_timeout_preserves_result_but_launcher_fails(tmp_path):
    result = tmp_path / "partial.json"
    result.write_text('{"ok": true, "synthetic_fixture": true}')
    code = run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], result, .2)
    assert code == 1
    assert json.loads(result.read_text())["synthetic_fixture"] is True
    assert json.loads(result.with_suffix(".watchdog.json").read_text())["timed_out"] is True

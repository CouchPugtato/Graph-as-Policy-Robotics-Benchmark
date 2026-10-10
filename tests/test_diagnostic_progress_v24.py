import json
from pathlib import Path

import pytest

from gap_isaac.diagnostic_progress_v24 import DiagnosticProgress


def sharing_error():
    error = PermissionError("temporary Windows sharing denial")
    error.winerror = 5
    return error


def test_transient_windows_replace_denial_preserves_actual_marker(tmp_path, monkeypatch):
    actual_replace = Path.replace
    attempts = []
    sleeps = []

    def replace(path, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise sharing_error()
        return actual_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr("gap_isaac.diagnostic_progress_v24.time.sleep", sleeps.append)
    progress = DiagnosticProgress(tmp_path / "progress.json")
    progress.mark("ik_done", solved=True)
    saved = json.loads(progress.path.read_text())
    assert saved["stage"] == "ik_done" and saved["details"] == {"solved": True}
    assert saved["commands_applied"] == 0
    assert progress.write_retry_count == 2
    assert sleeps == [.025, .025]


def test_persistent_replace_failure_remains_failure_and_is_bounded(tmp_path, monkeypatch):
    attempts = []
    sleeps = []

    def replace(path, target):
        attempts.append(target)
        raise sharing_error()

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr("gap_isaac.diagnostic_progress_v24.time.sleep", sleeps.append)
    progress = DiagnosticProgress(tmp_path / "progress.json")
    with pytest.raises(PermissionError):
        progress.mark("write_result", ok=True)
    assert len(attempts) == 10 and len(sleeps) == 9
    assert not progress.path.exists()

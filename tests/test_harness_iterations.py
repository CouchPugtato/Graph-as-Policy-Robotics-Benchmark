import json
from types import SimpleNamespace

import pytest

from gap_harness.engine import CandidateRejected, digest, dump
from gap_harness.iterations import iterate


class Rehearsal:
    def __init__(self, directory, scores=(9,), *, rejection_count=0, error=None, partial=False):
        self.session = directory
        self.task = SimpleNamespace(seeds=list(range(10)))
        self.task_sha256 = "test-task"
        self.events = []
        self.progress = SimpleNamespace(event=self.events.append)
        self.scores = scores
        self.rejection_count = rejection_count
        self.error = error
        self.partial = partial
        self.generated = self.evaluated = 0

    def prepare(self):
        self.session.mkdir(parents=True, exist_ok=True)

    def versions(self):
        return sorted(self.session.glob("v[0-9][0-9][0-9][0-9]"))

    def generate(self):
        self.generated += 1
        version = self.session / f"v{len(self.versions()) + 1:04d}"
        version.mkdir()
        if self.error:
            raise self.error
        if self.generated <= self.rejection_count:
            dump(version / "validation.json", {"ok": False})
            raise CandidateRejected("invalid candidate")
        dump(version / "graph.json", {"candidate": self.generated})
        return version

    def evaluate(self, version):
        score = self.scores[min(self.evaluated, len(self.scores) - 1)]
        self.evaluated += 1
        feedback = {
            "successes": score,
            "trials": 9 if self.partial else 10,
            "configured_trials": 10,
            "evaluation_complete": not self.partial,
            "graph_sha256": digest(version / "graph.json"),
            "task_sha256": self.task_sha256,
        }
        dump(version / "feedback.json", feedback)
        return feedback


def test_stops_at_first_ten_of_ten_and_preserves_every_revision(tmp_path):
    harness = Rehearsal(tmp_path / "session", scores=(4, 8, 10))
    result = iterate(harness, 10, 10)
    assert result["accepted"] and result["stop_reason"] == "accepted"
    assert harness.generated == harness.evaluated == result["attempts_this_run"] == 3
    assert len(harness.versions()) == 3
    assert all((version / "feedback.json").exists() for version in harness.versions())


def test_stops_after_ten_without_claiming_success(tmp_path):
    harness = Rehearsal(tmp_path / "session")
    result = iterate(harness, 10, 10)
    assert not result["accepted"]
    assert result["stop_reason"] == "attempt_limit"
    assert result["total_attempts"] == harness.generated == harness.evaluated == 10
    saved = next(harness.session.glob("iteration-run-*.json"))
    assert json.loads(saved.read_text())["stop_reason"] == "attempt_limit"


def test_rejected_graphs_count_toward_attempt_limit(tmp_path):
    harness = Rehearsal(tmp_path / "session", scores=(10,), rejection_count=2)
    result = iterate(harness, 10, 10)
    assert result["accepted"] and result["total_attempts"] == 3
    assert harness.evaluated == 1


def test_ten_rejections_never_start_simulation(tmp_path):
    harness = Rehearsal(tmp_path / "session", rejection_count=10)
    result = iterate(harness, 10, 10)
    assert not result["accepted"] and result["total_attempts"] == 10
    assert harness.evaluated == 0


@pytest.mark.parametrize("error", [ValueError("call budget exhausted"), RuntimeError("HTTP 429")])
def test_usage_and_server_errors_stop_without_retrying(tmp_path, error):
    harness = Rehearsal(tmp_path / "session", error=error)
    with pytest.raises(type(error), match=str(error)):
        iterate(harness, 10, 10)
    assert harness.generated == 1 and harness.evaluated == 0
    assert not (harness.session / ".iteration.lock").exists()


def test_partial_evaluation_cannot_be_accepted_or_refined(tmp_path):
    harness = Rehearsal(tmp_path / "session", scores=(9,), partial=True)
    with pytest.raises(ValueError, match="infrastructure"):
        iterate(harness, 10, 10)
    assert harness.generated == harness.evaluated == 1


def test_resuming_a_finished_session_makes_no_more_calls(tmp_path):
    harness = Rehearsal(tmp_path / "session")
    iterate(harness, 10, 10)
    result = iterate(harness, 10, 10)
    assert result["attempts_this_run"] == 0
    assert result["total_attempts"] == harness.generated == 10


def test_resuming_success_makes_no_more_calls(tmp_path):
    harness = Rehearsal(tmp_path / "session", scores=(10,))
    iterate(harness, 10, 10)
    result = iterate(harness, 10, 10)
    assert result["accepted"] and result["attempts_this_run"] == 0
    assert harness.generated == 1


def test_resume_counts_existing_revisions_in_ten_try_limit(tmp_path):
    harness = Rehearsal(tmp_path / "session")
    iterate(harness, 4, 10)
    result = iterate(harness, 10, 10)
    assert result["attempts_this_run"] == 6
    assert result["total_attempts"] == harness.generated == 10


def test_modified_successful_graph_is_not_reused(tmp_path):
    harness = Rehearsal(tmp_path / "session", scores=(10,))
    iterate(harness, 10, 10)
    (harness.versions()[0] / "graph.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        iterate(harness, 10, 10)
    assert harness.generated == 1


def test_powershell_wrapper_resumes_exhausted_session_without_api_calls(tmp_path):
    import os
    import shutil
    import subprocess
    from pathlib import Path

    from gap_harness.engine import Harness

    root = Path(__file__).resolve().parents[1]
    shell = shutil.which("pwsh")
    if os.name != "nt" or not shell or not (root / ".venv/Scripts/python.exe").is_file():
        pytest.skip("native PowerShell wrapper requires the project Windows environment")
    harness = Harness(
        root / "harness/tasks/libero-packing-local-qwen.json", tmp_path / "session", max_calls=100
    )
    harness.prepare()
    for number in range(1, 11):
        (harness.session / f"v{number:04d}").mkdir()
    process = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-File",
            str(root / "scripts/windows/run-until-success.ps1"),
            "packing",
            "-Session",
            str(harness.session),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=root,
    )
    assert process.returncode == 0, process.stderr
    assert '"attempts_this_run": 0' in process.stdout
    assert '"stop_reason": "attempt_limit"' in process.stdout
    assert not list(harness.session.glob("v*/calls/*/request.json"))

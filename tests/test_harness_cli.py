import json
import sys

import pytest

from gap_harness import cli
from gap_harness.engine import Harness


@pytest.mark.parametrize("option", [
    "--task", "--model", "--reasoning-effort", "--rounds", "--max-subgraphs", "--max-calls",
])
def test_extra_cli_options_are_not_accepted(monkeypatch, option):
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", ["gap-harness", "check", option, "1"])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2


def test_evaluation_session_budgets_are_read_automatically(tmp_path, monkeypatch, capsys):
    task = cli.ROOT / "harness/tasks/libero-packing-local-qwen.json"
    harness = Harness(task, tmp_path / "session", max_calls=48)
    harness.prepare()
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:18080/v1")
    monkeypatch.setattr(sys, "argv", [
        "gap-harness", "check", "--session", str(harness.session), "--max-attempts", "1",
    ])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["task"] == "libero-packing-local-qwen"
    assert report["max_calls"] == 48
    assert report["max_subgraphs"] == 8
    assert report["network_requests"] == 0


def test_normal_iteration_uses_ten_attempts_without_extra_flags(tmp_path, monkeypatch, capsys):
    observed = {}

    def iterate(harness, rounds, attempts, **kwargs):
        observed.update(rounds=rounds, attempts=attempts, calls=harness.max_calls,
                        subgraphs=harness.max_subgraphs)
        return {"accepted": False}

    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:18080/v1")
    monkeypatch.setattr(cli, "iterate", iterate)
    monkeypatch.setattr(sys, "argv", [
        "gap-harness", "iterate", "--scenario", "packing", "--session", str(tmp_path / "new"),
        "--live", "--execute",
    ])
    cli.main()
    assert observed == {"rounds": 10, "attempts": 10, "calls": 100, "subgraphs": 8}
    assert json.loads(capsys.readouterr().out)["accepted"] is False

import io
import json
import threading

import pytest
from pydantic import ValidationError

from gap_harness.cli import failure_summary
from gap_harness.models import Topology
from gap_harness.progress import Progress, session_usage, usage_summary


def test_progress_reports_elapsed_time_on_the_selected_stream():
    stream = io.StringIO()
    times = iter([10, 12.5])
    progress = Progress(stream=stream, clock=lambda: next(times))
    progress.event("coordinator: started")
    assert "2.5s" in stream.getvalue() and "coordinator: started" in stream.getvalue()


def test_waiting_heartbeat_stops_after_success():
    signalled = threading.Event()
    class Stream(io.StringIO):
        def write(self, value):
            if "still waiting" in value:
                signalled.set()
            return super().write(value)
    stream = Stream()
    progress = Progress(stream=stream, heartbeat_s=.01)
    with progress.waiting("coordinator", "call 1/6"):
        assert signalled.wait(timeout=1)
    text = stream.getvalue()
    assert "coordinator: started" in text
    assert "call 1/6" in text and "still waiting" in text
    assert not any(thread.name == "harness-progress" for thread in threading.enumerate())


def test_waiting_thread_stops_after_failure():
    progress = Progress(stream=io.StringIO(), heartbeat_s=.01)
    with pytest.raises(RuntimeError), progress.waiting("skill-agent"):
        raise RuntimeError("test failure")
    assert not any(thread.name == "harness-progress" for thread in threading.enumerate())


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), 61])
def test_heartbeat_does_not_allow_busy_loop_intervals(interval):
    with pytest.raises(ValueError):
        Progress(heartbeat_s=interval)


def test_usage_includes_prior_requests_and_marks_unknown_usage(tmp_path):
    for number, response in enumerate([
            {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            {"usage": {"input_tokens": 4, "output_tokens": 2}}, None]):
        call = tmp_path / "v0001/calls" / str(number)
        call.mkdir(parents=True)
        (call / "request.json").write_text("{}")
        if response:
            (call / "response.json").write_text(json.dumps(response))
    usage = session_usage(tmp_path)
    assert usage == {"attempts": 3, "input_tokens": 14, "output_tokens": 7,
                     "total_tokens": 21, "incomplete_usage_calls": 1}
    assert "calls 3/48" in usage_summary(tmp_path, 48)
    assert "incomplete usage" in usage_summary(tmp_path, 48)


def test_invalid_response_summary_does_not_echo_model_content():
    with pytest.raises(ValidationError) as captured:
        Topology.model_validate_json('"private model response content"')
    message = failure_summary(captured.value)
    assert "structured output validation failed" in message
    assert "private model response content" not in message

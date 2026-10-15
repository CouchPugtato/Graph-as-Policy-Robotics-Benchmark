import json

import numpy as np
import pytest
from PIL import Image

from gap_libero.recording import FrameRecorder
from visualizer import libero_export


def policy():
    return {"schema_version": 1, "graph_id": "recording-test", "start": "grasp__open",
            "metadata": {"subgraph_membership": {"grasp__open": "grasp"}},
            "nodes": [{"id": "grasp__open", "skill": "grasp", "action": "open",
                       "routes": {"succeeded": "DONE", "failed": "ABORT"}}],
            "terminals": {"DONE": "succeeded", "ABORT": "failed"}}


def session_fixture(tmp_path, count=2):
    session = tmp_path / "session"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"task_sha256": "task-hash",
        "task_manifest_path": str(tmp_path / "task.json"), "task": {"seeds": [0]}}))
    for number in range(1, count+1):
        version = session / f"v{number:04d}"
        version.mkdir()
        (version / "graph.json").write_text(json.dumps(policy()))
        digest = libero_export.sha256(version / "graph.json")
        (version / "validation.json").write_text(json.dumps(
            {"ok": True, "graph_sha256": digest}))
        (version / "feedback.json").write_text(json.dumps({"task_sha256": "task-hash",
            "graph_sha256": digest, "trials": 1, "successes": 1,
            "records": [{"seed": 0, "success": True}]}))
    return session


def recording_fixture(directory, graph):
    directory.mkdir()
    pixels = np.zeros((24, 32, 3), dtype=np.uint8)
    recorder = FrameRecorder(directory / "recording")
    for step in range(4):
        pixels[..., 0] = step * 50
        recorder.record({"agentview_image": pixels}, step)
    recorder.close()
    receipt = {"ok": True, "seed": 0, "steps": 3,
        "graph_sha256": libero_export.sha256(graph),
        "graph": {"trace": [{"node_id": "grasp__open", "action": "open",
                              "outcome": "succeeded"}]},
        "trajectory": [{"action": "open", "start_step": 0, "step": 3}]}
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return directory / "receipt.json"


def test_recording_has_real_steps_and_flips_vertical_origin(tmp_path):
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    pixels[0, :, 0] = 255
    recorder = FrameRecorder(tmp_path / "recording", max_frames=1)
    recorder.record({"agentview_image": pixels}, 0)
    with pytest.raises(RuntimeError, match="budget"):
        recorder.record({"agentview_image": pixels}, 1)
    recorder.close()
    with Image.open(tmp_path / "recording/frame-00000.png") as image:
        assert np.asarray(image)[-1, 0, 0] == 255
    index = libero_export.frame_index(tmp_path / "recording")
    assert index["fps"] == 20 and index["frames"][0]["step"] == 0


def test_recording_never_overwrites_files(tmp_path):
    FrameRecorder(tmp_path / "recording")
    with pytest.raises(FileExistsError):
        FrameRecorder(tmp_path / "recording")


def test_history_uses_measured_versions_and_no_manual_presentation_rates(tmp_path):
    session = session_fixture(tmp_path)
    data, _ = libero_export.session_history(session)
    assert [record["iteration"] for record in data["iterations"]] == [1, 2]
    assert [record["successes"] for record in data["iterations"]] == [1, 1]
    assert data["iterations"][1]["changed"] == []


def test_history_uses_retry_receipt_and_preserves_original(tmp_path):
    session = session_fixture(tmp_path, 1)
    version = session / "v0001"
    original = (version / "feedback.json").read_bytes()
    attempt = version / "evaluation-0002"
    attempt.mkdir()
    (attempt / "feedback.json").write_bytes(original)
    output = attempt / "seed-0"
    receipt_path = recording_fixture(output, version / "graph.json")
    data, _ = libero_export.session_history(session)
    assert data["iterations"][0]["receipt"] == json.loads(receipt_path.read_text())
    assert (version / "feedback.json").read_bytes() == original


def test_history_excludes_startup_failures_including_legacy_receipts(tmp_path):
    session = session_fixture(tmp_path, 1)
    path = session / "v0001/feedback.json"
    feedback = json.loads(path.read_text())
    feedback["records"][0].update(graph_bound=False, known_trace=False, seed_bound=False)
    path.write_text(json.dumps(feedback))
    with pytest.raises(ValueError, match="no evaluated"):
        libero_export.session_history(session)


def test_tampered_rate_or_graph_is_rejected(tmp_path):
    session = session_fixture(tmp_path)
    path = session / "v0002/feedback.json"
    feedback = json.loads(path.read_text())
    feedback["successes"] = 0
    path.write_text(json.dumps(feedback))
    with pytest.raises(ValueError, match="measured"):
        libero_export.session_history(session)


def test_timeline_tracks_active_and_completed_nodes():
    receipt = {"graph": {"trace": [{"node_id": "open", "action": "open",
                                     "outcome": "succeeded"},
                                    {"node_id": "lift", "action": "lift",
                                     "outcome": "failed"}]},
               "trajectory": [{"action": "open", "start_step": 0, "step": 3},
                              {"action": "lift", "start_step": 3, "step": 7}]}
    assert libero_export.graph_state(receipt, 2)["active_node"] == "open"
    state = libero_export.graph_state(receipt, 5)
    assert state["completed_nodes"] == ["open"] and state["active_node"] == "lift"
    assert libero_export.graph_state(receipt, 7)["failed_node"] == "lift"
    receipt["trajectory"][1]["action"] = "incorrect"
    with pytest.raises(ValueError, match="timeline"):
        libero_export.graph_state(receipt, 5)


def test_frames_cannot_escape_recording_folder(tmp_path):
    folder = tmp_path / "recording"
    folder.mkdir()
    (folder / "frames.json").write_text(json.dumps({"fps": 20,
        "frames": [{"file": "../secret.png", "step": 0}]}))
    with pytest.raises(ValueError):
        libero_export.frame_index(folder)


def test_existing_matching_footage_is_reused_without_starting_simulation(tmp_path, monkeypatch):
    session = session_fixture(tmp_path, 1)
    data, manifest = libero_export.session_history(session)
    directory = session / "visualizations/old/replay"
    directory.parent.mkdir(parents=True)
    receipt = recording_fixture(directory, session / "v0001/graph.json")
    def forbidden(*args, **kwargs):
        raise AssertionError("simulator must not start")
    monkeypatch.setattr(libero_export.subprocess, "run", forbidden)
    path, started = libero_export.record_final(session, data["iterations"][-1], manifest,
                                               tmp_path / "unused")
    assert path == receipt and not started


def test_export_encodes_real_video_and_provenance(tmp_path):
    session = session_fixture(tmp_path, 1)
    data, _ = libero_export.session_history(session)
    receipt = recording_fixture(tmp_path / "replay", session / "v0001/graph.json")
    output = tmp_path / "preview.mp4"
    manifest = libero_export.render_movie(data, receipt, output, fps=12,
                                          size=(640, 360), recap_s=.1)
    assert output.is_file() and output.stat().st_size > 1000
    assert manifest["native_simulation_fps"] == 20
    assert manifest["fps"] == 12 and manifest["rates"][0]["successes"] == 1


def test_automatic_session_export_uses_real_receipts_without_simulator(tmp_path, monkeypatch):
    session = session_fixture(tmp_path, 1)
    directory = session / "visualizations/existing/replay"
    directory.parent.mkdir(parents=True)
    recording_fixture(directory, session / "v0001/graph.json")
    monkeypatch.setattr(libero_export, "latest_session", lambda: session)
    output = libero_export.export_session()
    provenance = json.loads(output.with_suffix(".manifest.json").read_text())
    assert output.is_file() and not provenance["simulation_started"]
    assert provenance["fps"] == 60
    assert provenance["rates"] == [{"iteration": 1, "successes": 1, "trials": 1}]

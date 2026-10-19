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
    first_node = json.loads(graph.read_text())["nodes"][0]
    receipt = {"ok": True, "seed": 0, "steps": 3,
        "graph_sha256": libero_export.sha256(graph),
        "graph": {"trace": [{"node_id": first_node["id"], "action": first_node["action"],
                              "outcome": "succeeded"}]},
        "trajectory": [{"action": first_node["action"], "start_step": 0, "step": 3}]}
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


@pytest.mark.parametrize("first_step,last_step", [(1, 3), (0, 2)])
def test_incomplete_matching_footage_is_not_reused(tmp_path, first_step, last_step):
    session = session_fixture(tmp_path, 1)
    data, manifest = libero_export.session_history(session)
    directory = session / "visualizations/old/replay"
    directory.parent.mkdir(parents=True)
    recording_fixture(directory, session / "v0001/graph.json")
    index_path = directory / "recording/frames.json"
    index = json.loads(index_path.read_text())
    index["frames"] = [frame for frame in index["frames"]
                       if first_step <= frame["step"] <= last_step]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="runner with --record support"):
        libero_export.record_final(session, data["iterations"][-1], manifest,
                                  tmp_path / "unused")


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


def test_fast_playback_preserves_simulation_duration_and_measured_rate(tmp_path):
    session = session_fixture(tmp_path, 1)
    data, _ = libero_export.session_history(session)
    receipt = recording_fixture(tmp_path / "replay", session / "v0001/graph.json")
    manifest = libero_export.render_movie(data, receipt, tmp_path / "fast.mp4", fps=20,
                                          size=(640, 360), recap_s=.1, playback_speed=2)
    assert manifest["playback_speed"] == 2
    assert manifest["simulation_duration_s"] == .2
    assert manifest["motion_playback_duration_s"] == .1
    assert manifest["rates"][0]["successes"] == 1
    with pytest.raises(ValueError, match="playback speed"):
        libero_export.render_movie(data, receipt, tmp_path / "invalid.mp4", playback_speed=0)


def test_zero_step_actions_get_disclosed_presentation_holds(tmp_path):
    session = session_fixture(tmp_path, 1)
    data, _ = libero_export.session_history(session)
    path = recording_fixture(tmp_path / "replay", session / "v0001/graph.json")
    receipt = json.loads(path.read_text())
    receipt["trajectory"][0]["step"] = 0
    path.write_text(json.dumps(receipt))
    manifest = libero_export.render_movie(data, path, tmp_path / "zero-step.mp4", fps=4,
                                          size=(640, 360), recap_s=.1)
    assert manifest["zero_step_presentation_pause_s"] > 0
    assert manifest["simulation_duration_s"] == .2


def test_automatic_session_export_uses_real_receipts_without_simulator(tmp_path, monkeypatch):
    session = session_fixture(tmp_path, 1)
    directory = session / "visualizations/existing/replay"
    directory.parent.mkdir(parents=True)
    recording_fixture(directory, session / "v0001/graph.json")
    monkeypatch.setattr(libero_export, "latest_session", lambda: session)
    monkeypatch.setattr(libero_export, "ROOT", tmp_path)
    render_movie = libero_export.render_movie
    rendered = []
    def capture_render(*args, **kwargs):
        result = render_movie(*args, **kwargs)
        rendered.append(result)
        return result
    monkeypatch.setattr(libero_export, "render_movie", capture_render)
    output = libero_export.export_session()
    provenance = rendered[-1]
    assert output.is_file()
    assert provenance["fps"] == 60
    assert provenance["playback_speed"] == 2
    assert provenance["recap_seconds_per_revision"] == 5
    assert (output.parent / "policy-run-policy.png").is_file()
    assert provenance["rates"] == [{"iteration": 1, "successes": 1, "trials": 1}]
    assert libero_export.export_session() == output
    assert sorted(path.name for path in output.parent.iterdir()) == [
        "policy-run-policy.png", "policy-run-process.mp4"]
    assert not list(session.glob("visualizations/*/presentation.json"))


@pytest.mark.parametrize("task_id,folder,title", [
    ("libero-grocery-local-vlm", "grocery-order", "grocery-order"),
    ("libero-fulfillment-local-qwen", "fulfillment", "grocery-fulfillment"),
    ("libero-packing-local-qwen", "packing", "two-item-packing"),
    ("libero-popcorn-local-qwen", "popcorn", "popcorn-preparation"),
])
def test_presentation_outputs_are_named_by_task(task_id, folder, title):
    output = libero_export.presentation_output({"task": {"task_id": task_id}})
    assert output.parent.name == folder
    assert output.name == title + "-process.mp4"


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), 31])
def test_invalid_iteration_hold_is_rejected_before_recording(tmp_path, seconds):
    with pytest.raises(ValueError, match="iteration hold"):
        libero_export.export_session(tmp_path / "missing", recap_s=seconds)


def test_movie_cycles_distinct_revision_graphs_and_populates_rates(tmp_path,monkeypatch):
    session = session_fixture(tmp_path,2)
    first_feedback = session / "v0001/feedback.json"
    feedback = json.loads(first_feedback.read_text())
    feedback.update(successes=0,records=[{"seed":0,"success":False}])
    first_feedback.write_text(json.dumps(feedback))
    graph_path = session / "v0002/graph.json"
    graph = json.loads(graph_path.read_text())
    graph["graph_id"] = "second-revision"
    graph["nodes"][0]["action"] = "close"
    graph_path.write_text(json.dumps(graph))
    for name in ("validation.json","feedback.json"):
        path = graph_path.parent / name
        value = json.loads(path.read_text())
        value["graph_sha256"] = libero_export.sha256(graph_path)
        path.write_text(json.dumps(value))
    data,_ = libero_export.session_history(session)
    assert data["iterations"][1]["changed"] == ["grasp__open"]
    assert data["iterations"][0]["policy"]["nodes"][0]["action"] == "open"
    assert data["iterations"][1]["policy"]["nodes"][0]["action"] == "close"
    before = [path.read_bytes() for path in (session / "v0001/graph.json",graph_path)]
    receipt = recording_fixture(tmp_path / "replay",graph_path)
    rendered = []
    render = libero_export.app.render
    def capture(history,index,**kwargs):
        record = history["iterations"][index]
        rates = [(item["successes"],item["trials"])
                 for item in history["iterations"][:index+1]]
        rendered.append((record["iteration"],record["policy"]["nodes"][0]["action"],rates))
        return render(history,index,**kwargs)
    monkeypatch.setattr(libero_export.app,"render",capture)
    output = tmp_path / "revisions.mp4"
    result = libero_export.render_movie(data,receipt,output,fps=4,size=(640,360),recap_s=.1)
    assert rendered[:2] == [(1,"open",[(0,1)]),(2,"close",[(0,1),(1,1)])]
    assert all(item[0:2] == (2,"close") for item in rendered[2:])
    assert result["rates"] == [{"iteration":1,"successes":0,"trials":1},
                                {"iteration":2,"successes":1,"trials":1}]
    assert output.stat().st_size > 1000
    assert before == [path.read_bytes() for path in (session / "v0001/graph.json",graph_path)]


def test_unmeasured_attempts_are_disclosed_without_invented_rates(tmp_path):
    session = session_fixture(tmp_path,3)
    (session / "v0002/feedback.json").unlink()
    data,_ = libero_export.session_history(session)
    assert [record["iteration"] for record in data["iterations"]] == [1,3]
    assert data["skipped_iterations"] == [{"iteration":2,"reason":"not evaluated"}]


def test_normal_false_routing_does_not_mark_successful_loop_exit_failed():
    policy = {"terminals":{"done":"succeeded","abort":"failed"}}
    event = {"node_id":"next_item","action":"select_next_item","outcome":"false",
             "next_node":"check_all_packed"}
    receipt = {"graph":{"trace":[event]},
               "trajectory":[{"action":"select_next_item","start_step":0,"step":0}]}
    state = libero_export.graph_state(receipt,0,final=True,policy=policy)
    assert state["failed_node"] is None
    assert state["completed_nodes"] == ["next_item"]
    assert state["taken_routes"] == {"next_item":"false"}
    event["next_node"] = "abort"
    failed = libero_export.graph_state(receipt,0,final=True,policy=policy)
    assert failed["failed_node"] == "next_item"

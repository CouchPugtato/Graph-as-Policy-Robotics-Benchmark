import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("policy_visualizer", HERE/"app.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


def vision_record():
    nodes = []
    for identifier, action, target in (("observe", "observe_scene", "detect"),
                                       ("detect", "detect_target", "segment"),
                                       ("segment", "segment_target", "check")):
        nodes.append(dict(id=identifier, action=action, skill="vision",
                          routes={"succeeded": target, "failed": "ABORT"}))
    nodes.append(dict(id="check", action="check_target", skill="vision",
                      routes={"true": "DONE", "false": "ABORT", "failed": "ABORT"}))
    return {"graph_detail": "actions", "policy": {"start": "observe", "nodes": nodes,
                       "terminals": {"DONE": "succeeded", "ABORT": "failed"},
                       "metadata": {"subgraph_membership": {n["id"]: "target" for n in nodes},
                                    "node_kinds": {"check": "router"}}}, "changed": []}


def test_action_view_folds_vision_without_changing_execution():
    record = vision_record()
    original = copy.deepcopy(record)
    view = app.action_view(record)
    assert record == original
    assert [n["action"] for n in view["policy"]["nodes"]] == ["locate target", "check_target"]
    assert view["policy"]["nodes"][0]["routes"] == {"succeeded": "check", "failed": "ABORT"}
    assert view["policy"]["metadata"]["node_kinds"]["check"] == "router"
    assert app.action_view(dict(record, graph_detail="tools"))["policy"] == record["policy"]


def test_folded_progress_requires_all_members_and_maps_failure():
    record = vision_record()
    record.update(completed_nodes=["observe", "detect"], active_node="segment",
                  taken_routes={"observe": "succeeded", "detect": "succeeded"})
    view = app.action_view(record)
    assert view["active_node"] == "observe"
    assert view["completed_nodes"] == []
    assert "observe" not in view["taken_routes"]
    record.update(failed_node="segment", taken_routes={"segment": "failed"})
    view = app.action_view(record)
    assert view["failed_node"] == "observe"
    assert view["taken_routes"]["observe"] == "failed"
    record.update(completed_nodes=["observe", "detect", "segment"],
                  taken_routes={"segment": "succeeded"})
    assert app.action_view(record)["completed_nodes"] == ["observe"]


def test_action_view_does_not_hide_branching_vision():
    record = vision_record()
    record["policy"]["nodes"][1]["routes"]["retry"] = "observe"
    assert len(app.action_view(record)["policy"]["nodes"]) == 4


def fixture_history(tmp_path, records=None):
    graph = {"start": "open", "terminals": {"FAILED": "failed", "SUCCEEDED": "succeeded"},
             "nodes": [{"id": "open", "action": "open", "skill": "grasp",
                        "routes": {"failed": "FAILED", "succeeded": "SUCCEEDED"}}]}
    (tmp_path/"graph.json").write_text(json.dumps(graph))
    data = {"schema_version": 1, "title": "Policy test", "subtitle": "Test artifacts",
            "graph": "graph.json", "iterations": records or
            [{"iteration": 1, "successes": 4, "trials": 10},
             {"iteration": 2, "successes": 9, "trials": 10}]}
    path = tmp_path/"history.json"
    path.write_text(json.dumps(data))
    return path, graph


def test_same_graph_does_not_invent_patches(tmp_path):
    path, _ = fixture_history(tmp_path)
    data = app.load_history(path)
    assert all(not record["changed"] for record in data["iterations"])
    assert all(record["receipt"] is None for record in data["iterations"])


def test_real_node_edit_is_highlighted(tmp_path):
    path, graph = fixture_history(tmp_path)
    graph["nodes"][0]["skill"] = "changed grasp"
    (tmp_path/"second.json").write_text(json.dumps(graph))
    payload = json.loads(path.read_text())
    payload["iterations"][1]["graph"] = "second.json"
    path.write_text(json.dumps(payload))
    assert app.load_history(path)["iterations"][1]["changed"] == ["open"]


@pytest.mark.parametrize("key,value", [("successes", 11), ("trials", 0),
                                        ("iteration", 0)])
def test_invalid_counts_rejected(tmp_path, key, value):
    records = [{"iteration": 1, "successes": 4, "trials": 10}]
    records[0][key] = value
    path, _ = fixture_history(tmp_path, records)
    with pytest.raises(ValueError):
        app.load_history(path)


def test_wrong_receipt_graph_rejected(tmp_path):
    path, _ = fixture_history(tmp_path)
    (tmp_path/"result.json").write_text(json.dumps({"graph_sha256": "not the right hash"}))
    payload = json.loads(path.read_text())
    payload["iterations"][0]["result"] = "result.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different graph"):
        app.load_history(path)


def test_render_is_cpu_image_and_does_not_change_history(tmp_path):
    path, _ = fixture_history(tmp_path)
    data = app.load_history(path)
    before = copy.deepcopy(data)
    image = app.render(data, 1, size=(1280,720))
    assert image.shape == (720,1280,3)
    assert data == before


def test_project_policy_has_the_three_presentation_groups():
    data = app.load_history(HERE/"history.json")
    lanes = app.policy_lanes(data["iterations"][0]["policy"]["nodes"])
    assert [len(members) for _,members in lanes] == [4,3,3]
    assert [title for title,_ in lanes] == ["grasping-direct-ik",
        "transporting-objects / transport", "transporting-objects / release"]


def test_real_subgraph_stays_on_one_line_even_with_interleaved_nodes():
    nodes = [{"id": str(index), "action": "move", "skill": "physical"}
             for index in range(20)]
    membership = {node["id"]: "grasp" if int(node["id"]) % 2 else "place" for node in nodes}
    lanes = app.policy_lanes(nodes, membership)
    assert len(lanes) == 2
    assert [len(members) for _, members in lanes] == [10, 10]
    record = {"policy": {"nodes": nodes, "metadata": {"subgraph_membership": membership}}}
    layout = app.presentation_layout(record)
    assert layout["graph_axes"][2] > .9
    assert layout["footage_axes"][1] + layout["footage_axes"][3] < layout["graph_axes"][1]


def test_boolean_router_is_drawn_as_a_diamond_with_actual_route_labels():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    graph = {"nodes": [{"id": "found", "action": "check_target", "skill": "vision",
                        "routes": {"true": "DONE", "false": "ABORT", "failed": "ABORT"}}],
             "metadata": {"node_kinds": {"found": "router"},
                          "subgraph_membership": {"found": "perception"}},
             "terminals": {"DONE": "succeeded", "ABORT": "failed"}}
    figure, axis = plt.subplots()
    try:
        app.graph_panel(axis, {"policy": graph, "changed": []})
        assert sum(isinstance(patch, Polygon) for patch in axis.patches) == 1
        labels = {text.get_text().replace("\n", " ") for text in axis.texts}
        assert {"target found", "false / failed", "true"} <= labels
    finally:
        plt.close(figure)


def test_active_node_popup_shows_description_without_a_permanent_key():
    import matplotlib.pyplot as plt

    record = vision_record()
    record.update(graph_detail="tools", active_node="detect", node_outcome="succeeded")
    original = copy.deepcopy(record)
    figure, axis = plt.subplots(figsize=(16, 6))
    try:
        app.graph_panel(axis, record)
        popups = [text for text in axis.texts if hasattr(text, "arrow_patch")]
        assert len(popups) == 1
        popup = popups[0].get_text().replace("\n", " ")
        assert "detect target" in popup
        assert "Propose separate colored object regions" in popup
        assert "route  succeeded" in popup
        labels = [text.get_text() for text in axis.texts if text not in popups]
        assert "detect\ntarget" in labels
        assert not any("color object detection" in label for label in labels)
        assert record == original
    finally:
        plt.close(figure)

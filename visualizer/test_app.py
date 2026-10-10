import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("policy_visualizer", HERE/"app.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


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

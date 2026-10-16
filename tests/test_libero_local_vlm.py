import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.error import URLError
from urllib.request import ProxyHandler

import numpy as np
import pytest

from gap_libero import local_vlm
from gap_libero.local_vlm import LocalVLM
from gap_libero.vision import VisionActions, propose_candidates
from gap_libero.vision_workflow import adapt_workflow, vision_task


def image_and_candidates():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[10:25, 10:25] = [220, 20, 20]
    image[10:25, 45:60] = [20, 80, 220]
    return image, propose_candidates(image)


def response(selection="B", finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(
        {"candidate_id": selection, "reason": "blue cylinder"})}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12}}


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "https://openrouter.ai/v1",
                                 "http://127.0.0.1:1234/v1?key=secret"])
def test_local_selector_rejects_external_or_ambiguous_endpoints(url):
    with pytest.raises(ValueError):
        LocalVLM(base_url=url)


def test_local_request_disables_proxy_and_redirects_and_does_not_forward_cloud_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-forward")
    monkeypatch.setenv("OPENROUTER_API_KEY", "do-not-forward")
    monkeypatch.delenv("GAP_VLM_API_KEY", raising=False)
    server = MagicMock()
    server.__enter__.return_value.read.return_value = json.dumps(response()).encode()
    opener = MagicMock()
    opener.open.return_value = server
    factory = MagicMock(return_value=opener)
    monkeypatch.setattr(local_vlm, "build_opener", factory)
    client = LocalVLM()
    client.request("POST", client.endpoint, {"model": client.model})
    assert isinstance(factory.call_args.args[0], ProxyHandler)
    assert factory.call_args.args[0].proxies == {}
    assert isinstance(factory.call_args.args[1], local_vlm.NoRedirect)
    request = opener.open.call_args.args[0]
    assert "Authorization" not in request.headers
    assert opener.open.call_args.kwargs["timeout"] == 35


def test_unreachable_server_is_actionable_without_retry(monkeypatch):
    opener = MagicMock()
    opener.open.side_effect = URLError("secret-bearing backend detail")
    monkeypatch.setattr(local_vlm, "build_opener", lambda *args: opener)
    with pytest.raises(RuntimeError, match="start LM Studio") as error:
        LocalVLM().preflight()
    assert "secret-bearing" not in str(error.value)
    assert opener.open.call_count == 1


@pytest.mark.parametrize("free,accepted", [(12000, True), (2000, False)])
def test_renderer_requires_gpu_headroom(monkeypatch, free, accepted):
    monkeypatch.setattr(local_vlm.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=f"NVIDIA GeForce RTX 5080, {free}"))
    if accepted:
        assert local_vlm.renderer_headroom()["free_vram_before_renderer_mib"] == free
    else:
        with pytest.raises(RuntimeError, match="4096 MiB"):
            local_vlm.renderer_headroom()


def test_preflight_checks_model_without_inference(monkeypatch):
    client = LocalVLM()
    calls = []
    def request(method, url, payload=None):
        calls.append((method, url))
        return {"data": [{"id": client.model}]}
    monkeypatch.setattr(client, "request", request)
    assert client.preflight()["local_only"]
    assert len(calls) == 1 and calls[0][0] == "GET"
    assert client.calls == 0


def test_identification_is_bounded_saved_and_not_retried(tmp_path, monkeypatch):
    rgb, candidates = image_and_candidates()
    assert [item["id"] for item in candidates] == ["A", "B"]
    client, calls = LocalVLM(max_calls=1), []
    def request(method, url, payload):
        calls.append(payload)
        return response()
    monkeypatch.setattr(client, "request", request)
    report = client.identify(rgb, candidates, "blue cylinder", tmp_path / "selection.json")
    assert report["candidate_id"] == "B" and report["local_only"]
    assert calls[0]["max_tokens"] == 128 and calls[0]["stream"] is False
    assert len(calls[0]["messages"][0]["content"]) == 2
    assert (tmp_path / "selection.png").is_file()
    assert (tmp_path / "selection.response.json").is_file()
    with pytest.raises(ValueError, match="budget"):
        client.identify(rgb, candidates, "blue cylinder", tmp_path / "retry.json")
    assert len(calls) == 1


def test_touching_colors_are_separate_candidate_masks():
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[5:35, 5:25] = [15, 35, 220]
    rgb[5:35, 25:45] = [220, 210, 20]
    rgb[20:35, 5:25] = [8, 18, 110]
    candidates = propose_candidates(rgb)
    assert len(candidates) == 2
    assert sorted(item["pixels"] for item in candidates) == [600, 600]
    blue = next(item for item in candidates if item["mask"][10, 10])
    assert blue["mask"][25, 10]
    assert not blue["mask"][10, 30]


def test_touching_basket_and_red_object_are_not_one_candidate():
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[5:35, 5:25] = [15, 170, 30]
    rgb[5:35, 25:45] = [210, 12, 15]
    candidates = propose_candidates(rgb)
    assert len(candidates) == 2
    assert not np.any(candidates[0]["mask"] & candidates[1]["mask"])
    assert all(item["pixels"] == 600 for item in candidates)


def test_red_hue_wraparound_does_not_split_the_same_object():
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[5:20, 5:35] = [200, 12, 10]
    rgb[20:35, 5:35] = [200, 10, 12]
    candidates = propose_candidates(rgb)
    assert len(candidates) == 1
    assert candidates[0]["pixels"] == 900


@pytest.mark.parametrize("selection,finish", [("Z", "stop"), ("B", "length")])
def test_invalid_or_truncated_model_selection_fails_closed(
        tmp_path, monkeypatch, selection, finish):
    rgb, candidates = image_and_candidates()
    client = LocalVLM()
    monkeypatch.setattr(client, "request", lambda *args: response(selection, finish))
    with pytest.raises(ValueError):
        client.identify(rgb, candidates, "blue cylinder", tmp_path / "selection.json")
    assert client.calls == 1
    assert (tmp_path / "selection.response.json").is_file()


@pytest.mark.parametrize("selection", ["A", None])
def test_model_choice_controls_mask_without_known_blue_identity_fallback(tmp_path, selection):
    rgb, _ = image_and_candidates()
    selector = SimpleNamespace(model="test-local-model", identify=lambda *args: {
        "candidate_id": selection, "reason": "injected selection", "local_only": True})
    actions = VisionActions(lambda: None, tmp_path / "vision", identifier=selector)
    actions.bundle = SimpleNamespace(to_gap_camera=lambda: {"rgb": rgb},
                                    observation=SimpleNamespace(rgb_sha256="hash"))
    detection = actions.detect("alphabet_soup", {})
    assert len(detection.outputs["alphabet_soup_detection"]["candidates"]) == 2
    with pytest.raises(RuntimeError, match="identify"):
        actions.segment("alphabet_soup", {})
    actions.identify("alphabet_soup", {})
    assert not actions.masks["alphabet_soup"][15, 50]
    assert actions.masks["alphabet_soup"][15, 15] == (selection == "A")
    if selection is None:
        actions.geometries["alphabet_soup"] = {"valid": True}
        assert actions.found("alphabet_soup", {}).outcome == "false"


def test_local_vlm_catalog_and_adaptation_are_separate_and_idempotent():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "harness/tasks/libero-grocery-vision.json").read_text())
    task = vision_task(original, "local-vlm")
    assert task["task_id"] == "libero-grocery-local-vlm"
    assert original["task_id"] == "libero-grocery-vision"
    assert "identify_target" in task["required_actions"]
    assert "identify_container" in task["required_actions"]
    assert task["runner"]["timeout_s"] == 180
    assert vision_task(task, "local-vlm") == task
    workflow = {"topology": {"subgraphs": []}, "checkpoints": [], "subgraphs": [
        {"name": "target", "nodes": [
            {"name": "observe", "action": "observe_scene", "kind": "tool", "routes": []},
            {"name": "detect", "action": "detect_target", "kind": "tool", "routes": [
                {"outcome": "succeeded", "target": "segment"},
                {"outcome": "failed", "target": "error"}]}]}]}
    adapted = adapt_workflow(workflow, "local-vlm")
    assert adapted["subgraphs"][0]["nodes"][-1]["action"] == "identify_target"
    assert adapt_workflow(adapted, "local-vlm") == adapted

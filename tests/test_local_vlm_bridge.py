import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gap_isaac import local_vlm_bridge as bridge


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "openrouter")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("GAP_VLM_MODEL", "qwen3-vl-4b-instruct")
    monkeypatch.delenv("GAP_VLM_API_KEY", raising=False)
    return SimpleNamespace(
        _png_b64=lambda image: "encoded-pixels",
        _query=lambda *args: None,
        _gather_images=lambda image, images: (
            ([image] if image is not None else []) + (images or [])
        ),
    )


@pytest.fixture
def network(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({
        "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
    }).encode()
    opener = MagicMock()
    opener.open.return_value = response
    build = MagicMock(return_value=opener)
    monkeypatch.setattr(bridge, "build_opener", build)
    return build, opener, response


def test_upstream_dispatch_local_bounded_no_cloud_key(upstream, network, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "never-forward-cloud-key")
    build, opener, _ = network
    state = bridge.install_local_vlm_bridge(upstream)
    image = SimpleNamespace(shape=(396, 792, 3))
    assert upstream._query("Choose A or B", image, None, None, None) == "A"
    assert state["calls"] == 1
    request = opener.open.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:1234/v1/chat/completions"
    assert "never-forward-cloud-key" not in str(request.headers)
    payload = json.loads(request.data)
    assert payload["max_tokens"] == 128
    assert payload["model"] == "qwen3-vl-4b-instruct"
    assert build.call_args.args[0].proxies == {}
    assert set(upstream._PROVIDERS) == {"openrouter"}


@pytest.mark.parametrize("provider,model", [("vertex", None), (None, "other-model")])
def test_overrides_cannot_escape(upstream, network, provider, model):
    bridge.install_local_vlm_bridge(upstream)
    with pytest.raises(ValueError):
        upstream._query("Choose A or B", SimpleNamespace(shape=(396, 792, 3)),
                        None, provider, model)
    network[1].open.assert_not_called()


def test_budget_no_retry(upstream, network):
    bridge.install_local_vlm_bridge(upstream, max_calls=1)
    image = SimpleNamespace(shape=(396, 792, 3))
    upstream._query("A or B", image, None, None, None)
    with pytest.raises(ValueError, match="budget exhausted"):
        upstream._query("A or B", image, None, None, None)
    network[1].open.assert_called_once()


def test_truncated_reply_not_accepted(upstream, network):
    network[2].__enter__.return_value.read.return_value = json.dumps({
        "choices": [{"message": {"content": "A"}, "finish_reason": "length"}],
    }).encode()
    bridge.install_local_vlm_bridge(upstream)
    with pytest.raises(ValueError, match="truncated"):
        upstream._query("A or B", SimpleNamespace(shape=(396, 792, 3)), None, None, None)


@pytest.mark.parametrize("variable,value", [
    ("GAP_VLM_PROVIDER", ""), ("GAP_VLM_MODEL", ""),
    ("GAP_VLM_BASE_URL", "https://openrouter.ai/api/v1"),
])
def test_missing_or_remote_configuration_rejected(upstream, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError):
        bridge.install_local_vlm_bridge(upstream)

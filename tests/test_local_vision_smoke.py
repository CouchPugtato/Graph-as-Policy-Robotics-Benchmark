import json
import struct
from unittest.mock import MagicMock
from urllib.request import ProxyHandler

import pytest

from gap_isaac import local_vision_smoke as vision


@pytest.mark.parametrize("url", [
    "https://openrouter.ai/v1", "http://localhost:1234/v1",
    "http://127.0.0.1:1234/v1?x=1", "http://user@127.0.0.1:1234/v1",
    "http://127.0.0.1:0/v1", "http://127.0.0.1:1234/api",
])
def test_reject_nonlocal_or_ambiguous_endpoint(url):
    with pytest.raises(ValueError):
        vision.validated_endpoint(url)


def test_accept_numeric_loopback():
    assert vision.validated_endpoint("http://127.0.0.1:1234/v1/").endswith(
        "/v1/chat/completions"
    )


def test_redirect_blocked():
    assert vision.NoRedirect().redirect_request(None, None, 302, "", {}, "https://x") is None


def png_header(width=640, height=480):
    return (
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
        + struct.pack(">II", width, height) + bytes(9)
    )


def test_image_bounds(tmp_path):
    image = tmp_path / "rgb.png"
    image.write_bytes(png_header(641))
    with pytest.raises(ValueError, match="dimensions"):
        vision.image_payload(image)
    image.write_bytes(b"not png")
    with pytest.raises(ValueError, match="rgb.png"):
        vision.image_payload(image)
    image.write_bytes(bytes(vision.MAX_IMAGE_BYTES + 1))
    with pytest.raises(ValueError, match="2 MiB"):
        vision.image_payload(image)


def test_one_bounded_post_no_proxy_no_semantic_claim(tmp_path, monkeypatch):
    image = tmp_path / "rgb.png"
    image.write_bytes(png_header())
    probe = MagicMock()
    probe.__enter__.return_value.connect_ex.return_value = 1
    monkeypatch.setattr(vision.socket, "socket", lambda: probe)
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({
        "choices": [{"message": {"content": "A green open container."},
                     "finish_reason": "stop"}],
    }).encode()
    opener = MagicMock()
    opener.open.return_value = response
    build = MagicMock(return_value=opener)
    monkeypatch.setattr(vision, "build_opener", build)
    monkeypatch.delenv("GAP_VLM_API_KEY", raising=False)
    result = vision.run_test(image, "qwen3-vl-4b-instruct", "http://127.0.0.1:1234/v1")
    assert result["api_ok"] and not result["semantic_validated"]
    opener.open.assert_called_once()
    request = opener.open.call_args.args[0]
    payload = json.loads(request.data)
    assert payload["max_tokens"] == 256
    assert payload["stream"] is False
    assert request.method == "POST"
    assert opener.open.call_args.kwargs["timeout"] == 45
    assert isinstance(build.call_args.args[0], ProxyHandler)
    assert build.call_args.args[0].proxies == {}
    assert isinstance(build.call_args.args[1], vision.NoRedirect)


def test_refuse_running_connector(tmp_path, monkeypatch):
    probe = MagicMock()
    probe.__enter__.return_value.connect_ex.return_value = 0
    monkeypatch.setattr(vision.socket, "socket", lambda: probe)
    with pytest.raises(ValueError, match="Stop the Isaac"):
        vision.run_test(tmp_path / "missing.png", "qwen3-vl", "http://127.0.0.1:1234/v1")

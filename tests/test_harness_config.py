import json
import os

import httpx
import pytest

from gap_harness.config import PLATFORM_URL, endpoint, load_environment
from gap_harness.models import Topology
from gap_harness.provider import OpenAIResponses
from gap_harness.proxy import PACKAGE, command


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    environment = dict(os.environ)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "GAP_CODEX_PROXY_KEY", "OPENAI_MODEL",
                "OPENAI_REASONING_EFFORT"):
        environment.pop(key, None)
    monkeypatch.setattr(os, "environ", environment)


def test_dotenv_reads_only_known_settings_without_expansion(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "GAP_CODEX_PROXY_KEY", "OPENAI_MODEL",
                "OPENAI_REASONING_EFFORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "shell-model")
    monkeypatch.setenv("PATH", "unchanged")
    monkeypatch.setenv("SOURCE_TOKEN", "never-expand")
    path = tmp_path / ".env"
    path.write_text('OPENAI_BASE_URL="http://127.0.0.1:18080/v1"\n'
                    'OPENAI_API_KEY=\nGAP_CODEX_PROXY_KEY=${SOURCE_TOKEN}\n'
                    'OPENAI_MODEL=file-model\nOPENAI_REASONING_EFFORT=medium\nPATH=changed\n')
    load_environment(path)
    assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:18080/v1"
    assert os.environ["OPENAI_MODEL"] == "shell-model"
    assert os.environ["OPENAI_REASONING_EFFORT"] == "medium"
    assert os.environ["GAP_CODEX_PROXY_KEY"] == "${SOURCE_TOKEN}"
    assert os.environ["PATH"] == "unchanged"


@pytest.mark.parametrize("url", [PLATFORM_URL, "http://127.0.0.1:18080/v1/",
                                     "http://[::1]:18080/v1", "http://localhost:18080/v1"])
def test_endpoint_accepts_only_explicit_official_or_local_servers(url):
    value, local = endpoint(url)
    assert value.endswith("/v1")
    assert local == (url != PLATFORM_URL)


@pytest.mark.parametrize("url", ["http://api.openai.com/v1", "https://example.com/v1",
    "http://192.168.1.2:18080/v1", "http://localhost.evil.test/v1",
    "http://user:secret@127.0.0.1:18080/v1", "http://127.0.0.1:18080/v1?token=secret",
    "http://127.0.0.1:18080/v1#secret", "http://127.0.0.1:18080/responses",
    "http://127.0.0.1:invalid/v1"])
def test_endpoint_rejects_credentials_remote_proxies_and_ambiguous_urls(url):
    with pytest.raises(ValueError):
        endpoint(url)


def test_local_proxy_does_not_receive_platform_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-platform-key")
    monkeypatch.delenv("GAP_CODEX_PROXY_KEY", raising=False)
    captured = []
    def respond(request):
        captured.append(request)
        return httpx.Response(200, json={"status": "completed", "output": []})
    provider = OpenAIResponses(base_url="http://127.0.0.1:18080/v1",
                              transport=httpx.MockTransport(respond))
    provider.complete("test", {}, Topology)
    assert "authorization" not in captured[0].headers
    assert "private-platform-key" not in json.dumps(provider.public_settings())


def test_proxy_and_platform_credentials_are_separate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "platform-key")
    monkeypatch.setenv("GAP_CODEX_PROXY_KEY", "local-key")
    captured = []
    def respond(request):
        captured.append(request)
        return httpx.Response(200, json={"status": "completed", "output": []})
    transport = httpx.MockTransport(respond)
    OpenAIResponses(base_url="http://127.0.0.1:18080/v1", transport=transport).complete(
        "test", {}, Topology)
    OpenAIResponses(base_url=PLATFORM_URL, transport=transport).complete("test", {}, Topology)
    assert captured[0].headers["authorization"] == "Bearer local-key"
    assert captured[1].headers["authorization"] == "Bearer platform-key"


def test_proxy_daemon_command_places_subcommand_before_flags(monkeypatch):
    monkeypatch.setattr("gap_harness.proxy.shutil.which", lambda name: "uvx")
    arguments = command("start")
    assert arguments.index("start") == arguments.index(PACKAGE) + 1
    assert arguments.index("start") < arguments.index("--config")


def test_model_and_reasoning_are_explicit_in_requests_and_metadata():
    captured = []
    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "completed", "output": []})
    provider = OpenAIResponses(base_url="http://127.0.0.1:18080/v1",
        reasoning_effort="medium", transport=httpx.MockTransport(respond))
    provider.complete("test", {}, Topology)
    assert captured[0]["model"] == "gpt-6.1-sol"
    assert captured[0]["reasoning"] == {"effort": "medium"}
    assert provider.public_settings()["reasoning_effort"] == "medium"


def test_invalid_reasoning_is_rejected_before_a_request():
    with pytest.raises(ValueError, match="reasoning effort"):
        OpenAIResponses(reasoning_effort="unlimited")

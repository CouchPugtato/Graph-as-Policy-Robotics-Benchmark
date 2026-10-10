import importlib.metadata
import sys

import pytest

from gap_isaac import perception_preflight as module


def test_preflight_does_not_import_models_or_claim_inference(tmp_path, monkeypatch):
    before = set(sys.modules)
    monkeypatch.setattr(module, "_revision", lambda path: None)
    report = module.run_preflight(tmp_path)
    assert not ({"torch", "transformers", "sam3", "triton"} & (set(sys.modules) - before))
    assert report["models_loaded"] is False
    assert report["network_requests"] == 0
    assert report["native_inference_validated"] is False
    assert report["ready_for_import_probe"] is False


def test_credential_values_never_appear_in_report(tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "OPENROUTER_API_KEY", "GAP_VLM_API_KEY", "GAP_VLM_BASE_URL"):
        monkeypatch.setenv(name, "private-value-do-not-print")
    report = module.run_preflight(tmp_path)
    assert "private-value-do-not-print" not in str(report)
    assert report["credentials_presence_only"]["hf_token_environment"] is True
    assert report["credentials_validated"] is False


def test_missing_packages_are_reported_without_installing(tmp_path, monkeypatch):
    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(module.importlib.metadata, "version", absent)
    report = module.run_preflight(tmp_path)
    assert report["missing_packages"] == list(module.PACKAGES)
    assert not report["checks"]["packages_present"]


def test_pinned_files_and_isolated_python_allow_import_probe_only(tmp_path, monkeypatch):
    skills = tmp_path / "cache/upstream/open-robot-skills"
    script = skills / "skills/perceiving-objects/scripts/perceive_dino_vlm.py"
    script.parent.mkdir(parents=True)
    script.touch()
    monkeypatch.setattr(module.sys, "version_info", (3, 12))
    monkeypatch.setattr(module.sys, "prefix", "isolated")
    monkeypatch.setattr(module.sys, "base_prefix", "base")
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "present")
    monkeypatch.setattr(
        module,
        "_revision",
        lambda path: (
            module.SKILLS_REVISION if path.name == "open-robot-skills" else module.CORE_REVISION
        ),
    )
    report = module.run_preflight(tmp_path)
    assert report["ready_for_import_probe"] is True
    assert report["native_inference_validated"] is False
    assert report["weights_validated"] is False


def test_git_inspection_is_read_only_and_bounded(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    calls = []

    class Result:
        returncode = 0
        stdout = module.SKILLS_REVISION

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._revision(tmp_path) == module.SKILLS_REVISION
    assert calls[0][0][-2:] == ["rev-parse", "HEAD"]
    assert calls[0][1]["timeout"] == 5


def test_no_git_checkout_reports_unknown_revision(tmp_path):
    assert module._revision(tmp_path) is None


@pytest.mark.parametrize(
    "endpoint",
    ["http://127.0.0.1:1234/v1", "http://localhost:1234/v1/", "http://[::1]:1234/v1"],
)
def test_explicit_loopback_endpoints_accepted(endpoint):
    assert module.is_loopback_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "https://openrouter.ai/api/v1",
        "http://192.168.1.1:1234/v1",
        "http://localhost.example/v1",
        "http://user:secret@localhost/v1",
        "http://localhost/v1?key=secret",
        "http://localhost/v1#secret",
        "http://localhost:99999/v1",
        "http://localhost:0/v1",
        "http://localhost",
        "file:///v1",
    ],
)
def test_remote_implicit_or_credential_endpoints_rejected(endpoint):
    assert not module.is_loopback_endpoint(endpoint)


def test_local_configuration_is_explicit_and_separate_from_import_readiness(tmp_path, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "openrouter")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("GAP_VLM_MODEL", "private-model-id")
    report = module.run_preflight(tmp_path)
    assert report["local_vlm_configuration_ready"] is True
    assert report["inference_policy"] == "local-only"
    assert report["local_server_validated"] is False
    assert report["graph_generator_configuration_validated"] is False
    assert "private-model-id" not in str(report)


def test_hosted_url_never_counts_as_local_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "openrouter")
    monkeypatch.setenv("GAP_VLM_MODEL", "a-model")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "https://openrouter.ai/api/v1")
    report = module.run_preflight(tmp_path)
    assert not report["local_vlm_configuration_ready"]

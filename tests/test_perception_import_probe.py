import socket

import pytest

from gap_isaac import perception_import_probe as probe


def test_network_guard():
    with pytest.raises(RuntimeError, match="disabled"):
        probe.blocked_connect()


def test_imports_do_not_construct_models_and_restore_network(monkeypatch):
    original = socket.socket.connect
    imported = []

    def fake_import(name):
        assert socket.socket.connect is probe.blocked_connect
        imported.append(name)

    monkeypatch.setattr(probe.importlib, "import_module", fake_import)
    result = probe.run_probe()
    assert result["ok"]
    assert imported == list(probe.IMPORTS)
    assert result["models_loaded"] is False
    assert result["inference_validated"] is False
    assert socket.socket.connect is original


def test_import_errors_are_reported(monkeypatch):
    def fail(name):
        raise ImportError("missing dependency")

    monkeypatch.setattr(probe.importlib, "import_module", fail)
    result = probe.run_probe()
    assert not result["ok"]
    assert result["errors"]["sam3.model_builder"]["type"] == "ImportError"

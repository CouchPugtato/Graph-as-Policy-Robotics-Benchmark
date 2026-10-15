import os
from pathlib import Path

import pytest

from gap_libero.runtime_config import prepare_runtime_config


def test_missing_explicit_config_fails_before_libero_import(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="LIBERO config missing"):
        prepare_runtime_config()
    assert not (tmp_path / "config.yaml").exists()


def test_prepared_config_is_selected(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("assets: assets\n")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    assert prepare_runtime_config() == tmp_path.resolve()
    assert Path(os.environ["LIBERO_CONFIG_PATH"]) == tmp_path.resolve()


def test_missing_environment_uses_project_config(tmp_path, monkeypatch):
    import gap_libero.runtime_config as config

    module = tmp_path / "src/gap_libero/runtime_config.py"
    monkeypatch.setattr(config, "__file__", str(module))
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    directory = tmp_path / "cache/libero-config"
    directory.mkdir(parents=True)
    (directory / "config.yaml").write_text("assets: assets\n")
    assert prepare_runtime_config() == directory.resolve()

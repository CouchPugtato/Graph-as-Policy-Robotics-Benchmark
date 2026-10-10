from unittest.mock import MagicMock

import pytest

from gap_isaac import cuda_probe


def test_exact_versions(monkeypatch):
    monkeypatch.setattr(cuda_probe.importlib.metadata, "version", cuda_probe.EXPECTED.__getitem__)
    assert cuda_probe.check_versions() == cuda_probe.EXPECTED
    monkeypatch.setattr(cuda_probe.importlib.metadata, "version", lambda name: "wrong")
    with pytest.raises(ValueError, match="Expected pinned"):
        cuda_probe.check_versions()


@pytest.mark.parametrize("reading", ["NVIDIA GeForce RTX 5080, 5113", "Other GPU, 500"])
def test_busy_or_wrong_gpu_rejected(monkeypatch, reading):
    probe = MagicMock()
    probe.__enter__.return_value.connect_ex.return_value = 1
    monkeypatch.setattr(cuda_probe.socket, "socket", lambda: probe)
    monkeypatch.setattr(cuda_probe.subprocess, "run", lambda *a, **k: MagicMock(stdout=reading))
    with pytest.raises(ValueError, match="RTX 5080"):
        cuda_probe.check_idle_gpu()


def test_idle_5080_accepted(monkeypatch):
    probe = MagicMock()
    probe.__enter__.return_value.connect_ex.return_value = 1
    monkeypatch.setattr(cuda_probe.socket, "socket", lambda: probe)
    monkeypatch.setattr(cuda_probe.subprocess, "run", lambda *a, **k:
                        MagicMock(stdout="NVIDIA GeForce RTX 5080, 512"))
    assert cuda_probe.check_idle_gpu()["memory_used_before_mib"] == 512

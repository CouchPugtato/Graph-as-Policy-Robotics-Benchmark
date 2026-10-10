import hashlib

import pytest

from gap_isaac.perception_weights import ARTIFACTS, verify_file


def test_selected_artifacts_are_pinned():
    for _, revision, _, files in ARTIFACTS:
        assert len(revision) == 40
        assert "pytorch_model.bin" not in files


def test_checkpoint_hash_and_size(tmp_path):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    assert verify_file(path, 7, digest) == digest
    with pytest.raises(ValueError, match="size"):
        verify_file(path, 8, digest)
    with pytest.raises(ValueError, match="SHA256"):
        verify_file(path, 7, "0" * 64)


def test_git_etag_is_not_treated_as_file_sha256(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"{}")
    assert verify_file(path, 2, "a" * 40) == hashlib.sha256(b"{}").hexdigest()

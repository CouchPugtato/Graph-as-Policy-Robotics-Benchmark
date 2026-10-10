from pathlib import Path

from gap_isaac.manifest import build_manifest, sha256_file, write_manifest


def test_manifest_hashes_inputs_without_serializing_environment_secrets(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "config.yaml"
    source.write_text("value: 1\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")

    manifest = build_manifest(tmp_path, [source], metadata={"run": "test"})
    assert manifest["files"]["config.yaml"] == sha256_file(source)
    assert "must-not-leak" not in str(manifest)

    output = tmp_path / "manifest.json"
    write_manifest(output, manifest)
    assert output.read_text(encoding="utf-8").endswith("\n")

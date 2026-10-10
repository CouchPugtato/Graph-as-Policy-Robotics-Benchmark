import json
import hashlib

import pytest

from gap_isaac.dino_smoke import local_snapshot
from gap_isaac.perception_weights import ARTIFACTS, DINO_REVISION


def fixture_snapshot(root):
    snapshot = root / "cache/models/huggingface/models--IDEA-Research--grounding-dino-base/snapshots" / DINO_REVISION
    snapshot.mkdir(parents=True)
    records = []
    for name in ARTIFACTS[0][3]:
        (snapshot / name).write_bytes(b"fixture")
        records.append({"repo": ARTIFACTS[0][0], "revision": DINO_REVISION,
                        "file": name, "bytes": 7,
                        "sha256": hashlib.sha256(b"fixture").hexdigest()})
    manifest = root / "weights.json"
    manifest.write_text(json.dumps({"artifacts": records}))
    return snapshot, manifest


def test_local_snapshot_verifies_every_selected_file(tmp_path):
    snapshot, manifest = fixture_snapshot(tmp_path)
    assert local_snapshot(tmp_path, manifest) == snapshot
    (snapshot / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        local_snapshot(tmp_path, manifest)


def test_manifest_cannot_redirect_to_other_paths(tmp_path):
    snapshot, manifest = fixture_snapshot(tmp_path)
    data = json.loads(manifest.read_text())
    for record in data["artifacts"]:
        record["path"] = "not-the-local-snapshot"
    manifest.write_text(json.dumps(data))
    assert local_snapshot(tmp_path, manifest) == snapshot

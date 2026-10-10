"""Download immutable perception artifacts serially, without importing ML models."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

DINO_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
SAM_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
ARTIFACTS = (
    ("IDEA-Research/grounding-dino-base", DINO_REVISION, False,
     ("config.json", "preprocessor_config.json", "special_tokens_map.json",
      "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors")),
    ("facebook/sam3", SAM_REVISION, True, ("config.json", "sam3.pt")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, etag: str) -> str:
    if path.stat().st_size != size:
        raise ValueError("Downloaded file size mismatch")
    digest = sha256_file(path)
    # HF LFS ETags are SHA256; ordinary Git blob ETags are not plain file hashes.
    expected = etag.strip('"')
    if len(expected) == 64 and digest != expected:
        raise ValueError("Downloaded checkpoint SHA256 mismatch")
    return digest


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    cache = root / "cache" / "models" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "0"
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        import requests
        from huggingface_hub import (configure_http_backend, get_token,
                                     get_hf_file_metadata, hf_hub_url, hf_hub_download)

        def direct_session():
            session = requests.Session()
            session.trust_env = False
            return session

        configure_http_backend(backend_factory=direct_session)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        credential = get_token()
        if not credential:
            raise ValueError("A saved Hugging Face login is required")
        planned = []
        for repo, revision, gated, files in ARTIFACTS:
            for name in files:
                metadata = get_hf_file_metadata(
                    hf_hub_url(repo, name, revision=revision),
                    token=credential if gated else False)
                if metadata.commit_hash != revision or metadata.size is None:
                    raise ValueError("Artifact metadata does not match immutable revision")
                planned.append((repo, revision, gated, name, metadata))
        total = sum(item[4].size for item in planned)
        if shutil.disk_usage(cache).free < total * 2 + 1024**3:
            raise ValueError("Insufficient disk space for cache and Windows copy fallback")
        print(json.dumps({"planned_bytes": total, "files": len(planned),
                          "models_loaded": False}), flush=True)
        records = []
        for repo, revision, gated, name, metadata in planned:
            print(f"Downloading/verifying {repo}/{name}", flush=True)
            path = Path(hf_hub_download(repo, name, revision=revision,
                        cache_dir=str(cache), token=credential if gated else False))
            digest = verify_file(path, metadata.size, metadata.etag)
            records.append({"repo": repo, "revision": revision, "file": name,
                            "path": str(path), "bytes": metadata.size,
                            "sha256": digest, "hub_etag": metadata.etag})
            print(f"Verified {name}: {metadata.size} bytes", flush=True)
        destination = root / "results" / "perception"
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / ("weights-" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ") + ".json")
        result = {"ok": True, "models_loaded": False, "inference_validated": False,
                  "total_bytes": total, "artifacts": records}
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps({"ok": True, "manifest": str(manifest),
                          "total_bytes": total, "models_loaded": False}), flush=True)
        return 0
    except Exception as exc:
        # Signed download URLs and credentials must never appear in error output.
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "http_status": getattr(getattr(exc, "response", None),
                                                 "status_code", None),
                          "models_loaded": False}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

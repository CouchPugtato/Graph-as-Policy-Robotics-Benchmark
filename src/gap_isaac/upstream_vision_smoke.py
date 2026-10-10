"""Test the real pinned upstream VLM query with one local image request."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def run_test(root: Path, image: Path, model: str, base_url: str) -> dict:
    from gap_isaac.local_vision_smoke import image_payload, validated_endpoint
    from gap_isaac.local_vlm_bridge import install_local_vlm_bridge
    from gap_isaac.perception_preflight import SKILLS_REVISION

    validated_endpoint(base_url)
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", 8765)) == 0:
            raise ValueError("Stop the Isaac connector first")
    pixels, width, height = image_payload(image)
    checkout = root / "cache/upstream/open-robot-skills"
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()
    if head != SKILLS_REVISION or changes:
        raise ValueError("Upstream skills must be clean and at the pinned revision")
    # These imports do not load models or initialize CUDA.
    import numpy as np
    from PIL import Image

    source = checkout / "tools/vlm/tools.py"
    spec = importlib.util.spec_from_file_location("gap_local_upstream_vlm", source)
    if spec is None or spec.loader is None:
        raise ValueError("Upstream VLM source not found")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    os.environ["GAP_VLM_PROVIDER"] = "openrouter"
    os.environ["GAP_VLM_BASE_URL"] = base_url
    os.environ["GAP_VLM_MODEL"] = model
    state = install_local_vlm_bridge(module, max_calls=1)
    with Image.open(io.BytesIO(pixels)) as preview:
        rgb = np.asarray(preview.convert("RGB"), dtype=np.uint8)
    result = module.query(
        prompt="Briefly list the colored objects and open container visible. "
               "Say uncertain when unclear; do not invent product names.",
        image=rgb,
    )
    return {
        "api_ok": True, "upstream_tool": "vlm.query",
        "skills_revision": head, "local_only": True,
        "model": model, "vlm_calls": state["calls"],
        "image_sha256": hashlib.sha256(pixels).hexdigest(),
        "width": width, "height": height, "reply": result["text"],
        "semantic_validated": False, "masks_produced": False,
        "models_loaded_by_test": False, "motion_enabled": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    # Direct-file execution in the isolated environment, without installing
    # the project's unrelated web server dependencies into that environment.
    sys.path.insert(0, str(root / "src"))
    try:
        result = run_test(root, args.image, args.model, args.base_url)
    except Exception as exc:
        # Don't print backend exception bodies or credential-containing headers.
        print(f"ERROR: upstream VLM test failed ({type(exc).__name__}); "
              "check installed dependencies and LM Studio server logs", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

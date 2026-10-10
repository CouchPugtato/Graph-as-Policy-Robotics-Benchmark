"""One offline, bounded call to the pinned upstream Grounding DINO tool."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def check_stopped_apps():
    import psutil
    for process in psutil.process_iter(["name"]):
        name = (process.info["name"] or "").lower()
        if name in ("kit.exe", "isaac-sim.exe", "lm studio.exe", "llmster.exe"):
            raise ValueError("Close Isaac and LM Studio before testing DINO")
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", 1234)) == 0:
            raise ValueError("Stop the local vision server before testing DINO")
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise ValueError("At least 8 GiB available system RAM is required")


def local_snapshot(root: Path, manifest: Path) -> Path:
    from gap_isaac.perception_weights import ARTIFACTS, DINO_REVISION, verify_file
    data = json.loads(manifest.read_text(encoding="utf-8"))
    records = {item["file"]: item for item in data["artifacts"]
               if item["repo"] == ARTIFACTS[0][0] and item["revision"] == DINO_REVISION}
    snapshot = root / "cache/models/huggingface/models--IDEA-Research--grounding-dino-base/snapshots" / DINO_REVISION
    for name in ARTIFACTS[0][3]:
        record = records[name]
        if verify_file(snapshot / name, record["bytes"], record["sha256"]) != record["sha256"]:
            raise ValueError("DINO artifact provenance mismatch")
    return snapshot


def run(root: Path, image_path: Path, manifest: Path, query: str) -> dict:
    from gap_isaac.cuda_probe import check_idle_gpu, check_versions
    from gap_isaac.perception_preflight import SKILLS_REVISION
    check_stopped_apps()
    versions = check_versions()
    gpu = check_idle_gpu()
    snapshot = local_snapshot(root, manifest)
    checkout = root / "cache/upstream/open-robot-skills"
    for args, expected in ((["rev-parse", "HEAD"], SKILLS_REVISION),
                           (["status", "--porcelain"], "")):
        actual = subprocess.run(["git", "-C", str(checkout), *args],
                                capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        if actual != expected:
            raise ValueError("Upstream skills must be clean and pinned")
    os.environ.update(CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      GAP_DINO_DEVICE="cuda:0", GAP_DINO_MODEL=str(snapshot))
    # Prevent fallback downloads, cloud calls, or local VLM calls in this worker.
    def blocked(*args, **kwargs):
        raise RuntimeError("Network disabled in offline DINO worker")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    # leave memory headroom rather than letting detection consume the whole gpu
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
    if torch.cuda.mem_get_info(0)[0] < 8 * 1024**3:
        raise ValueError("At least 8 GiB free VRAM required")
    with Image.open(image_path) as image:
        if image.size != (640, 480):
            raise ValueError("This bounded smoke test requires a 640x480 capture")
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    spec = importlib.util.spec_from_file_location("gap_offline_dino", checkout / "tools/grounding-dino/tools.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    started = time.monotonic()
    print("Loading only DINO and running one saved image; network blocked.", flush=True)
    with torch.inference_mode():
        result = module.detect(image=array, query=query, box_threshold=0.2, text_threshold=0.2)
    torch.cuda.synchronize()
    output = root / "results/perception" / ("dino-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    crops = []
    for index, detection in enumerate(result["detections"]):
        box = detection["box"]
        bounds = (max(0, int(box["x1"])), max(0, int(box["y1"])),
                  min(640, int(np.ceil(box["x2"]))), min(480, int(np.ceil(box["y2"]))))
        if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
            name = f"candidate-{index:02d}.png"
            rgb.crop(bounds).save(output / name)
            crops.append({"detection_index": index, "file": name})
    report = {"ok": True, "inference_validated": True, "semantic_validated": False,
              "upstream_tool": "grounding-dino.detect", "skills_revision": SKILLS_REVISION,
              "model_snapshot": str(snapshot), "image": str(image_path.resolve()),
              "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "query": query, "detections": result["detections"], "crops": crops,
              "detection_count": len(result["detections"]), "gpu": gpu, "packages": versions,
              "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(0),
              "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(0),
              "elapsed_s": time.monotonic() - started, "network_connections_blocked": True,
              "sam3_loaded": False, "vlm_calls": 0, "motion_enabled": False}
    (output / "detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"ok": True, "result": str(output / "detections.json"),
            "detection_count": report["detection_count"],
            "peak_torch_allocated_bytes": report["peak_torch_allocated_bytes"],
            "semantic_validated": False, "network_connections_blocked": True}


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query", default="red cylinder. brown cylinder. yellow cylinder. blue cylinder. green open box.")
    args = parser.parse_args()
    try:
        print(json.dumps(run(root, args.image, args.manifest, args.query), indent=2), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "detail": str(exc)[:1500], "motion_enabled": False}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

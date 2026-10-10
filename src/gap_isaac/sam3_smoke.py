"""Offline SAM3-only box segmentation using the real pinned upstream tool."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def select_candidate(report: dict, label: str = "blue cylinder") -> tuple[int, dict]:
    candidates = [(index, item) for index, item in enumerate(report["detections"])
                  if item["label"].strip() == label]
    if not candidates:
        raise ValueError("No exact target label in DINO candidates")
    # only compare confidence among candidates with the requested label
    index, candidate = max(candidates, key=lambda pair: pair[1]["score"])
    box = candidate["box"]
    if not all(math.isfinite(float(box[key])) for key in ("x1", "y1", "x2", "y2")):
        raise ValueError("Nonfinite DINO box")
    if not (0 <= box["x1"] < box["x2"] <= 640 and 0 <= box["y1"] < box["y2"] <= 480):
        raise ValueError("DINO box outside saved capture")
    return index, candidate


def local_checkpoint(root: Path, manifest: Path) -> Path:
    from gap_isaac.perception_weights import SAM_REVISION, verify_file
    data = json.loads(manifest.read_text(encoding="utf-8"))
    snapshot = root / "cache/models/huggingface/models--facebook--sam3/snapshots" / SAM_REVISION
    records = {item["file"]: item for item in data["artifacts"]
               if item["repo"] == "facebook/sam3" and item["revision"] == SAM_REVISION}
    for name in ("config.json", "sam3.pt"):
        record = records[name]
        if verify_file(snapshot / name, record["bytes"], record["sha256"]) != record["sha256"]:
            raise ValueError("SAM3 artifact provenance mismatch")
    return snapshot / "sam3.pt"


def run(root: Path, detections_path: Path, manifest: Path, object_id: str = "alphabet_soup") -> dict:
    from gap_isaac.cuda_probe import check_idle_gpu, check_versions
    from gap_isaac.dino_smoke import check_stopped_apps
    from gap_isaac.perception_preflight import SKILLS_REVISION, SAM3_REVISION
    check_stopped_apps()
    versions = check_versions()
    gpu = check_idle_gpu()
    report = json.loads(detections_path.read_text(encoding="utf-8"))
    # fixed scene colors replace model based object identification
    labels = {"alphabet_soup": "blue cylinder", "basket": "green open box"}
    if object_id not in labels:
        raise ValueError("Only the target and basket are in this bounded test")
    index, candidate = select_candidate(report, labels[object_id])
    image_path = Path(report["image"])
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if image_sha != report["image_sha256"]:
        raise ValueError("Capture changed since DINO inference")
    checkpoint = local_checkpoint(root, manifest)
    checkout = root / "cache/upstream/open-robot-skills"
    for args, expected in ((["rev-parse", "HEAD"], SKILLS_REVISION),
                           (["status", "--porcelain"], "")):
        actual = subprocess.run(["git", "-C", str(checkout), *args],
                                capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        if actual != expected:
            raise ValueError("Upstream skills must be clean and pinned")
    os.environ.update(CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      GAP_SAM3_DEVICE="cuda")
    def blocked(*args, **kwargs):
        raise RuntimeError("Network disabled in offline SAM3 worker")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    import importlib.metadata
    import numpy as np
    import torch
    from PIL import Image
    # Verify the installed SAM3 source pin, not just its package version.
    origin = json.loads(importlib.metadata.distribution("sam3").read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != SAM3_REVISION:
        raise ValueError("Installed SAM3 source must match the pinned revision")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.50, 0)
    if torch.cuda.mem_get_info(0)[0] < 10 * 1024**3:
        raise ValueError("At least 10 GiB free VRAM required for SAM3-only test")
    with Image.open(image_path) as image:
        if image.size != (640, 480):
            raise ValueError("This bounded test requires the 640x480 capture")
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    spec = importlib.util.spec_from_file_location("gap_offline_sam3", checkout / "tools/sam3/tools.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    started = time.monotonic()
    print(f"Selected DINO candidate {index}: {labels[object_id]}. Loading only SAM3; network blocked.", flush=True)
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    # Only the loader is adapted: preserve actual upstream box prompting and mask processing.
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False,
                                   device="cuda", eval_mode=True, compile=False,
                                   enable_inst_interactivity=True)
    processor = Sam3Processor(model, confidence_threshold=0.0)
    module._get_model = lambda device=None: (model, processor)
    with torch.inference_mode():
        result = module.segment_box(image=array, box=candidate["box"])
    torch.cuda.synchronize()
    if not result["masks"]:
        raise RuntimeError("SAM3 returned no candidate masks")
    output = root / "results/perception" / ("sam3-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    masks = []
    for mask_index, (mask, score) in enumerate(zip(result["masks"], result["scores"], strict=True)):
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.shape != (480, 640) or not np.isin(mask, (0, 255)).all():
            raise RuntimeError("SAM3 mask violates 640x480 binary mask contract")
        ys, xs = np.nonzero(mask)
        filename = f"mask-{mask_index:02d}.png"
        Image.fromarray(mask).save(output / filename)
        masks.append({"file": filename, "score": float(score), "foreground_pixels": len(xs),
                      "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None})
    if not masks[0]["foreground_pixels"]:
        raise RuntimeError("Top-scoring SAM3 mask is empty")
    result_report = {"ok": True, "inference_validated": True, "semantic_validated": False,
                     "upstream_tool": "sam3.segment_box", "skills_revision": SKILLS_REVISION,
                     "sam3_source_revision": SAM3_REVISION, "checkpoint": str(checkpoint),
                     "loader_adaptation": "explicit verified checkpoint; load_from_HF=False; compile=False",
                     "selection_source": "scene-config color mapping plus exact DINO label; no VLM selection",
                     "target_id": object_id, "object_id": object_id,
                     "dino_result": str(detections_path.resolve()),
                     "dino_candidate_index": index, "dino_candidate": candidate,
                     "image": str(image_path), "image_sha256": image_sha, "masks": masks,
                     "gpu": gpu, "packages": versions,
                     "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(0),
                     "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(0),
                     "elapsed_s": time.monotonic() - started, "network_connections_blocked": True,
                     "dino_loaded": False, "vlm_calls": 0, "motion_enabled": False}
    (output / "segmentation.json").write_text(json.dumps(result_report, indent=2), encoding="utf-8")
    return {"ok": True, "result": str(output / "segmentation.json"), "mask_count": len(masks),
            "top_mask_pixels": masks[0]["foreground_pixels"],
            "peak_torch_allocated_bytes": result_report["peak_torch_allocated_bytes"],
            "semantic_validated": False}


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object-id", choices=("alphabet_soup", "basket"), default="alphabet_soup")
    args = parser.parse_args()
    try:
        print(json.dumps(run(root, args.detections, args.manifest, args.object_id), indent=2), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "detail": str(exc)[:1500], "motion_enabled": False}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

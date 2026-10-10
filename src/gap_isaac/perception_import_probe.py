"""CPU-only, network-blocked imports. Never constructs or loads models."""

from __future__ import annotations

import importlib
import json
import os
import socket

IMPORTS = (
    "transformers.models.grounding_dino.modeling_grounding_dino",
    "transformers.models.grounding_dino.processing_grounding_dino",
    "sam3.model_builder",
    "sam3.model.sam3_image_processor",
    "cv2", "open3d", "sklearn", "decord", "pycocotools.mask",
)


def blocked_connect(*args, **kwargs):
    raise RuntimeError("Network connections are disabled during the import probe")


def run_probe() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection
    checks = {}
    errors = {}
    try:
        socket.socket.connect = blocked_connect
        socket.socket.connect_ex = blocked_connect
        socket.create_connection = blocked_connect
        for name in IMPORTS:
            try:
                importlib.import_module(name)
                checks[name] = True
            except Exception as exc:
                checks[name] = False
                errors[name] = {"type": type(exc).__name__, "detail": str(exc)[:1200]}
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create
    # A successful import is not proof that weights, authentication or inference work.
    return {"ok": all(checks.values()), "checks": checks, "errors": errors,
            "models_loaded": False, "gpu_visible": False,
            "network_connections_blocked": True, "inference_validated": False,
            "sam3_gated_access_validated": False}


if __name__ == "__main__":
    report = run_probe()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)

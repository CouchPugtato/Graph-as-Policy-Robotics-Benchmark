"""Small native CUDA/Triton probe; no models, inference, or network calls."""

from __future__ import annotations

import importlib.metadata
import json
import socket
import subprocess
import sys

EXPECTED = {
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "triton-windows": "3.3.1.post21",
}


def check_versions() -> dict:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise ValueError(f"Expected pinned CUDA packages {EXPECTED}; found {versions}")
    return versions


def check_idle_gpu() -> dict:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", 8765)) == 0:
            raise ValueError("Stop the Isaac connector first")
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    name, used = result.stdout.strip().split(",")
    if "RTX 5080" not in name or int(used.strip()) > 2048:
        raise ValueError("GPU 0 must be the RTX 5080 with at most 2048 MiB in use; unload models")
    return {"name": name.strip(), "memory_used_before_mib": int(used.strip())}


def run_probe() -> dict:
    versions = check_versions()
    gpu = check_idle_gpu()
    import torch
    import torchvision

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    if "RTX 5080" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Visible CUDA device 0 is not the RTX 5080")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.05, 0)
    free, total = torch.cuda.mem_get_info(0)
    if free < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free VRAM is required for this conservative probe")
    x = torch.arange(1024, device="cuda:0", dtype=torch.float32)
    y = torch.ones_like(x)
    output = torch.empty_like(x)
    expected = x + y
    torch.cuda.synchronize()
    torch_ok = torch.equal(expected, x + 1)
    # Verify torchvision's binary CUDA operation, not just its Python import.
    boxes = torch.tensor([[0., 0., 1., 1.]], device="cuda:0")
    scores = torch.tensor([1.], device="cuda:0")
    nms_ok = torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0]
    import triton_probe_kernel

    triton_probe_kernel.add_kernel[(4,)](x, y, output, 1024, BLOCK_SIZE=256, num_warps=4)
    torch.cuda.synchronize()
    triton_ok = torch.equal(output, expected)
    if not (torch_ok and nms_ok and triton_ok):
        raise RuntimeError("CUDA or Triton result did not match the expected output")
    return {
        "ok": True, "packages": versions, "gpu": gpu,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda_add": torch_ok, "torchvision_cuda_nms": nms_ok,
        "triton_compiled_add": triton_ok, "tensor_elements": 1024,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "total_vram_bytes": total, "models_loaded": False, "network_requests": 0,
        "sam3_validated": False,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(run_probe(), indent=2))
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "detail": str(exc), "models_loaded": False}), file=sys.stderr)
        raise SystemExit(1) from None

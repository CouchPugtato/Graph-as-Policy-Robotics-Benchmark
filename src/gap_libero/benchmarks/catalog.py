"""Pinned upstream environment selections without simulator imports."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
PRO_REVISION = "eafdb809426b13153aa1e4c42d6601844217dfec"
PRO_DATA_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"
PRO_DATASET = "zhouxueyang/LIBERO-Pro"
BENCHMARKS = {
    "libero-object": ("LIBERO", "libero_object", LIBERO_REVISION),
    "libero-spatial": ("LIBERO", "libero_spatial", LIBERO_REVISION),
    "libero-goal": ("LIBERO", "libero_goal", LIBERO_REVISION),
    "libero-10": ("LIBERO", "libero_10", LIBERO_REVISION),
    "libero-90": ("LIBERO", "libero_90", LIBERO_REVISION),
    "libero-pro-object": ("LIBERO-PRO", "libero_object_object", PRO_REVISION),
    "libero-pro-swap": ("LIBERO-PRO", "libero_object_swap", PRO_REVISION),
}


def profile(name, root=ROOT):
    if name not in BENCHMARKS:
        raise ValueError(f"unknown upstream benchmark {name}")
    source_name, suite, revision = BENCHMARKS[name]
    source = root / "cache/upstream" / source_name
    config = root / "cache/benchmarks/config" / source_name.lower()
    return {
        "benchmark": name,
        "suite": suite,
        "source": source,
        "revision": revision,
        "config": config,
        "data": root / "cache/benchmarks/libero-pro-data",
        "output_root": root / "results/benchmarks" / name,
    }

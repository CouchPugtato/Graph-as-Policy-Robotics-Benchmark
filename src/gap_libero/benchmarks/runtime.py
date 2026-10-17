"""Select upstream LIBERO sources without changing custom defaults."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

from .catalog import ROOT, profile


def configure(name, root=ROOT):
    selected = profile(name, root)
    source = selected["source"]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != selected["revision"]:
        raise ValueError("upstream checkout does not match the pinned revision")
    upstream = source / "libero/libero"
    data = selected["data"] if name.startswith("libero-pro-") else upstream
    if not (data / "bddl_files" / selected["suite"]).is_dir():
        raise FileNotFoundError("selected upstream suite files have not been downloaded")
    from gap_libero.cli import configure as configure_runtime

    (source / "datasets").mkdir(parents=True, exist_ok=True)
    configure_runtime(source, selected["config"], data_root=data, allow_existing=True)
    return selected


def activate(name, root=ROOT):
    if "libero.libero" in sys.modules:
        raise RuntimeError("upstream source selection requires a fresh worker")
    selected = configure(name, root)
    # source selection is private to this worker process
    os.environ["LIBERO_CONFIG_PATH"] = str(selected["config"].resolve())
    sys.path.insert(0, str(selected["source"].resolve()))
    importlib.invalidate_caches()
    package = importlib.import_module("libero.libero")
    if not Path(package.__file__).resolve().is_relative_to(selected["source"].resolve()):
        raise RuntimeError("another LIBERO installation shadowed the selected upstream source")
    return selected

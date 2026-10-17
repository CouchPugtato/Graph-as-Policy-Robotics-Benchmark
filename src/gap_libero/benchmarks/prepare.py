"""Prepare upstream assets for the shared LIBERO runtime."""

import argparse
import json
import os
import shutil
import subprocess

from .catalog import BENCHMARKS, PRO_DATA_REVISION, PRO_DATASET, ROOT, profile
from .runtime import configure


def prepare(name):
    selected = profile(name)
    if name.startswith("libero-pro-"):
        source = selected["source"]
        if not source.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    "--no-checkout",
                    "https://github.com/Zxy-MLlab/LIBERO-PRO.git",
                    str(source),
                ],
                check=True,
                timeout=180,
            )
            subprocess.run(
                ["git", "-C", str(source), "fetch", "--depth", "1", "origin", selected["revision"]],
                check=True,
                timeout=180,
            )
            subprocess.run(
                ["git", "-C", str(source), "checkout", "--detach", selected["revision"]],
                check=True,
                timeout=180,
            )
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != selected["revision"]:
            raise ValueError("existing LIBERO-Pro checkout differs and was not changed")
        subprocess.run(
            ["git", "-C", str(source), "sparse-checkout", "set", "libero/libero"],
            check=True,
            timeout=300,
        )
        cli = ROOT / ".venv-perception/Scripts/hf.exe"
        executable = str(cli) if cli.is_file() else shutil.which("hf")
        if not executable:
            raise FileNotFoundError("hf CLI is required to download official task and state files")
        environment = os.environ.copy()
        environment.pop("HF_HUB_OFFLINE", None)
        # only evaluation definitions and initial states are downloaded
        subprocess.run(
            [
                executable,
                "download",
                PRO_DATASET,
                "--repo-type",
                "dataset",
                "--revision",
                PRO_DATA_REVISION,
                "--local-dir",
                str(selected["data"]),
                "--max-workers",
                "1",
                "--include",
                f"bddl_files/{selected['suite']}/*",
                f"init_files/{selected['suite']}/*",
            ],
            env=environment,
            check=True,
            timeout=300,
        )
    selected = configure(name)
    return {
        "ok": True,
        "benchmark": name,
        "source_revision": selected["revision"],
        "config": str(selected["config"]),
        "simulator_started": False,
        "models_loaded": False,
        "custom_scenarios_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="libero-object")
    args = parser.parse_args()
    print(json.dumps(prepare(args.benchmark), indent=2))


if __name__ == "__main__":
    main()

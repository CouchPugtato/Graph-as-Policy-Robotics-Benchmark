"""Project command-line entry point."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import httpx
import typer

from gap_isaac.config import load_task, validate_project
from gap_isaac.connector_smoke import run_connector_smoke
from gap_isaac.diagnostics import run_checks
from gap_isaac.fake_backend import FakeBackend
from gap_isaac.gateway import create_app
from gap_isaac.manifest import build_manifest, write_manifest
from gap_isaac.observation_bundle import ObservationBundle, capture_observation
from gap_isaac.reset import build_episode_reset_plan
from gap_isaac.sampling import sample_instance
from gap_isaac.statistics import wilson_interval

app = typer.Typer(no_args_is_help=True)
ROOT = Path(__file__).resolve().parents[2]


@app.command()
def validate(
    root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = ROOT,
) -> None:
    """Validate all project configuration files."""

    for summary in validate_project(root):
        typer.echo(f"OK  {summary}")


@app.command()
def sample(
    seed: Annotated[int, typer.Option()] = 0,
    task_path: Annotated[Path, typer.Option()] = ROOT / "configs/tasks/grocery_fulfillment.yaml",
) -> None:
    """Print a deterministic task instance as JSON."""

    instance = sample_instance(load_task(task_path), seed)
    typer.echo(json.dumps(instance.as_dict(), indent=2, sort_keys=True))


@app.command("plan-reset")
def plan_reset(
    seed: Annotated[int, typer.Option()] = 0,
    target: Annotated[str | None, typer.Option()] = None,
    task_path: Annotated[Path, typer.Option()] = ROOT / "configs/tasks/grocery_fulfillment.yaml",
) -> None:
    """Print the deterministic simulator reset state for one episode."""

    plan = build_episode_reset_plan(load_task(task_path), seed=seed, target_id=target)
    typer.echo(json.dumps(plan.as_dict(), indent=2, sort_keys=True))


@app.command()
def doctor() -> None:
    """Report environment readiness without modifying the host."""

    for check in run_checks():
        typer.echo(f"{check.status.upper():7} {check.name:16} {check.detail}")


@app.command("smoke-connector")
def smoke_connector(
    base_url: Annotated[str, typer.Option()] = "http://127.0.0.1:8765",
) -> None:
    """Validate observations and prove that motion remains disabled."""

    token = os.environ.get("GAP_ISAAC_CONNECTOR_TOKEN")
    if not token:
        raise typer.BadParameter("GAP_ISAAC_CONNECTOR_TOKEN must be set")
    try:
        result = run_connector_smoke(base_url, token)
    except httpx.ConnectError:
        typer.echo(
            f"ERROR: connector unreachable at {base_url}. "
            "Start the read-only connector and leave its terminal running, then retry.",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except httpx.TimeoutException:
        typer.echo(f"ERROR: connector at {base_url} timed out.", err=True)
        raise typer.Exit(code=1) from None
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status == 401:
            detail = "the token does not match the running connector"
        else:
            detail = f"HTTP {status} from {error.request.url}"
        typer.echo(f"ERROR: connector smoke test failed: {detail}.", err=True)
        raise typer.Exit(code=1) from None
    except RuntimeError as error:
        typer.echo(f"ERROR: connector smoke test failed: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(result.as_dict(), indent=2, sort_keys=True))


@app.command("capture-observation")
def capture_rgbd(
    output: Annotated[Path | None, typer.Option()] = None,
    base_url: Annotated[str, typer.Option()] = "http://127.0.0.1:8765",
) -> None:
    """Save one verified RGB-D frame and preview without running any models."""
    token = os.environ.get("GAP_ISAAC_CONNECTOR_TOKEN")
    if not token:
        raise typer.BadParameter("GAP_ISAAC_CONNECTOR_TOKEN must match the running connector")
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "results/observations" / stamp
    try:
        if output.exists():
            raise ValueError(f"output already exists; choose a new directory: {output}")
        bundle = capture_observation(base_url, token)
        bundle.save(output)
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
        if isinstance(error, httpx.ConnectError):
            detail = "connector unreachable; start the read-only connector first"
        elif isinstance(error, httpx.HTTPStatusError):
            detail = f"connector returned HTTP {error.response.status_code}"
        elif isinstance(error, httpx.HTTPError):
            detail = "connector request failed or timed out"
        else:
            detail = str(error)
        typer.echo(f"ERROR: observation capture failed: {detail}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        json.dumps(
            {
                "ok": True,
                "sequence": bundle.observation.sequence,
                "bundle": str(output.resolve()),
                "preview": str((output / "rgb.png").resolve()),
                "depth": bundle.depth_summary(),
                "models_loaded": False,
                "motion_enabled": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("inspect-observation")
def inspect_rgbd(
    bundle_path: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    pixel_x: Annotated[int | None, typer.Option()] = None,
    pixel_y: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Check a saved frame; optionally deproject one pixel (not object detection)."""
    try:
        if (pixel_x is None) != (pixel_y is None):
            raise ValueError("supply both --pixel-x and --pixel-y")
        bundle = ObservationBundle.load(bundle_path)
        result = {
            "ok": True,
            "sequence": bundle.observation.sequence,
            "captured_at": bundle.observation.captured_at.isoformat(),
            "camera": bundle.observation.camera.model_dump(),
            "world_frame": bundle.observation.world_frame,
            "depth": bundle.depth_summary(),
            "geometry_source": "connector-rgbd-only",
        }
        if pixel_x is not None and pixel_y is not None:
            result["pixel_world_position_m"] = bundle.pixel_to_world(pixel_x, pixel_y)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        typer.echo(f"ERROR: observation inspection failed: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("make-manifest")
def make_manifest(
    output: Annotated[Path, typer.Option()] = ROOT / "results/manifests/environment.json",
) -> None:
    """Record environment and checked-in configuration provenance."""

    files = list((ROOT / "configs").rglob("*.yaml"))
    write_manifest(output, build_manifest(ROOT, files))
    typer.echo(str(output))


@app.command()
def interval(successes: int, trials: int) -> None:
    """Print a Wilson 95% interval for a success count."""

    result = wilson_interval(successes, trials)
    typer.echo(
        f"{result.successes}/{result.trials} = {result.rate:.3f} "
        f"(95% CI {result.lower:.3f}-{result.upper:.3f})"
    )


@app.command("serve-fake")
def serve_fake(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
    motion_enabled: Annotated[bool, typer.Option()] = False,
) -> None:
    """Serve the connector contract with a fake backend."""

    import uvicorn

    token = os.environ.get("GAP_ISAAC_CONNECTOR_TOKEN")
    if not token:
        raise typer.BadParameter("GAP_ISAAC_CONNECTOR_TOKEN must be set")
    uvicorn.run(create_app(FakeBackend(motion_enabled=motion_enabled), token), host=host, port=port)


if __name__ == "__main__":
    app()

import hashlib

import httpx
import pytest
from typer.testing import CliRunner

from gap_isaac import cli
from gap_isaac.connector_smoke import _same_origin, _validate_asset

runner = CliRunner()


def test_asset_origin_must_match_connector() -> None:
    assert _same_origin(
        "http://127.0.0.1:8765",
        "http://127.0.0.1:8765/v1/assets/1/rgb",
    )
    assert not _same_origin(
        "http://127.0.0.1:8765",
        "http://example.com/v1/assets/1/rgb",
    )


def test_asset_size_and_hash_are_verified() -> None:
    body = b"pixels"
    digest = hashlib.sha256(body).hexdigest()
    _validate_asset(body, len(body), digest, "RGB")

    with pytest.raises(RuntimeError, match="byte count"):
        _validate_asset(body, len(body) + 1, digest, "RGB")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _validate_asset(body, len(body), "0" * 64, "RGB")


def test_cli_reports_unreachable_connector_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GAP_ISAAC_CONNECTOR_TOKEN", "test-token")

    def refuse_connection(base_url: str, token: str) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(cli, "run_connector_smoke", refuse_connection)

    result = runner.invoke(cli.app, ["smoke-connector"])

    assert result.exit_code == 1
    assert "connector unreachable at http://127.0.0.1:8765" in result.output
    assert "Traceback" not in result.output

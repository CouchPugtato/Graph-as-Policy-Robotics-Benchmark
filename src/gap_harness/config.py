"""Explicit project settings without loading shell expansions or external endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_REASONING_EFFORT",
            "GAP_CODEX_PROXY_KEY"}
PLATFORM_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def load_environment(path=ROOT / ".env"):
    path = Path(path)
    if not path.is_file():
        return
    # shell settings take precedence and unrelated variables remain untouched
    for key, value in dotenv_values(path, interpolate=False).items():
        if key in ENV_KEYS and value is not None:
            os.environ.setdefault(key, value)


def endpoint(value=None):
    value = value if value is not None else os.environ.get("OPENAI_BASE_URL", PLATFORM_URL)
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("invalid API endpoint port") from None
    if (parsed.username is not None or parsed.password is not None or parsed.query
            or parsed.fragment or parsed.path != "/v1"):
        raise ValueError("API base URL must end in /v1 without credentials query or fragment")
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    official = (parsed.hostname == "api.openai.com" and parsed.scheme == "https"
                and port in {None, 443})
    if not official and not (local and parsed.scheme in {"http", "https"}):
        raise ValueError("only the official HTTPS API or a loopback proxy is allowed")
    return value, local

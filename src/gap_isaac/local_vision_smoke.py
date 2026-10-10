"""One bounded, local-only image request; runnable without project dependencies."""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import socket
import struct
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validated_endpoint(value: str) -> str:
    url = urlsplit(value)
    try:
        local = ipaddress.ip_address(url.hostname).is_loopback
        port = url.port
    except (ValueError, TypeError):
        local, port = False, None
    if (
        not local
        or url.scheme not in {"http", "https"}
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.path.rstrip("/") != "/v1"
        or port == 0
    ):
        raise ValueError("Use a numeric loopback API base, such as http://127.0.0.1:1234/v1")
    return value.rstrip("/") + "/chat/completions"


def image_payload(path: Path) -> tuple[bytes, int, int]:
    with path.open("rb") as source:
        data = source.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 2 MiB smoke-test limit")
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("Use the rgb.png saved by capture-observation")
    width, height = struct.unpack(">II", data[16:24])
    if not (0 < width <= 640 and 0 < height <= 480):
        raise ValueError("Image dimensions must not exceed 640 by 480")
    return data, width, height


def run_test(image: Path, model: str, base_url: str) -> dict:
    endpoint = validated_endpoint(base_url)
    if not model.strip() or "embedding" in model.lower() or "embed-text" in model.lower():
        raise ValueError("Specify the loaded vision model, not an embedding model")
    # A connector is a definite sign that Isaac is still running. Other Isaac
    # sessions must also be stopped by the operator before invoking this test.
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", 8765)) == 0:
            raise ValueError("Stop the Isaac connector before running local vision")
    data, width, height = image_payload(image)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": (
                "Briefly list the small colored objects and any open container visible, "
                "with approximate image locations. Say uncertain when unclear. "
                "Do not invent product names."
            )},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")
            }},
        ]}],
        "max_tokens": 256,
        "temperature": 0,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("GAP_VLM_API_KEY"):
        headers["Authorization"] = "Bearer " + token
    request = Request(endpoint, json.dumps(payload).encode(), headers, method="POST")
    # No inherited proxies, redirects, retries, or cloud fallback.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=45) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Server response exceeds the smoke-test limit")
    result = json.loads(raw)
    choice = result["choices"][0]
    reply = choice["message"]["content"]
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("Server did not return a nonempty text reply")
    return {
        "api_ok": True,
        "semantic_validated": False,
        "model": model,
        "image": str(image.resolve()),
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
        "reply": reply,
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage"),
        "motion_enabled": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args()
    try:
        report = run_test(args.image, args.model, args.base_url)
    except HTTPError as exc:
        print(f"ERROR: local vision HTTP {exc.code}; check LM Studio server logs", file=sys.stderr)
        return 1
    except URLError:
        print("ERROR: local vision server unreachable or request timed out", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"ERROR: local vision test failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

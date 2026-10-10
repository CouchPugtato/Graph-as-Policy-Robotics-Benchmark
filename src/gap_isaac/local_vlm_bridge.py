"""Process-scoped local-only transport for the pinned upstream VLM tool bundle.

No model imports or downloads. Install this bridge after importing the upstream
VLM module and before registering its tools in an offline perception worker.
This confines VLM requests only; it is not a general network sandbox.
"""

from __future__ import annotations

import json
import os
from urllib.request import ProxyHandler, Request, build_opener

from gap_isaac.local_vision_smoke import MAX_RESPONSE_BYTES, NoRedirect, validated_endpoint


def install_local_vlm_bridge(upstream, *, max_calls: int = 16) -> dict:
    """Preserve upstream image encoding and query tools, replace hosted dispatch.

    Requires explicit configuration; per-call model/provider overrides cannot
    escape it. Neither OPENROUTER_API_KEY nor GAP_LLM_* is inherited.
    """
    if not 1 <= max_calls <= 16:
        raise ValueError("Local VLM call budget must be between 1 and 16")
    if os.environ.get("GAP_VLM_PROVIDER") != "openrouter":
        raise ValueError("GAP_VLM_PROVIDER must explicitly select openrouter transport")
    base_url = os.environ.get("GAP_VLM_BASE_URL", "")
    endpoint = validated_endpoint(base_url)
    model = os.environ.get("GAP_VLM_MODEL", "").strip()
    if not model or "embed" in model.lower():
        raise ValueError("GAP_VLM_MODEL must explicitly name the loaded vision model")
    token = os.environ.get("GAP_VLM_API_KEY", "")
    # Require the audited interface; don't silently adapt unrelated revisions.
    if not all(
        callable(getattr(upstream, name, None))
        for name in ("_png_b64", "_query", "_gather_images")
    ):
        raise ValueError("Upstream VLM module does not expose the audited interface")
    state = {"calls": 0, "max_calls": max_calls, "model": model, "local_only": True}

    def query_local(prompt, images, requested_model):
        if requested_model is not None and requested_model != model:
            raise ValueError("Per-call model override is not allowed")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192:
            raise ValueError("VLM prompt must be nonempty and at most 8192 characters")
        if not 1 <= len(images) <= 2:
            raise ValueError("Local perception queries require one or two image crops")
        content = [{"type": "text", "text": prompt}]
        for image in images:
            # The canonical A/B sheet is 792 by 396, bigger than the overview.
            shape = getattr(image, "shape", ())
            if len(shape) != 3 or not (0 < shape[0] <= 1024 and 0 < shape[1] <= 1024):
                raise ValueError("VLM crop exceeds the 1024 by 1024 budget")
            encoded = upstream._png_b64(image)
            if len(encoded) > 3 * 1024 * 1024:
                raise ValueError("Encoded VLM crop exceeds the byte budget")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + encoded,
            }})
        if state["calls"] >= max_calls:
            raise ValueError("Local VLM call budget exhausted; no retry or fallback")
        state["calls"] += 1
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        payload = {
            "model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 128, "temperature": 0, "stream": False,
        }
        request = Request(endpoint, json.dumps(payload).encode(), headers, method="POST")
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=45) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("VLM response exceeds the byte budget")
        choice = json.loads(raw)["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("VLM reply was truncated; refusing an ambiguous crop decision")
        text = choice["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Local VLM returned no text")
        return text

    def dispatch(prompt, image, images, provider, requested_model):
        if provider is not None and provider != "openrouter":
            raise ValueError("Hosted provider overrides are disabled")
        arrays = upstream._gather_images(image, images)
        return query_local(prompt, arrays, requested_model)

    # Tool functions resolve these globals when invoked. No source file edits
    # to third-party code, no global patch to httpx, and no default-cloud route.
    upstream._query = dispatch
    upstream._query_openrouter = query_local
    upstream._PROVIDERS = {"openrouter": query_local}
    return state

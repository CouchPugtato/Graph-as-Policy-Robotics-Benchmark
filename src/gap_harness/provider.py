"""Small Responses API transport without an agent framework."""

from __future__ import annotations

import json
import os

import httpx

from gap_harness.config import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, REASONING_EFFORTS, endpoint


class OpenAIResponses:
    def __init__(self, model=DEFAULT_MODEL, max_output_tokens=4096, timeout_s=120,
                 transport=None, base_url=None, reasoning_effort=DEFAULT_REASONING_EFFORT):
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("unsupported reasoning effort")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.transport = transport
        self.base_url, self.local_proxy = endpoint(base_url)

    def public_settings(self):
        return {"base_url": self.base_url,
                "authentication": "codex-local-proxy" if self.local_proxy else "platform-api",
                "model": self.model, "reasoning_effort": self.reasoning_effort}

    def complete(self, instructions, context, output_model):
        # a platform credential must never be forwarded to a local compatibility server
        key_name = "GAP_CODEX_PROXY_KEY" if self.local_proxy else "OPENAI_API_KEY"
        key = os.environ.get(key_name, "").strip()
        if not key and not self.local_proxy:
            raise ValueError("set OPENAI_API_KEY privately before enabling live API calls")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        # explicit endpoints still bypass environment proxies and redirects
        with httpx.Client(timeout=self.timeout_s, trust_env=False, follow_redirects=False,
                          transport=self.transport) as client:
            response = client.post(self.base_url + "/responses",
                headers=headers,
                json={"model": self.model, "instructions": instructions,
                      "input": json.dumps(context, allow_nan=False), "store": False,
                      "max_output_tokens": self.max_output_tokens,
                      "reasoning": {"effort": self.reasoning_effort},
                      "text": {"format": {"type": "json_schema", "name": output_model.__name__,
                                          "strict": True,
                                          "schema": output_model.model_json_schema()}}})
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI request failed with HTTP {response.status_code}")
        data = response.json()
        text = "".join(part.get("text", "") for item in data.get("output", [])
                       if item.get("type") == "message" for part in item.get("content", [])
                       if part.get("type") == "output_text")
        return {"id": data.get("id"), "model": data.get("model", self.model),
                "status": data.get("status"), "usage": data.get("usage"), "text": text}

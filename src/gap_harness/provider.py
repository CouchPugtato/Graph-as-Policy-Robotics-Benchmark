"""Small Responses API transport without an agent framework."""

from __future__ import annotations

import json
import os

import httpx


class OpenAIResponses:
    def __init__(self, model="gpt-6-astra", max_output_tokens=4096, timeout_s=120,
                 transport=None):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.transport = transport

    def complete(self, instructions, context, output_model):
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("set OPENAI_API_KEY privately before enabling live API calls")
        # do not inherit proxies or alternate endpoints for credential bearing requests
        with httpx.Client(timeout=self.timeout_s, trust_env=False, follow_redirects=False,
                          transport=self.transport) as client:
            response = client.post("https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": self.model, "instructions": instructions,
                      "input": json.dumps(context, allow_nan=False), "store": False,
                      "max_output_tokens": self.max_output_tokens,
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

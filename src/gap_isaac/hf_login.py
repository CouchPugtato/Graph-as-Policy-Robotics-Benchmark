"""Explicit hidden-input HF login using a direct, official-endpoint connection."""

from __future__ import annotations

import getpass
import json
import logging
import os


def clean_token(value: str) -> str:
    token = value.strip()
    if not token.startswith("hf_") or any(c.isspace() for c in token):
        raise ValueError("Expected one nonempty hf_ personal access token")
    return token


def main() -> int:
    # Set before SDK import: its constants read the environment at import time.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    # Child-process only; don't let stale environment credentials override saved login.
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    try:
        import requests
        from huggingface_hub import configure_http_backend, get_token, login

        class DirectSession(requests.Session):
            def __init__(self):
                super().__init__()
                self.trust_env = False

            def request(self, method, url, **kwargs):
                # Login needs only whoami; credentials cannot follow redirects.
                if url != "https://huggingface.co/api/whoami-v2":
                    raise ValueError("Unexpected endpoint in authentication-only helper")
                kwargs["allow_redirects"] = False
                kwargs["timeout"] = 15
                return super().request(method, url, **kwargs)

        configure_http_backend(backend_factory=DirectSession)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        token = clean_token(getpass.getpass("Paste HF Read token (hidden): "))
        login(token=token, add_to_git_credential=False)
        if get_token() != token:
            raise RuntimeError("Saved credential verification failed")
    except Exception as exc:
        response = getattr(exc, "response", None)
        # No exception body, headers, token value, or stored token name in output.
        print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                          "http_status": getattr(response, "status_code", None)}))
        return 1
    print(json.dumps({"ok": True, "token_saved": True,
                      "git_credential_added": False, "weights_downloaded": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

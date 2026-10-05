"""
core/anthropic_http.py — Drop-in replacement para anthropic.Anthropic

Usa urllib directamente para evitar la incompatibilidad de pydantic 2.13.x
con Python 3.14t (free-threading build).

Interfaz replicada:
    client = Anthropic(api_key=...)
    resp   = client.messages.create(model=..., max_tokens=..., messages=[...])
    resp.content[0].text
    resp.usage.input_tokens
    resp.usage.output_tokens
"""

import json
import urllib.request
import urllib.error

ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class _TextBlock:
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Response:
    def __init__(self, data: dict):
        self.content = [
            _TextBlock(b["text"])
            for b in data.get("content", [])
            if b.get("type") == "text"
        ]
        usage = data.get("usage", {})
        self.usage = _Usage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        self.model = data.get("model", "")
        self.stop_reason = data.get("stop_reason", "")


class _Messages:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def create(self, model: str, max_tokens: int, messages: list,
               system: str | None = None, **kwargs) -> _Response:
        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ANTHROPIC_API,
            data=data,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return _Response(json.loads(resp.read().decode("utf-8")))


class Anthropic:
    """Minimal Anthropic client sin dependencia de pydantic."""

    def __init__(self, api_key: str):
        self.messages = _Messages(api_key)

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .string_utils import to_lower, trim


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient:
    def __init__(self) -> None:
        self._backend = "none"
        self._ollama_host = os.getenv("ALICE_OLLAMA_HOST", "http://127.0.0.1:11434")
        self._ollama_model = os.getenv("ALICE_OLLAMA_MODEL", "qwen2.5:3b")
        self._openai_model = os.getenv("ALICE_OPENAI_MODEL", "gpt-4o-mini")

        desired = to_lower(trim(os.getenv("ALICE_LLM_BACKEND", "auto")))
        if desired not in {"auto", "ollama", "openai", "none"}:
            desired = "auto"

        if desired in {"auto", "ollama"} and self._ollama_available():
            self._backend = "ollama"
            return
        if desired == "ollama":
            self._backend = "none"
            return

        if desired in {"auto", "openai"} and self._openai_available():
            self._backend = "openai"
            return

        self._backend = "none"

    def backend(self) -> str:
        return self._backend

    def available(self) -> bool:
        return self._backend in {"ollama", "openai"}

    def _ollama_available(self) -> bool:
        url = f"{self._ollama_host.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return 200 <= response.status < 400
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False

    @staticmethod
    def _openai_available() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def _ollama_chat(self, messages: list[ChatMessage], temperature: float) -> str:
        payload = {
            "model": self._ollama_model,
            "messages": [message.__dict__ for message in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        url = f"{self._ollama_host.rstrip('/')}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            return ""

        message = data.get("message", {}) if isinstance(data, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return trim(content)

    def _openai_chat(self, messages: list[ChatMessage], temperature: float) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ""

        payload = {
            "model": self._openai_model,
            "messages": [message.__dict__ for message in messages],
            "temperature": temperature,
            "max_tokens": 180,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            return ""

        if not isinstance(data, dict):
            return ""
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message", {})
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        return trim(content) if isinstance(content, str) else ""

    def chat(self, messages: list[ChatMessage], temperature: float = 0.3) -> str:
        if self._backend == "ollama":
            return self._ollama_chat(messages, temperature)
        if self._backend == "openai":
            return self._openai_chat(messages, temperature)
        return ""

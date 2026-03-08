from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


class LLMClient:
    def __init__(self) -> None:
        self._backend = "none"
        self._ollama_host = os.getenv("ALICE_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self._ollama_cmd = os.getenv("ALICE_OLLAMA_CMD", "ollama").strip() or "ollama"
        self._ollama_chat_model = os.getenv("ALICE_OLLAMA_MODEL", "qwen2.5:3b")
        self._ollama_intent_model = os.getenv("ALICE_OLLAMA_INTENT_MODEL", self._ollama_chat_model)
        self._openai_chat_model = os.getenv("ALICE_OPENAI_MODEL", "gpt-4o-mini")
        self._openai_intent_model = os.getenv("ALICE_INTENT_MODEL", self._openai_chat_model)
        self._openai_client = None
        self._ollama_process: subprocess.Popen[bytes] | None = None

        desired = os.getenv("ALICE_LLM_BACKEND", "auto").strip().lower()
        if desired not in {"auto", "ollama", "openai", "none"}:
            desired = "auto"

        if desired in {"auto", "ollama"}:
            if self._ollama_available():
                self._backend = "ollama"
                return
            if self._start_ollama_server() and self._wait_for_ollama():
                self._backend = "ollama"
                return
        if desired == "ollama":
            self._backend = "none"
            return

        if desired in {"auto", "openai"} and self._setup_openai():
            self._backend = "openai"
            return

        self._backend = "none"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def available(self) -> bool:
        return self._backend in {"ollama", "openai"}

    def _resolve_ollama_command(self) -> str | None:
        if os.path.isabs(self._ollama_cmd) and os.path.exists(self._ollama_cmd):
            return self._ollama_cmd

        if os.sep in self._ollama_cmd:
            candidate = os.path.expanduser(self._ollama_cmd)
            if os.path.exists(candidate):
                return candidate

        found = shutil.which(self._ollama_cmd)
        if found:
            return found

        mac_bundle_bin = "/Applications/Ollama.app/Contents/Resources/ollama"
        if os.path.exists(mac_bundle_bin):
            return mac_bundle_bin

        return None

    def _ollama_available(self) -> bool:
        request = urllib.request.Request(f"{self._ollama_host}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=1.5) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def _wait_for_ollama(self, timeout_seconds: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._ollama_available():
                return True
            time.sleep(0.25)
        return False

    def _stop_started_ollama(self) -> None:
        process = self._ollama_process
        if process is None:
            return
        self._ollama_process = None
        try:
            if process.poll() is None:
                process.terminate()
        except Exception:
            pass

    def _start_ollama_server(self) -> bool:
        command = self._resolve_ollama_command()
        if command is None:
            return False

        try:
            process = subprocess.Popen(
                [command, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return False

        time.sleep(0.2)
        if process.poll() is not None:
            return False

        self._ollama_process = process
        atexit.register(self._stop_started_ollama)
        return True

    def _setup_openai(self) -> bool:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False
        try:
            from openai import OpenAI
        except Exception:
            return False

        try:
            self._openai_client = OpenAI(api_key=api_key)
        except Exception:
            self._openai_client = None
            return False
        return True

    def _ollama_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        req = urllib.request.Request(
            f"{self._ollama_host}/api/chat",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        message = data.get("message", {})
        content = message.get("content", "")
        return str(content).strip()

    def _openai_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        if self._openai_client is None:
            return ""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 180,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._openai_client.chat.completions.create(**kwargs)
        return str((response.choices[0].message.content or "").strip())

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> str:
        if self._backend == "ollama":
            try:
                return self._ollama_chat(
                    model=self._ollama_chat_model,
                    messages=messages,
                    temperature=temperature,
                )
            except Exception:
                return ""
        if self._backend == "openai":
            try:
                return self._openai_chat(
                    model=self._openai_chat_model,
                    messages=messages,
                    temperature=temperature,
                )
            except Exception:
                return ""
        return ""

    def chat_json(self, *, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        content = ""
        if self._backend == "ollama":
            try:
                content = self._ollama_chat(
                    model=self._ollama_intent_model,
                    messages=messages,
                    temperature=0.0,
                    json_mode=True,
                )
            except Exception:
                return None
        elif self._backend == "openai":
            try:
                content = self._openai_chat(
                    model=self._openai_intent_model,
                    messages=messages,
                    temperature=0.0,
                    json_mode=True,
                )
            except Exception:
                return None
        else:
            return None

        try:
            return json.loads(content)
        except Exception:
            return None

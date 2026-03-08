from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from env_utils import load_project_env


class AliceBrain:
    def __init__(self) -> None:
        load_project_env(Path(__file__).resolve().parent.parent)

        self._history: list[tuple[str, str]] = []
        self._client = None
        self._model = os.getenv("ALICE_OPENAI_MODEL", "gpt-4o-mini")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return

        try:
            from openai import OpenAI
        except ImportError:
            return

        self._client = OpenAI(api_key=api_key)

    @property
    def using_openai(self) -> bool:
        return self._client is not None

    def _fallback_reply(self, text: str) -> str:
        t = text.strip().lower()

        if any(k in t for k in {"hello", "hi", "hey"}):
            return "Hello Fabio. I am here with you."
        if "how are you" in t:
            return "I am doing well and ready to help."
        if "what can you do" in t or "help" in t:
            return (
                "I can chat with you and run local commands like listing files "
                "or running scripts after confirmation."
            )
        if "time" in t:
            return f"It is {datetime.now().strftime('%I:%M %p')}."
        if "date" in t or "day" in t:
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."

        return (
            "I heard you. I can chat, and I can also run files or manage folders "
            "when you ask with the wake word Alice."
        )

    def _openai_reply(self, text: str) -> str:
        if self._client is None:
            return self._fallback_reply(text)

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are Alice, a concise voice assistant for Fabio. "
                    "Be natural, useful, and brief. Do not claim actions you did not perform."
                ),
            }
        ]

        for user_msg, assistant_msg in self._history[-4:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": text})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.4,
                max_tokens=140,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                return self._fallback_reply(text)
            return answer
        except Exception:
            return self._fallback_reply(text)

    def reply(self, text: str) -> str:
        answer = self._openai_reply(text)
        self._history.append((text, answer))
        if len(self._history) > 12:
            self._history = self._history[-12:]
        return answer

from __future__ import annotations

from .llm_client import ChatMessage, LLMClient
from .string_utils import format_clock_time, format_long_date, join, normalize_text, trim


class AliceBrain:
    def __init__(self) -> None:
        self._history: list[tuple[str, str]] = []
        self._llm = LLMClient()

    def using_llm(self) -> bool:
        return self._llm.available()

    def llm_backend(self) -> str:
        return self._llm.backend()

    def _fallback_reply(self, text: str) -> str:
        lowered = normalize_text(text)
        if "hello" in lowered or lowered in {"hi", "hey"}:
            return "Hello Fabio. I am here with you. How is your day going?"
        if "sorry" in lowered or "never mind" in lowered:
            return "No problem at all. We can keep going."
        if "how are you" in lowered:
            return "I am doing well and ready to help."
        if "what can you do" in lowered or lowered == "help":
            return "I can chat with you and run local commands like listing files or running scripts with confirmation."
        if "time" in lowered:
            return f"It is {format_clock_time()}."
        if "date" in lowered or "day" in lowered:
            return f"Today is {format_long_date()}."
        return "I hear you. I can chat naturally, and I can also run files or manage folders when asked."

    def reply(self, text: str, memories: list[str] | None = None) -> str:
        memories = memories or []

        if not self._llm.available():
            answer = self._fallback_reply(text)
            self._history.append((text, answer))
            self._history = self._history[-12:]
            return answer

        messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=(
                    "You are Alice, a concise voice assistant for Fabio. "
                    "Be natural, useful, and brief. Do not claim actions you did not perform."
                ),
            )
        ]

        if memories:
            limited = memories[:6]
            messages.append(ChatMessage(role="system", content=f"Relevant long-term memory: {join(limited, ' ; ')}"))

        for user_text, assistant_text in self._history[-4:]:
            messages.append(ChatMessage(role="user", content=user_text))
            messages.append(ChatMessage(role="assistant", content=assistant_text))
        messages.append(ChatMessage(role="user", content=text))

        answer = trim(self._llm.chat(messages, 0.4))
        if not answer:
            answer = self._fallback_reply(text)

        self._history.append((text, answer))
        self._history = self._history[-12:]
        return answer

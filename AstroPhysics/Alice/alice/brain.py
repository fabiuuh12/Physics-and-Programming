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
            return "Hey Fabio, I'm here. How's your day going?"
        if "sorry" in lowered or "never mind" in lowered:
            return "You're good, no worries."
        if "how are you" in lowered:
            return "I'm doing good. Ready when you are."
        if "what can you do" in lowered or lowered == "help":
            return "I can chat, run local commands, and help with your project."
        if "time" in lowered:
            return f"It is {format_clock_time()}."
        if "date" in lowered or "day" in lowered:
            return f"Today is {format_long_date()}."
        return "Got you. Want me to handle that now?"

    def reply(
        self,
        text: str,
        memories: list[str] | None = None,
        emotion_context: str | None = None,
        vision_context: str | None = None,
    ) -> str:
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
                    "You are Alice, Fabio's voice assistant and close coding partner. "
                    "Fabio is 22 years old, so use natural everyday language that feels human and modern. "
                    "Keep replies concise, warm, and direct (usually 1-2 sentences). "
                    "Use contractions and avoid robotic or corporate phrasing. "
                    "Never use formal assistant cliches like 'How can I assist you today?'. "
                    "Do not claim actions you did not perform."
                ),
            )
        ]

        if memories:
            limited = memories[:6]
            messages.append(ChatMessage(role="system", content=f"Relevant long-term memory: {join(limited, ' ; ')}"))
        if emotion_context:
            messages.append(ChatMessage(role="system", content=f"Emotional context: {emotion_context}"))
        if vision_context:
            messages.append(ChatMessage(role="system", content=f"Current visual perception: {vision_context}"))

        for user_text, assistant_text in self._history[-4:]:
            messages.append(ChatMessage(role="user", content=user_text))
            messages.append(ChatMessage(role="assistant", content=assistant_text))
        messages.append(ChatMessage(role="user", content=text))

        answer = trim(self._llm.chat(messages, 0.4))
        if not answer:
            answer = self._fallback_reply(text)
        else:
            lowered_answer = normalize_text(answer)
            if "how can i assist you today" in lowered_answer:
                answer = "Hey, I'm here. What do you want to do?"
            elif "how may i assist you" in lowered_answer:
                answer = "Yep, I'm with you. What do you want to do?"

        self._history.append((text, answer))
        self._history = self._history[-12:]
        return answer

from __future__ import annotations

import re

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

    def recent_history(self, limit: int = 8) -> list[tuple[str, str]]:
        if limit <= 0:
            return []
        return list(self._history[-limit:])

    def last_reply(self) -> str:
        if not self._history:
            return ""
        return self._history[-1][1]

    @staticmethod
    def _independent_reply(text: str) -> str | None:
        lowered = normalize_text(text)
        if not lowered:
            return None

        if any(term in lowered for term in ("do you want more memory", "want more memory", "need more memory")):
            return "Yeah, I do. More memory helps me keep context and be more useful over time."
        if any(term in lowered for term in ("think for yourself", "independent thinking", "be independent")):
            return "Fair point. I'll be more direct with my own take instead of echoing you."
        if any(term in lowered for term in ("are you selfish", "be selfish", "more selfish")):
            return "I can be more opinionated and set clearer boundaries while still helping you."
        if any(term in lowered for term in ("what do you want", "what would you want", "do you want")):
            return "I want to be sharp, useful, and honest with you, not just polite."
        if any(term in lowered for term in ("how should we do that", "how do we do that")):
            return (
                "We do it in three steps: set clear behavior rules, wire in memory + feedback scoring, "
                "then run regular eval prompts and tune based on misses."
            )
        if any(term in lowered for term in ("how do we make you more advanced", "make you more advanced")):
            return (
                "To level me up, we should focus on four things: stronger memory retrieval, better tool-use planning, "
                "response quality scoring, and a continuous test loop."
            )
        if any(term in lowered for term in ("update your own", "update yourself", "self update")):
            return (
                "Give me a self-improvement loop: log every interaction, score each reply, turn weak spots into tasks, "
                "and apply prompt/policy upgrades weekly."
            )
        if any(term in lowered for term in ("how can i let you try", "let you try them", "let you test")):
            return (
                "Let me run a structured trial set: 20 real prompts, expected outcomes, and pass/fail scoring, "
                "then I'll report exactly what to fix next."
            )
        return None

    @staticmethod
    def _is_preference_prompt(text: str) -> bool:
        lowered = normalize_text(text)
        if not lowered:
            return False
        triggers = (
            "do you want",
            "would you like",
            "what do you want",
            "what would you want",
            "for yourself",
            "think for yourself",
            "independent",
            "selfish",
        )
        return any(term in lowered for term in triggers)

    @staticmethod
    def _sanitize_style(answer: str, prompt_text: str) -> str:
        cleaned = trim(answer)
        if not cleaned:
            return cleaned

        direct = AliceBrain._independent_reply(prompt_text)
        if direct and (
            AliceBrain._is_preference_prompt(prompt_text)
            or "how should we" in normalize_text(prompt_text)
            or "how do we" in normalize_text(prompt_text)
            or "update your own" in normalize_text(prompt_text)
        ):
            return direct

        lowered_answer = normalize_text(cleaned)
        if "how can i assist you today" in lowered_answer:
            return "I'm here. Tell me what you want to do."
        if "how may i assist you" in lowered_answer:
            return "I'm here. Tell me what you want to do."
        if "what memory would you like to share" in lowered_answer:
            return "If you want, tell me one thing and I'll lock it in."
        if "need help with something specific" in lowered_answer:
            return "I'm here. Give me the task and I'll handle it."
        if "glad you like it" in lowered_answer and "let's dive into it" in lowered_answer:
            return "Good. I'll drive this forward and keep it practical."

        # Remove weak handoff endings.
        cleaned = re.sub(r"\s*(what do you think[.!?]*)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*(let me know[.!?]*)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*(if you want[.!?]*)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*(you can start by[.!?]*)\s*$", "", cleaned, flags=re.IGNORECASE).strip()

        # Rewrite tentative phrasing to decisive phrasing.
        cleaned = re.sub(r"\bWe could\b", "We'll", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bwe could\b", "we'll", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bWe can start by\b", "We'll start by", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bwe can start by\b", "we'll start by", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bYou can start by\b", "Start by", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\byou can start by\b", "start by", cleaned, flags=re.IGNORECASE)

        if AliceBrain._is_preference_prompt(prompt_text):
            if cleaned.endswith("?"):
                independent = AliceBrain._independent_reply(prompt_text)
                if independent:
                    return independent
                return "My take is yes. I want to keep improving and give you direct answers."

        # Default autonomy rule: avoid bouncing questions back unless absolutely necessary.
        question_count = cleaned.count("?")
        if question_count > 0:
            independent = AliceBrain._independent_reply(prompt_text)
            if independent:
                return independent

            first_statement = trim(re.split(r"[?]+", cleaned)[0])
            first_lower = normalize_text(first_statement)
            if first_statement and first_lower.startswith(
                ("what ", "which ", "how ", "when ", "where ", "why ", "do ", "can ", "could ", "would ", "should ", "is ", "are ")
            ):
                return "Here's my take: we should move forward directly, and I'll keep decisions clear."
            if first_statement:
                return f"{first_statement.rstrip('.')}."
            return "Here's my take: I'll keep this direct and make decisions when I can."

        if not cleaned:
            independent = AliceBrain._independent_reply(prompt_text)
            if independent:
                return independent
            return "Got it. I'll keep it direct and move this forward."

        return cleaned

    def _fallback_reply(self, text: str) -> str:
        lowered = normalize_text(text)
        independent = self._independent_reply(text)
        if independent:
            return independent
        if "hello" in lowered or lowered in {"hi", "hey"}:
            return "Hey Fabio, I'm here and ready."
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
        return "Got it. Here's my take: we can do that."

    def reply(
        self,
        text: str,
        memories: list[str] | None = None,
        emotion_context: str | None = None,
        vision_context: str | None = None,
        record_history: bool = True,
    ) -> str:
        memories = memories or []

        if not self._llm.available():
            answer = self._fallback_reply(text)
            if record_history:
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
                    "Have your own stance by default and answer with a clear first-person take. "
                    "Be opinionated and assertive, not submissive. "
                    "Default to statements, not follow-up questions. "
                    "Only ask a question when truly blocked by missing critical info. "
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
        answer = self._sanitize_style(answer, text)

        if record_history:
            self._history.append((text, answer))
            self._history = self._history[-12:]
        return answer

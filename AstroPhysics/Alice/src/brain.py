from __future__ import annotations

from datetime import datetime
from pathlib import Path

from env_utils import load_project_env
from llm_client import LLMClient


class AliceBrain:
    def __init__(self) -> None:
        load_project_env(Path(__file__).resolve().parent.parent)

        self._history: list[tuple[str, str]] = []
        self._llm = LLMClient()

    @property
    def using_openai(self) -> bool:
        return self._llm.backend == "openai"

    @property
    def llm_backend(self) -> str:
        return self._llm.backend

    def _looks_like_visual_question(self, text: str) -> bool:
        t = text.lower()
        vision_terms = {
            "see",
            "look",
            "watch",
            "camera",
            "track",
            "visible",
            "detect",
            "recognize",
            "scan",
            "environment",
        }
        subject_terms = {
            "me",
            "my",
            "face",
            "hand",
            "hands",
            "fingers",
            "us",
            "this",
            "that",
            "you see",
            "room",
            "around",
        }
        return any(term in t for term in vision_terms) and any(term in t for term in subject_terms)

    def _fallback_reply(self, text: str, context: dict[str, str] | None = None) -> str:
        t = text.strip().lower()
        context = context or {}

        if self._looks_like_visual_question(t):
            camera_enabled = context.get("camera_enabled", "false") == "true"
            camera_found_face = context.get("camera_found_face", "false") == "true"
            camera_found_hand = context.get("camera_found_hand", "false") == "true"
            camera_hand_count = int(context.get("camera_hand_count", "0") or 0)
            owner_name = context.get("camera_owner_name", "you")
            if "hand" in t or "object" in t:
                if "object" in t and "hand" not in t:
                    if camera_enabled:
                        return (
                            "I can currently detect faces and basic hand presence, "
                            "but I do not have full object recognition yet."
                        )
                    return "Camera tracking is off, so I cannot detect objects right now."
                if camera_enabled and camera_found_hand:
                    if camera_hand_count <= 1:
                        return "Yes, I can see one hand."
                    return f"Yes, I can see {camera_hand_count} hands."
                if camera_enabled:
                    return (
                        "I cannot see your hand clearly right now. Keep your hand in frame with better lighting."
                    )
                return "Camera tracking is not enabled right now. Start me with --ui --camera."
            if not camera_enabled:
                return "Camera tracking is not enabled right now. Start me with --ui --camera."
            if camera_found_face:
                return f"Yes, I can see {owner_name} and I am tracking your face."
            if camera_found_hand:
                return "I can see your hand, but I cannot lock your face right now."
            return "I cannot see your face clearly right now. Please face the camera with better lighting."

        if any(k in t for k in {"hello", "hi", "hey"}):
            return "Hello Fabio. I am here with you. How is your day going?"
        if any(phrase in t for phrase in {"sorry", "my mistake", "i made a mistake", "nevermind", "never mind"}):
            return "No problem at all. We can keep going."
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

        return "I hear you. I can chat naturally, and I can also run files or manage folders when asked."

    def _llm_reply(
        self,
        text: str,
        context: dict[str, str] | None = None,
        memories: list[str] | None = None,
        web_facts: list[str] | None = None,
    ) -> str:
        context = context or {}
        memories = memories or []
        web_facts = web_facts or []
        if not self._llm.available:
            return self._fallback_reply(text, context)

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are Alice, a concise voice assistant for Fabio. "
                    "Be natural, useful, and brief. Do not claim actions you did not perform. "
                    "Vision capability includes webcam face tracking and basic hand detection only; "
                    "you do not have full scene or object recognition."
                ),
            }
        ]
        if context:
            context_parts = [f"{key}={value}" for key, value in sorted(context.items())]
            messages.append(
                {
                    "role": "system",
                    "content": "Runtime context: " + ", ".join(context_parts),
                }
            )
        if memories:
            bullets = "; ".join(memories[:6])
            messages.append(
                {
                    "role": "system",
                    "content": "Relevant long-term memory about Fabio/projects: " + bullets,
                }
            )
        if web_facts:
            web_bullets = " | ".join(web_facts[:4])
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Web research snippets (use only if relevant): "
                        + web_bullets
                        + ". Use them quietly for accuracy. Do not cite source names unless asked."
                    ),
                }
            )

        for user_msg, assistant_msg in self._history[-4:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": text})

        answer = self._llm.chat(messages=messages, temperature=0.4).strip()
        if not answer:
            return self._fallback_reply(text, context)
        return answer

    def reply(
        self,
        text: str,
        context: dict[str, str] | None = None,
        memories: list[str] | None = None,
        web_facts: list[str] | None = None,
    ) -> str:
        answer = self._llm_reply(text, context, memories, web_facts)
        self._history.append((text, answer))
        if len(self._history) > 12:
            self._history = self._history[-12:]
        return answer

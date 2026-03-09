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
        vision_terms = {"see", "look", "watch", "camera", "track", "visible"}
        subject_terms = {"me", "my", "face", "hand", "us", "this", "that", "you see"}
        return any(term in t for term in vision_terms) and any(term in t for term in subject_terms)

    def _fallback_reply(self, text: str, context: dict[str, str] | None = None) -> str:
        t = text.strip().lower()
        context = context or {}

        if self._looks_like_visual_question(t):
            camera_enabled = context.get("camera_enabled", "false") == "true"
            camera_found_face = context.get("camera_found_face", "false") == "true"
            owner_name = context.get("camera_owner_name", "you")
            if "hand" in t or "object" in t:
                if camera_enabled:
                    return (
                        "I can currently track faces, but I do not have hand or object detection yet."
                    )
                return (
                    "Not yet. Camera tracking is off, and currently I only support face tracking."
                )
            if not camera_enabled:
                return "Camera tracking is not enabled right now. Start me with --ui --camera."
            if camera_found_face:
                return f"Yes, I can see {owner_name} and I am tracking your face."
            return "I cannot see your face clearly right now. Please face the camera with better lighting."

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
                    "Vision capability is limited to face tracking from a webcam; "
                    "you do not have full scene understanding or hand/object detection."
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
                        + ". If you use them, mention the source briefly."
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

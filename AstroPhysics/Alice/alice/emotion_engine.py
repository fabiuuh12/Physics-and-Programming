from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

from .string_utils import normalize_text


@dataclass(frozen=True)
class EmotionProfile:
    name: str
    valence: float
    arousal: float


@dataclass(frozen=True)
class EmotionState:
    name: str
    intensity: float
    valence: float
    arousal: float
    top_emotions: tuple[tuple[str, float], ...]

    def as_prompt(self) -> str:
        top = ", ".join([f"{name}:{score:.2f}" for name, score in self.top_emotions[:4]])
        return (
            f"Current emotion={self.name} intensity={self.intensity:.2f} "
            f"valence={self.valence:.2f} arousal={self.arousal:.2f} top=[{top}]"
        )


_EMOTION_PROFILES: dict[str, EmotionProfile] = {
    "neutral": EmotionProfile("neutral", 0.0, 0.0),
    "calm": EmotionProfile("calm", 0.45, -0.35),
    "content": EmotionProfile("content", 0.55, -0.2),
    "joy": EmotionProfile("joy", 0.9, 0.55),
    "excitement": EmotionProfile("excitement", 0.85, 0.95),
    "curiosity": EmotionProfile("curiosity", 0.4, 0.6),
    "focus": EmotionProfile("focus", 0.35, 0.25),
    "playfulness": EmotionProfile("playfulness", 0.75, 0.65),
    "affection": EmotionProfile("affection", 0.8, 0.4),
    "empathy": EmotionProfile("empathy", 0.5, 0.2),
    "gratitude": EmotionProfile("gratitude", 0.78, 0.3),
    "pride": EmotionProfile("pride", 0.72, 0.48),
    "relief": EmotionProfile("relief", 0.62, -0.05),
    "hope": EmotionProfile("hope", 0.58, 0.36),
    "anticipation": EmotionProfile("anticipation", 0.35, 0.52),
    "awe": EmotionProfile("awe", 0.7, 0.65),
    "surprise": EmotionProfile("surprise", 0.2, 0.86),
    "amusement": EmotionProfile("amusement", 0.74, 0.55),
    "admiration": EmotionProfile("admiration", 0.68, 0.32),
    "confidence": EmotionProfile("confidence", 0.63, 0.28),
    "concern": EmotionProfile("concern", -0.22, 0.5),
    "confusion": EmotionProfile("confusion", -0.25, 0.42),
    "uncertainty": EmotionProfile("uncertainty", -0.2, 0.28),
    "anxiety": EmotionProfile("anxiety", -0.62, 0.84),
    "fear": EmotionProfile("fear", -0.82, 0.95),
    "frustration": EmotionProfile("frustration", -0.56, 0.82),
    "anger": EmotionProfile("anger", -0.88, 0.92),
    "disappointment": EmotionProfile("disappointment", -0.6, -0.15),
    "sadness": EmotionProfile("sadness", -0.78, -0.42),
    "loneliness": EmotionProfile("loneliness", -0.7, -0.32),
    "embarrassment": EmotionProfile("embarrassment", -0.45, 0.5),
    "guilt": EmotionProfile("guilt", -0.7, 0.2),
    "boredom": EmotionProfile("boredom", -0.28, -0.62),
    "fatigue": EmotionProfile("fatigue", -0.35, -0.72),
    "overwhelm": EmotionProfile("overwhelm", -0.68, 0.9),
    "alertness": EmotionProfile("alertness", 0.05, 0.78),
    "determination": EmotionProfile("determination", 0.48, 0.56),
    "serenity": EmotionProfile("serenity", 0.62, -0.58),
    "nostalgia": EmotionProfile("nostalgia", 0.12, -0.22),
}


class EmotionEngine:
    def __init__(self) -> None:
        self._scores: dict[str, float] = {name: 0.0 for name in _EMOTION_PROFILES if name != "neutral"}
        self._last_tick = time.monotonic()
        self._face_visible = False
        self._last_face_seen_at = self._last_tick

    @staticmethod
    def all_emotions() -> list[str]:
        return sorted(_EMOTION_PROFILES.keys())

    def _boost(self, emotion: str, amount: float) -> None:
        if emotion not in self._scores:
            return
        self._scores[emotion] = max(0.0, min(1.6, self._scores[emotion] + amount))

    def _decay(self, dt_seconds: float) -> None:
        if dt_seconds <= 0:
            return
        # Half-life of roughly 8 seconds.
        decay = math.exp(-dt_seconds / 11.55)
        for key in self._scores:
            self._scores[key] *= decay

    def tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        self._decay(dt)

        if not self._face_visible and (now - self._last_face_seen_at) > 20.0:
            self._boost("loneliness", 0.05)
            self._boost("concern", 0.02)

    def observe_user_text(self, text: str) -> None:
        cleaned = normalize_text(text)
        if not cleaned:
            return

        def any_term(terms: Iterable[str]) -> bool:
            return any(term in cleaned for term in terms)

        if any_term(("hello", "hi ", "hey ", "good morning", "good afternoon", "good evening")):
            self._boost("joy", 0.22)
            self._boost("affection", 0.14)
        if any_term(("thank you", "thanks", "appreciate")):
            self._boost("gratitude", 0.34)
            self._boost("content", 0.14)
        if any_term(("love", "you are awesome", "great job", "nice work")):
            self._boost("affection", 0.34)
            self._boost("pride", 0.15)
        if any_term(("sorry", "my bad", "my fault")):
            self._boost("empathy", 0.22)
            self._boost("relief", 0.1)
        if any_term(("wow", "amazing", "incredible", "no way")):
            self._boost("surprise", 0.32)
            self._boost("awe", 0.22)
            self._boost("excitement", 0.16)
        if any_term(("confused", "i do not understand", "don't understand", "what do you mean")):
            self._boost("confusion", 0.32)
            self._boost("concern", 0.18)
        if any_term(("angry", "mad", "annoyed", "hate this")):
            self._boost("anger", 0.28)
            self._boost("frustration", 0.28)
        if any_term(("sad", "upset", "depressed", "unhappy")):
            self._boost("sadness", 0.33)
            self._boost("empathy", 0.18)
        if any_term(("scared", "afraid", "worried", "anxious")):
            self._boost("anxiety", 0.3)
            self._boost("concern", 0.2)
        if any_term(("bored", "tired", "exhausted", "sleepy")):
            self._boost("fatigue", 0.24)
            self._boost("boredom", 0.28)
        if any_term(("stuck", "hard", "difficult", "problem", "issue")):
            self._boost("determination", 0.2)
            self._boost("concern", 0.16)
        if any_term(("let's do this", "we can do it", "go for it", "start now")):
            self._boost("confidence", 0.23)
            self._boost("determination", 0.26)
            self._boost("anticipation", 0.18)

    def observe_intent(self, action: str) -> None:
        if action in {"run_file", "list_files", "open_folder", "web_research"}:
            self._boost("focus", 0.22)
            self._boost("curiosity", 0.14)
        if action in {"chat", "remember_memory", "recall_memory"}:
            self._boost("curiosity", 0.1)
            self._boost("affection", 0.06)
        if action in {"web_search", "web_research"}:
            self._boost("anticipation", 0.14)
        if action in {"help"}:
            self._boost("confidence", 0.15)

    def observe_result(self, ok: bool) -> None:
        if ok:
            self._boost("relief", 0.18)
            self._boost("pride", 0.1)
            self._boost("confidence", 0.12)
            self._boost("frustration", -0.08)
        else:
            self._boost("frustration", 0.22)
            self._boost("concern", 0.15)
            self._boost("uncertainty", 0.12)

    def observe_environment(
        self,
        *,
        face_found: bool,
        face_count: int,
        scene_label: str,
        motion_level: str,
        light_level: str,
    ) -> None:
        self._face_visible = face_found
        if face_found:
            self._last_face_seen_at = time.monotonic()
            self._boost("affection", 0.07)
            self._boost("curiosity", 0.04)
            if face_count >= 2:
                self._boost("excitement", 0.06)
                self._boost("playfulness", 0.05)

        scene = scene_label.lower()
        if "office" in scene or "study" in scene or "workspace" in scene:
            self._boost("focus", 0.04)
            self._boost("determination", 0.03)
        elif "outdoor" in scene:
            self._boost("awe", 0.06)
            self._boost("serenity", 0.04)
        elif "living" in scene or "bedroom" in scene:
            self._boost("calm", 0.04)
            self._boost("content", 0.03)
        elif "kitchen" in scene:
            self._boost("curiosity", 0.05)

        if motion_level == "high":
            self._boost("alertness", 0.12)
            self._boost("anticipation", 0.08)
        elif motion_level == "medium":
            self._boost("alertness", 0.06)

        if light_level == "dim":
            self._boost("fatigue", 0.05)
            self._boost("concern", 0.04)
        elif light_level == "bright":
            self._boost("excitement", 0.03)
            self._boost("joy", 0.03)

    def current(self) -> EmotionState:
        ranked = sorted(self._scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ranked[0] if ranked else ("neutral", 0.0)

        if best_score < 0.16:
            return EmotionState(
                name="neutral",
                intensity=0.12,
                valence=0.0,
                arousal=0.0,
                top_emotions=(("neutral", 1.0),),
            )

        intensity = max(0.0, min(1.0, (best_score - 0.14) / 1.2))
        top = tuple((name, min(1.0, score)) for name, score in ranked[:6] if score > 0.07)

        weighted_total = 0.0
        valence_sum = 0.0
        arousal_sum = 0.0
        for name, score in top:
            profile = _EMOTION_PROFILES[name]
            weight = score
            weighted_total += weight
            valence_sum += profile.valence * weight
            arousal_sum += profile.arousal * weight
        neutral_weight = max(0.0, 0.8 - weighted_total)
        weighted_total += neutral_weight

        valence = 0.0 if weighted_total <= 0 else valence_sum / weighted_total
        arousal = 0.0 if weighted_total <= 0 else arousal_sum / weighted_total

        return EmotionState(
            name=best_name,
            intensity=intensity,
            valence=max(-1.0, min(1.0, valence)),
            arousal=max(-1.0, min(1.0, arousal)),
            top_emotions=top if top else ((best_name, min(1.0, best_score)),),
        )

from __future__ import annotations


class Speaker:
    def __init__(self, enable_tts: bool = True) -> None:
        self._engine = None
        if not enable_tts:
            return

        try:
            import pyttsx3
        except ImportError:
            return

        self._engine = pyttsx3.init()

    @property
    def tts_enabled(self) -> bool:
        return self._engine is not None

    def say(self, text: str) -> None:
        print(f"Alice> {text}")
        if self._engine is not None:
            self._engine.say(text)
            self._engine.runAndWait()

from __future__ import annotations

from dataclasses import dataclass


class ListenerError(RuntimeError):
    pass


class BaseListener:
    def listen(self, prompt: str | None = None) -> str | None:
        raise NotImplementedError


class TextListener(BaseListener):
    def listen(self, prompt: str | None = None) -> str | None:
        try:
            return input(prompt or "You> ").strip()
        except EOFError:
            return "quit"
        except KeyboardInterrupt:
            return "quit"


@dataclass(frozen=True)
class VoiceSettings:
    timeout: int = 6
    phrase_time_limit: int = 8
    calibrate_seconds: float = 0.3


class VoiceListener(BaseListener):
    def __init__(self, settings: VoiceSettings | None = None) -> None:
        self.settings = settings or VoiceSettings()

        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise ListenerError(
                "Voice mode requires 'SpeechRecognition'. Install dependencies first."
            ) from exc

        self.sr = sr
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen(self, prompt: str | None = None) -> str | None:
        if prompt:
            print(prompt, end="", flush=True)

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=self.settings.calibrate_seconds
                )
                audio = self.recognizer.listen(
                    source,
                    timeout=self.settings.timeout,
                    phrase_time_limit=self.settings.phrase_time_limit,
                )
            return self.recognizer.recognize_google(audio).strip()
        except self.sr.WaitTimeoutError:
            return None
        except self.sr.UnknownValueError:
            return None
        except self.sr.RequestError as exc:
            raise ListenerError(
                "Speech service is unavailable right now. Check your internet connection."
            ) from exc

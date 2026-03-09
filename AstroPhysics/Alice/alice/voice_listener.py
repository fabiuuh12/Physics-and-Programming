from __future__ import annotations

import contextlib
import io
import os
import time
from typing import Callable, Optional

try:
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        import speech_recognition as sr
except Exception:  # pragma: no cover - optional dependency
    sr = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None


class VoiceListener:
    def __init__(self) -> None:
        self._available = False
        self._backend = "none"
        self._error = "Voice mode is unavailable on this platform."
        self._recognizer = None
        self._microphone = None
        self._sample_rate = 16000

        if sr is None:
            self._error = (
                "SpeechRecognition is not installed. "
                "Install with: python3 -m pip install SpeechRecognition pyaudio"
            )
            return

        self._recognizer = sr.Recognizer()

        has_pyaudio = False
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            try:
                import pyaudio  # type: ignore  # pragma: no cover

                has_pyaudio = pyaudio is not None
            except Exception:
                has_pyaudio = False

        if has_pyaudio:
            try:
                self._microphone = sr.Microphone()
                self._available = True
                self._backend = "speech_recognition"
                self._error = ""
                return
            except Exception:
                self._microphone = None

        if sd is None or np is None:
            self._error = "Microphone unavailable: PyAudio is not working and sounddevice is not installed."
            self._available = False
            return

        try:
            devices = sd.query_devices()
            has_input = any((device.get("max_input_channels", 0) or 0) > 0 for device in devices)
        except Exception as exc:
            self._error = f"Microphone unavailable: {exc}"
            self._available = False
            return

        if not has_input:
            self._error = "Microphone unavailable: no input audio device found."
            self._available = False
            return

        try:
            default_sr = int(sd.query_devices(None, "input")["default_samplerate"])
            if default_sr > 0:
                self._sample_rate = min(48000, max(8000, default_sr))
        except Exception:
            self._sample_rate = 16000

        self._available = True
        self._backend = "speech_recognition+sounddevice"
        self._error = ""

    def available(self) -> bool:
        return self._available

    def backend_name(self) -> str:
        return self._backend

    def last_error(self) -> str:
        return self._error

    def listen(
        self,
        timeout_seconds: float = 6.0,
        phrase_time_limit_seconds: float = 8.0,
        tick: Callable[[], None] | None = None,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> Optional[str]:
        if not self._available or self._recognizer is None or sr is None:
            return None

        audio = None
        if self._microphone is not None:
            try:
                with self._microphone as source:
                    if tick is not None:
                        tick()
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.35)
                    audio = self._recognizer.listen(
                        source,
                        timeout=max(0.5, timeout_seconds),
                        phrase_time_limit=max(0.8, phrase_time_limit_seconds),
                    )
            except sr.WaitTimeoutError:
                self._error = "No speech detected."
                return None
            except Exception as exc:
                self._error = f"Audio capture failed: {exc}"
                return None
        else:
            if sd is None or np is None:
                self._error = "Audio capture failed: no backend available."
                return None

            record_seconds = max(1.0, min(max(phrase_time_limit_seconds, 1.0), max(timeout_seconds + 1.0, 2.0)))
            frames = int(self._sample_rate * record_seconds)
            try:
                recorded = sd.rec(frames, samplerate=self._sample_rate, channels=1, dtype="int16")
                steps = max(1, int(record_seconds / 0.05))
                for _ in range(steps):
                    if tick is not None:
                        tick()
                    time.sleep(0.05)
                sd.wait()
                audio = sr.AudioData(recorded.tobytes(), self._sample_rate, 2)
            except Exception as exc:
                self._error = f"Audio capture failed: {exc}"
                return None

        engine = os.getenv("ALICE_STT_ENGINE", "google").strip().lower()
        try:
            if engine == "whisper" and hasattr(self._recognizer, "recognize_whisper"):
                text = self._recognizer.recognize_whisper(audio)
            else:
                text = self._recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            self._error = "Could not understand speech."
            return None
        except Exception as exc:
            self._error = f"Speech recognition failed: {exc}"
            return None

        text = text.strip()
        if not text:
            self._error = "Empty speech result."
            return None

        if on_partial_text is not None:
            on_partial_text(text)
        self._error = ""
        return text

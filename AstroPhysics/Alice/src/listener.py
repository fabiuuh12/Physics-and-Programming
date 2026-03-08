from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


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
            raise
        except KeyboardInterrupt:
            raise


@dataclass(frozen=True)
class VoiceSettings:
    timeout: int = 6
    phrase_time_limit: int = 8
    calibrate_seconds: float = 0.3


class VoiceListener(BaseListener):
    def __init__(self, settings: VoiceSettings | None = None) -> None:
        self.settings = settings or VoiceSettings()
        self._stt_backend = "none"
        self._whisper_model = None

        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise ListenerError(
                "Voice mode requires 'SpeechRecognition'. Install dependencies first."
            ) from exc

        self.sr = sr
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        requested_backend = os.getenv("ALICE_STT_BACKEND", "auto").strip().lower()
        if requested_backend not in {"auto", "faster_whisper", "google"}:
            requested_backend = "auto"

        if requested_backend in {"auto", "faster_whisper"}:
            if self._setup_faster_whisper():
                self._stt_backend = "faster_whisper"
            elif requested_backend == "faster_whisper":
                raise ListenerError(
                    "ALICE_STT_BACKEND=faster_whisper is set but faster-whisper is unavailable."
                )

        if self._stt_backend == "none":
            self._stt_backend = "google"

    @property
    def backend_name(self) -> str:
        return self._stt_backend

    def _setup_faster_whisper(self) -> bool:
        try:
            from faster_whisper import WhisperModel
        except Exception:
            return False

        model_name = os.getenv("ALICE_WHISPER_MODEL", "small")
        device = os.getenv("ALICE_WHISPER_DEVICE", "auto")
        compute_type = os.getenv("ALICE_WHISPER_COMPUTE_TYPE", "int8")

        try:
            self._whisper_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception:
            self._whisper_model = None
            return False
        return True

    def _transcribe_with_faster_whisper(self, audio: "object") -> str | None:
        if self._whisper_model is None:
            return None

        fd, tmp_name = tempfile.mkstemp(prefix="alice_stt_", suffix=".wav")
        tmp_path = Path(tmp_name)
        os.close(fd)
        try:
            wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
            tmp_path.write_bytes(wav_bytes)
            segments, _ = self._whisper_model.transcribe(
                str(tmp_path),
                beam_size=1,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text or None
        except Exception:
            return None
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

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
            if self._stt_backend == "faster_whisper":
                local_text = self._transcribe_with_faster_whisper(audio)
                if local_text:
                    return local_text
            return self.recognizer.recognize_google(audio).strip()
        except self.sr.WaitTimeoutError:
            return None
        except self.sr.UnknownValueError:
            return None
        except self.sr.RequestError as exc:
            raise ListenerError(
                "Speech service is unavailable right now. Check your internet connection."
            ) from exc
        except KeyboardInterrupt:
            raise

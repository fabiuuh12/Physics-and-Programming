from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


class ListenerError(RuntimeError):
    pass


class BaseListener:
    def listen(
        self,
        prompt: str | None = None,
        *,
        timeout: float | None = None,
        phrase_time_limit: float | None = None,
        calibrate: bool | None = None,
    ) -> str | None:
        raise NotImplementedError


class TextListener(BaseListener):
    def listen(
        self,
        prompt: str | None = None,
        *,
        timeout: float | None = None,
        phrase_time_limit: float | None = None,
        calibrate: bool | None = None,
    ) -> str | None:
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
    calibrate_seconds: float = 0.15


class VoiceListener(BaseListener):
    def __init__(self, settings: VoiceSettings | None = None) -> None:
        self.settings = settings or VoiceSettings()
        self._stt_backend = "none"
        self._whisper_model = None
        self._last_calibration_at = 0.0
        calibration_interval = os.getenv("ALICE_STT_CALIBRATION_INTERVAL", "28").strip()
        try:
            self._calibration_interval_s = max(8.0, min(180.0, float(calibration_interval)))
        except ValueError:
            self._calibration_interval_s = 28.0
        energy_raw = os.getenv("ALICE_STT_ENERGY_BASE", "110").strip()
        try:
            self._energy_base = max(50.0, min(380.0, float(energy_raw)))
        except ValueError:
            self._energy_base = 110.0
        energy_min_raw = os.getenv("ALICE_STT_ENERGY_MIN", "68").strip()
        energy_max_raw = os.getenv("ALICE_STT_ENERGY_MAX", "235").strip()
        try:
            self._energy_min = max(40.0, min(280.0, float(energy_min_raw)))
        except ValueError:
            self._energy_min = 68.0
        try:
            self._energy_max = max(120.0, min(420.0, float(energy_max_raw)))
        except ValueError:
            self._energy_max = 235.0
        if self._energy_max < self._energy_min:
            self._energy_min, self._energy_max = self._energy_max, self._energy_min

        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise ListenerError(
                "Voice mode requires 'SpeechRecognition'. Install dependencies first."
            ) from exc

        self.sr = sr
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = self._energy_base
        self.recognizer.dynamic_energy_adjustment_damping = 0.10
        self.recognizer.dynamic_energy_ratio = 1.20
        self.recognizer.pause_threshold = 0.58
        self.recognizer.non_speaking_duration = 0.16
        self.recognizer.phrase_threshold = 0.18
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

    def listen(
        self,
        prompt: str | None = None,
        *,
        timeout: float | None = None,
        phrase_time_limit: float | None = None,
        calibrate: bool | None = None,
    ) -> str | None:
        if prompt:
            print(prompt, end="", flush=True)

        listen_timeout = float(timeout) if timeout is not None else float(self.settings.timeout)
        listen_phrase_limit = (
            float(phrase_time_limit)
            if phrase_time_limit is not None
            else float(self.settings.phrase_time_limit)
        )

        try:
            with self.microphone as source:
                now = time.monotonic()
                should_calibrate = now - self._last_calibration_at >= self._calibration_interval_s
                if calibrate is True:
                    should_calibrate = True
                elif calibrate is False:
                    should_calibrate = False
                if should_calibrate:
                    self.recognizer.adjust_for_ambient_noise(
                        source, duration=self.settings.calibrate_seconds
                    )
                    self._last_calibration_at = now
                else:
                    # Drift threshold down a bit over time so normal speaking volume is enough.
                    self.recognizer.energy_threshold *= 0.992
                self.recognizer.energy_threshold = max(
                    self._energy_min,
                    min(self._energy_max, self.recognizer.energy_threshold),
                )
                audio = self.recognizer.listen(
                    source,
                    timeout=listen_timeout,
                    phrase_time_limit=listen_phrase_limit,
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

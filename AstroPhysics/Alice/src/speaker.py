from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


class Speaker:
    def __init__(self, enable_tts: bool = True) -> None:
        self._engine = None
        self._openai_client = None
        self._backend = "none"
        self._fallback_backend = "none"
        self._voice = os.getenv("ALICE_VOICE")
        self._openai_tts_model = os.getenv("ALICE_OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self._openai_tts_voice = os.getenv("ALICE_OPENAI_TTS_VOICE", "sage")
        self._openai_tts_speed = self._read_openai_speed()
        self._openai_tts_style = os.getenv("ALICE_OPENAI_TTS_STYLE", "").strip()
        self._speak_lock = threading.Lock()
        if not enable_tts:
            return

        requested_backend = os.getenv("ALICE_TTS_BACKEND", "auto").strip().lower()
        if requested_backend not in {"auto", "say", "pyttsx3", "openai"}:
            requested_backend = "auto"

        if requested_backend == "openai":
            if self._setup_openai_backend():
                self._backend = "openai"
                # Keep a local fallback to avoid silence on network/API failures.
                if sys.platform == "darwin" and self._command_exists("say"):
                    self._fallback_backend = "say"
                return
            requested_backend = "auto"

        # macOS native backend is the most reliable choice for audible output.
        if requested_backend in {"auto", "say"} and sys.platform == "darwin":
            if self._command_exists("say"):
                self._backend = "say"
                return
            if requested_backend == "say":
                return

        if requested_backend in {"auto", "pyttsx3"}:
            try:
                import pyttsx3
            except ImportError:
                return

            self._engine = pyttsx3.init()
            self._backend = "pyttsx3"

    def _read_openai_speed(self) -> float:
        raw = os.getenv("ALICE_OPENAI_TTS_SPEED", "1.0").strip()
        try:
            speed = float(raw)
        except ValueError:
            return 1.0
        if speed < 0.25:
            return 0.25
        if speed > 4.0:
            return 4.0
        return speed

    def _command_exists(self, command: str) -> bool:
        return subprocess.run(
            ["which", command],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0

    def _setup_openai_backend(self) -> bool:
        if not os.getenv("OPENAI_API_KEY"):
            return False
        if not self._command_exists("afplay"):
            return False

        try:
            from openai import OpenAI
        except ImportError:
            return False

        try:
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            self._openai_client = None
            return False
        return True

    def _chunk_text(self, text: str, max_chunk_len: int = 220) -> list[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        if len(cleaned) <= max_chunk_len:
            return [cleaned]

        pieces = re.split(r"(?<=[.!?])\s+", cleaned)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            candidate = piece if not current else f"{current} {piece}"
            if len(candidate) <= max_chunk_len:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(piece) <= max_chunk_len:
                current = piece
            else:
                for i in range(0, len(piece), max_chunk_len):
                    chunks.append(piece[i : i + max_chunk_len])
                current = ""
        if current:
            chunks.append(current)
        return chunks

    def _speak_with_say(self, text: str) -> None:
        command = ["say"]
        if self._voice:
            command.extend(["-v", self._voice])
        command.append(text)
        subprocess.run(command, check=False)

    def _speak_with_pyttsx3(self, text: str) -> None:
        if self._engine is None:
            return
        self._engine.say(text)
        self._engine.runAndWait()

    def _speak_with_openai(self, text: str) -> bool:
        if self._openai_client is None:
            return False

        instructions = self._openai_tts_style or None
        fd, temp_name = tempfile.mkstemp(prefix="alice_tts_", suffix=".mp3")
        temp_file_path = Path(temp_name)
        os.close(fd)

        try:
            response = self._openai_client.audio.speech.create(
                model=self._openai_tts_model,
                voice=self._openai_tts_voice,
                input=text,
                response_format="mp3",
                speed=self._openai_tts_speed,
                instructions=instructions,
            )
            response.stream_to_file(str(temp_file_path))
            subprocess.run(["afplay", str(temp_file_path)], check=False)
            return True
        except Exception:
            return False
        finally:
            try:
                temp_file_path.unlink(missing_ok=True)
            except OSError:
                pass

    @property
    def backend_name(self) -> str:
        if self._backend == "openai":
            return f"openai ({self._openai_tts_voice})"
        if self._backend == "say" and self._voice:
            return f"say ({self._voice})"
        return self._backend

    @property
    def tts_enabled(self) -> bool:
        return self._backend != "none"

    def say(self, text: str) -> None:
        print(f"Alice> {text}")
        chunks = self._chunk_text(text)
        if not chunks:
            return

        with self._speak_lock:
            for chunk in chunks:
                if self._backend == "openai":
                    if self._speak_with_openai(chunk):
                        continue
                    if self._fallback_backend == "say":
                        self._speak_with_say(chunk)
                        continue
                    if self._engine is not None:
                        self._speak_with_pyttsx3(chunk)
                        continue
                if self._backend == "say":
                    self._speak_with_say(chunk)
                    continue
                if self._backend == "pyttsx3":
                    self._speak_with_pyttsx3(chunk)

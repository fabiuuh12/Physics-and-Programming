from __future__ import annotations

from typing import Callable, Optional


class VoiceListener:
    def __init__(self) -> None:
        self._available = False
        self._backend = "none"
        self._error = "Voice mode is unavailable on this platform."

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
        del timeout_seconds, phrase_time_limit_seconds, tick, on_partial_text
        return None

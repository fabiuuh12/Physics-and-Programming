from __future__ import annotations


class AliceUI:
    def __init__(self) -> None:
        self._running = False

    def start(self) -> bool:
        self._running = False
        return False

    def pump(self) -> None:
        return None

    def stop(self) -> None:
        self._running = False

    def running(self) -> bool:
        return self._running

    def set_state(self, _state: str) -> None:
        return None

    def set_status(self, _status: str) -> None:
        return None

    def add_message(self, _speaker: str, _text: str) -> None:
        return None

    def set_face_target(self, _x: float, _y: float, _found: bool, _face_count: int = 0) -> None:
        return None

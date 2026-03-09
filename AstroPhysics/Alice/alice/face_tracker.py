from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaceObservation:
    found: bool = False
    x: float = 0.0
    y: float = 0.0
    face_count: int = 0


class FaceTracker:
    def __init__(self) -> None:
        self._running = False
        self._error = "Face tracking unavailable on this platform."
        self._observation = FaceObservation()

    def start(self, camera_index: int = 0) -> bool:
        del camera_index
        self._running = False
        return False

    def stop(self) -> None:
        self._running = False

    def running(self) -> bool:
        return self._running

    def last_error(self) -> str:
        return self._error

    def latest(self) -> FaceObservation:
        return self._observation

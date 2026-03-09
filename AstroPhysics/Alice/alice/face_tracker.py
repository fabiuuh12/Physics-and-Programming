from __future__ import annotations

from dataclasses import dataclass
import threading
import time

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dep
    cv2 = None


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
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._cascade = None

    def start(self, camera_index: int = 0) -> bool:
        if self._running:
            return True

        if cv2 is None:
            self._error = "OpenCV is not installed. Install with: python3 -m pip install opencv-python"
            self._running = False
            return False

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            self._error = "Failed to load Haar face detector."
            self._running = False
            return False

        avfoundation = getattr(cv2, "CAP_AVFOUNDATION", None)
        if avfoundation is not None:
            cap = cv2.VideoCapture(camera_index, avfoundation)
        else:
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            self._error = (
                "Could not open camera. Check System Settings > Privacy & Security > Camera for Terminal/iTerm."
            )
            self._running = False
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._capture = cap
        self._cascade = cascade
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="alice-face-tracker", daemon=True)
        self._thread.start()
        self._running = True
        self._error = ""
        return True

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        self._running = False

    def running(self) -> bool:
        return self._running

    def last_error(self) -> str:
        return self._error

    def latest(self) -> FaceObservation:
        with self._lock:
            return FaceObservation(
                found=self._observation.found,
                x=self._observation.x,
                y=self._observation.y,
                face_count=self._observation.face_count,
            )

    def _update_observation(self, found: bool, x: float, y: float, face_count: int) -> None:
        with self._lock:
            self._observation = FaceObservation(
                found=found,
                x=max(-1.0, min(1.0, x)),
                y=max(-1.0, min(1.0, y)),
                face_count=face_count,
            )

    def _loop(self) -> None:
        if cv2 is None or self._capture is None or self._cascade is None:
            return

        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(70, 70),
            )

            height, width = gray.shape[:2]
            if len(faces) == 0:
                self._update_observation(False, 0.0, 0.0, 0)
                continue

            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)

            # Normalized camera-space coordinates in [-1, 1], with positive y = up.
            nx = ((cx / max(1.0, width)) - 0.5) * 2.0
            ny = -(((cy / max(1.0, height)) - 0.5) * 2.0)
            self._update_observation(True, nx, ny, len(faces))

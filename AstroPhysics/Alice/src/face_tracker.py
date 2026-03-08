from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class FaceObservation:
    found: bool
    x: float = 0.0
    y: float = 0.0
    face_count: int = 0
    owner_locked: bool = False
    owner_name: str | None = None


class FaceTracker:
    def __init__(self, *, camera_index: int = 0, owner_name: str = "You") -> None:
        self.camera_index = camera_index
        self.owner_name = owner_name
        self.last_error: str | None = None
        self._cv2 = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest = FaceObservation(found=False)

        self._owner_signature: tuple[float, float, float, float] | None = None
        self._lost_frames = 0

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        try:
            import cv2
        except Exception as exc:
            self.last_error = f"OpenCV unavailable: {exc}"
            return False

        self._cv2 = cv2
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_latest(self) -> FaceObservation:
        with self._lock:
            return self._latest

    def _set_latest(self, value: FaceObservation) -> None:
        with self._lock:
            self._latest = value

    def _pick_face(
        self,
        faces: list[tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int] | None:
        if not faces:
            return None

        if self._owner_signature is None:
            return max(faces, key=lambda f: f[2] * f[3])

        ox, oy, ow, oh = self._owner_signature
        best_face = None
        best_score = float("inf")
        for x, y, w, h in faces:
            cx = x + w * 0.5
            cy = y + h * 0.5
            center_dist = math.hypot(cx - ox, cy - oy) / max(frame_w, frame_h)
            area_ratio = (w * h) / max(1.0, ow * oh)
            size_cost = abs(math.log(max(area_ratio, 1e-4)))
            score = center_dist * 2.0 + size_cost * 0.6
            if score < best_score:
                best_score = score
                best_face = (x, y, w, h)
        return best_face

    def _run(self) -> None:
        assert self._cv2 is not None
        cv2 = self._cv2

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.last_error = f"Cannot open camera index {self.camera_index}"
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            self.last_error = "Could not load Haar face cascade."
            cap.release()
            self._running = False
            return

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                frame_h, frame_w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                raw_faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(70, 70),
                )
                faces = [(int(x), int(y), int(w), int(h)) for x, y, w, h in raw_faces]
                selected = self._pick_face(faces, frame_w, frame_h)

                if selected is None:
                    self._lost_frames += 1
                    if self._lost_frames > 80:
                        self._owner_signature = None
                    self._set_latest(
                        FaceObservation(
                            found=False,
                            face_count=len(faces),
                            owner_locked=self._owner_signature is not None,
                            owner_name=self.owner_name if self._owner_signature else None,
                        )
                    )
                    time.sleep(0.02)
                    continue

                self._lost_frames = 0
                x, y, w, h = selected
                cx = x + w * 0.5
                cy = y + h * 0.5

                if self._owner_signature is None:
                    self._owner_signature = (cx, cy, float(w), float(h))
                else:
                    ox, oy, ow, oh = self._owner_signature
                    alpha = 0.85
                    self._owner_signature = (
                        ox * alpha + cx * (1.0 - alpha),
                        oy * alpha + cy * (1.0 - alpha),
                        ow * alpha + float(w) * (1.0 - alpha),
                        oh * alpha + float(h) * (1.0 - alpha),
                    )

                x_norm = ((cx / max(1, frame_w)) - 0.5) * 2.0
                y_norm = ((cy / max(1, frame_h)) - 0.5) * 2.0
                x_norm = max(-1.0, min(1.0, x_norm))
                y_norm = max(-1.0, min(1.0, y_norm))

                self._set_latest(
                    FaceObservation(
                        found=True,
                        x=x_norm,
                        y=y_norm,
                        face_count=len(faces),
                        owner_locked=True,
                        owner_name=self.owner_name,
                    )
                )
                time.sleep(0.015)
        finally:
            cap.release()
            self._running = False

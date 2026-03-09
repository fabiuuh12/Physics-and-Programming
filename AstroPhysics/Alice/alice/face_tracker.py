from __future__ import annotations

from dataclasses import dataclass, field
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
    people_count: int = 0
    scene_label: str = "unknown"
    scene_confidence: float = 0.0
    light_level: str = "unknown"
    motion_level: str = "low"
    dominant_color: str = "unknown"
    objects: tuple[str, ...] = field(default_factory=tuple)
    summary: str = "No visual observation yet."
    timestamp: float = 0.0


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
        self._hog = None
        self._last_people_count = 0
        self._prev_gray = None
        self._frame_index = 0

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
        self._prev_gray = None
        self._frame_index = 0
        self._last_people_count = 0
        self._hog = None

        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog
        except Exception:
            self._hog = None

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
                people_count=self._observation.people_count,
                scene_label=self._observation.scene_label,
                scene_confidence=self._observation.scene_confidence,
                light_level=self._observation.light_level,
                motion_level=self._observation.motion_level,
                dominant_color=self._observation.dominant_color,
                objects=tuple(self._observation.objects),
                summary=self._observation.summary,
                timestamp=self._observation.timestamp,
            )

    @staticmethod
    def _safe_ratio(mask) -> float:
        try:
            return float(mask.mean()) / 255.0
        except Exception:
            return 0.0

    def _people_count(self, frame) -> int:
        if self._hog is None:
            return max(0, self._last_people_count)

        # Running this every frame is expensive; sample every 8th frame.
        if self._frame_index % 8 != 0:
            return max(0, self._last_people_count)

        try:
            rects, _weights = self._hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            self._last_people_count = len(rects)
        except Exception:
            pass
        return max(0, self._last_people_count)

    def _scene_from_features(
        self,
        *,
        brightness: float,
        edge_density: float,
        green_ratio: float,
        warm_ratio: float,
        screen_score: int,
        people_count: int,
    ) -> tuple[str, float]:
        if green_ratio > 0.24 and brightness > 85:
            conf = min(0.95, 0.58 + green_ratio * 0.95 + (brightness / 255.0) * 0.14)
            return "outdoor view", conf
        if screen_score >= 1 and edge_density > 0.055:
            conf = min(0.96, 0.54 + min(0.3, edge_density * 1.6) + min(0.24, screen_score * 0.16))
            return "office workspace", conf
        if warm_ratio > 0.24 and 0.045 <= edge_density <= 0.16:
            conf = min(0.92, 0.44 + warm_ratio * 0.9 + edge_density * 0.4)
            return "kitchen or dining area", conf
        if edge_density < 0.05 and brightness < 145:
            conf = min(0.9, 0.43 + (0.07 - edge_density) * 2.6)
            return "bedroom or quiet room", conf
        if people_count >= 2 and edge_density > 0.09:
            conf = min(0.93, 0.46 + edge_density * 0.9 + min(0.25, people_count * 0.08))
            return "social indoor area", conf
        return "general indoor room", 0.36 + min(0.24, edge_density * 0.9)

    def _analyze_scene(self, frame, gray, face_count: int, motion: float) -> dict:
        if cv2 is None:
            return {
                "people_count": face_count,
                "scene_label": "unknown",
                "scene_confidence": 0.0,
                "light_level": "unknown",
                "motion_level": "low",
                "dominant_color": "unknown",
                "objects": tuple(),
                "summary": "Vision unavailable.",
            }

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = float(hsv[:, :, 2].mean())
        saturation = float(hsv[:, :, 1].mean())

        edges = cv2.Canny(gray, 70, 160)
        edge_density = float((edges > 0).sum()) / float(max(1, gray.size))

        green_mask = cv2.inRange(hsv, (35, 42, 40), (90, 255, 255))
        blue_mask = cv2.inRange(hsv, (90, 35, 40), (140, 255, 255))
        warm_mask_1 = cv2.inRange(hsv, (0, 45, 45), (22, 255, 255))
        warm_mask_2 = cv2.inRange(hsv, (155, 45, 45), (180, 255, 255))
        warm_mask = cv2.bitwise_or(warm_mask_1, warm_mask_2)

        green_ratio = self._safe_ratio(green_mask)
        blue_ratio = self._safe_ratio(blue_mask)
        warm_ratio = self._safe_ratio(warm_mask)

        bright = cv2.inRange(gray, 175, 255)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        screen_score = 0
        frame_area = float(frame.shape[0] * frame.shape[1])
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame_area * 0.02:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / float(max(1, h))
            extent = area / float(max(1, w * h))
            if 1.05 <= aspect <= 2.6 and extent > 0.45:
                screen_score += 1

        people_count = max(face_count, self._people_count(frame))
        scene_label, scene_confidence = self._scene_from_features(
            brightness=brightness,
            edge_density=edge_density,
            green_ratio=green_ratio,
            warm_ratio=warm_ratio,
            screen_score=screen_score,
            people_count=people_count,
        )

        if brightness < 70:
            light_level = "dim"
        elif brightness > 165:
            light_level = "bright"
        else:
            light_level = "medium"

        if motion > 0.11:
            motion_level = "high"
        elif motion > 0.05:
            motion_level = "medium"
        else:
            motion_level = "low"

        dominant_color = "neutral"
        if green_ratio > max(blue_ratio, warm_ratio):
            dominant_color = "green"
        elif blue_ratio > max(green_ratio, warm_ratio):
            dominant_color = "blue"
        elif warm_ratio > max(green_ratio, blue_ratio):
            dominant_color = "warm"

        objects: list[str] = []
        if face_count > 0:
            objects.append("face")
        if people_count > 0:
            objects.append("person")
        if screen_score > 0:
            objects.append("monitor_or_tv")
        if green_ratio > 0.14:
            objects.append("plant_or_greenery")
        if edge_density > 0.13:
            objects.append("furniture")
        if warm_ratio > 0.28:
            objects.append("wood_or_warm_surfaces")
        if motion_level != "low":
            objects.append("movement")
        if light_level == "dim":
            objects.append("shadows")
        if saturation < 45:
            objects.append("neutral_tones")

        unique_objects = tuple(dict.fromkeys(objects))

        summary_bits = [
            f"{scene_label} ({scene_confidence:.2f} confidence)",
            f"{light_level} light",
            f"{motion_level} motion",
        ]
        if unique_objects:
            summary_bits.append("objects: " + ", ".join(unique_objects[:6]))
        summary = "; ".join(summary_bits)

        return {
            "people_count": people_count,
            "scene_label": scene_label,
            "scene_confidence": max(0.0, min(1.0, scene_confidence)),
            "light_level": light_level,
            "motion_level": motion_level,
            "dominant_color": dominant_color,
            "objects": unique_objects,
            "summary": summary,
        }

    def _update_observation(self, found: bool, x: float, y: float, face_count: int, scene: dict) -> None:
        with self._lock:
            self._observation = FaceObservation(
                found=found,
                x=max(-1.0, min(1.0, x)),
                y=max(-1.0, min(1.0, y)),
                face_count=face_count,
                people_count=int(scene.get("people_count", 0)),
                scene_label=str(scene.get("scene_label", "unknown")),
                scene_confidence=float(scene.get("scene_confidence", 0.0)),
                light_level=str(scene.get("light_level", "unknown")),
                motion_level=str(scene.get("motion_level", "low")),
                dominant_color=str(scene.get("dominant_color", "unknown")),
                objects=tuple(scene.get("objects", tuple())),
                summary=str(scene.get("summary", "")),
                timestamp=time.time(),
            )

    def _loop(self) -> None:
        if cv2 is None or self._capture is None or self._cascade is None:
            return

        while not self._stop_event.is_set():
            self._frame_index += 1

            ok, frame = self._capture.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            motion = 0.0
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                motion = float(diff.mean()) / 255.0
            self._prev_gray = gray

            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(70, 70),
            )

            scene = self._analyze_scene(frame, gray, len(faces), motion)

            height, width = gray.shape[:2]
            if len(faces) == 0:
                self._update_observation(False, 0.0, 0.0, 0, scene)
                continue

            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)

            # Normalized camera-space coordinates in [-1, 1], with positive y = up.
            nx = ((cx / max(1.0, width)) - 0.5) * 2.0
            ny = -(((cy / max(1.0, height)) - 0.5) * 2.0)
            self._update_observation(True, nx, ny, len(faces), scene)

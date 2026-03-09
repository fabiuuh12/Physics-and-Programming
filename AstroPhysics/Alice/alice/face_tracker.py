from __future__ import annotations

from dataclasses import dataclass, field
import math
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
    face_descriptions: tuple[str, ...] = field(default_factory=tuple)
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
        self._cascades: tuple[object, ...] = tuple()
        self._smile_cascade = None
        self._eye_cascades: tuple[object, ...] = tuple()
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

        face_cascades: list[object] = []
        cascade_candidates = (
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        )
        for filename in cascade_candidates:
            try:
                path = cv2.data.haarcascades + filename
                cascade = cv2.CascadeClassifier(path)
            except Exception:
                continue
            if not cascade.empty():
                face_cascades.append(cascade)

        if not face_cascades:
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
        self._cascades = tuple(face_cascades)
        self._smile_cascade = None
        self._eye_cascades = tuple()
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

        try:
            smile_cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
            smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
            if not smile_cascade.empty():
                self._smile_cascade = smile_cascade
        except Exception:
            self._smile_cascade = None

        eye_cascades: list[object] = []
        eye_candidates = (
            "haarcascade_eye.xml",
            "haarcascade_eye_tree_eyeglasses.xml",
        )
        for filename in eye_candidates:
            try:
                path = cv2.data.haarcascades + filename
                cascade = cv2.CascadeClassifier(path)
            except Exception:
                continue
            if not cascade.empty():
                eye_cascades.append(cascade)
        self._eye_cascades = tuple(eye_cascades)

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
        self._cascades = tuple()
        self._smile_cascade = None
        self._eye_cascades = tuple()
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
                face_descriptions=tuple(self._observation.face_descriptions),
                summary=self._observation.summary,
                timestamp=self._observation.timestamp,
            )

    @staticmethod
    def _safe_ratio(mask) -> float:
        try:
            return float(mask.mean()) / 255.0
        except Exception:
            return 0.0

    @staticmethod
    def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2 = ax1 + aw
        ay2 = ay1 + ah
        bx2 = bx1 + bw
        by2 = by1 + bh

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0.0:
            return 0.0
        a_area = float(max(1, aw * ah))
        b_area = float(max(1, bw * bh))
        union = a_area + b_area - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    @staticmethod
    def _center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        acx = ax + (aw / 2.0)
        acy = ay + (ah / 2.0)
        bcx = bx + (bw / 2.0)
        bcy = by + (bh / 2.0)
        return math.hypot(acx - bcx, acy - bcy)

    def _dedupe_faces(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []
        ordered = sorted(boxes, key=lambda item: item[2] * item[3], reverse=True)
        selected: list[tuple[int, int, int, int]] = []
        for candidate in ordered:
            keep = True
            for existing in selected:
                if self._box_iou(candidate, existing) >= 0.34:
                    keep = False
                    break
                dist = self._center_distance(candidate, existing)
                size = max(candidate[2], candidate[3], existing[2], existing[3])
                if dist < (size * 0.35):
                    keep = False
                    break
            if keep:
                selected.append(candidate)
            if len(selected) >= 8:
                break
        return selected

    def _detect_faces(self, gray) -> list[tuple[int, int, int, int]]:
        if not self._cascades:
            return []

        raw_boxes: list[tuple[int, int, int, int]] = []
        # First pass: stable/precise. Second pass: more permissive to reacquire faces.
        passes = (
            {"scaleFactor": 1.1, "minNeighbors": 6, "minSize": (64, 64)},
            {"scaleFactor": 1.06, "minNeighbors": 4, "minSize": (44, 44)},
        )
        for params in passes:
            for cascade in self._cascades:
                try:
                    found = cascade.detectMultiScale(gray, **params)
                except Exception:
                    continue
                for (x, y, w, h) in found:
                    box = (int(x), int(y), int(w), int(h))
                    if box[2] > 0 and box[3] > 0:
                        raw_boxes.append(box)
            deduped = self._dedupe_faces(raw_boxes)
            if deduped:
                return deduped
        return []

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

    @staticmethod
    def _horizontal_label(x_ratio: float) -> str:
        if x_ratio < 0.33:
            return "left side"
        if x_ratio > 0.67:
            return "right side"
        return "center"

    @staticmethod
    def _vertical_label(y_ratio: float) -> str:
        if y_ratio < 0.33:
            return "upper frame"
        if y_ratio > 0.67:
            return "lower frame"
        return "middle frame"

    @staticmethod
    def _distance_label(area_ratio: float) -> str:
        if area_ratio > 0.11:
            return "very close"
        if area_ratio > 0.06:
            return "close"
        if area_ratio > 0.03:
            return "mid distance"
        return "farther back"

    @staticmethod
    def _face_light_label(mean_gray: float) -> str:
        if mean_gray < 72:
            return "dim"
        if mean_gray > 152:
            return "bright"
        return "medium"

    @staticmethod
    def _skin_mask_ycrcb(ycrcb_img):
        if cv2 is None:
            return None
        return cv2.inRange(ycrcb_img, (0, 133, 77), (255, 173, 127))

    @staticmethod
    def _estimate_skin_tone(face_bgr) -> str:
        if cv2 is None or face_bgr is None or getattr(face_bgr, "size", 0) == 0:
            return "uncertain"

        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = FaceTracker._skin_mask_ycrcb(ycrcb)
        if skin_mask is None:
            return "uncertain"

        area = max(1, int(face_bgr.shape[0] * face_bgr.shape[1]))
        skin_count = int(cv2.countNonZero(skin_mask))
        if skin_count < max(50, int(area * 0.04)):
            return "uncertain"

        y_channel = ycrcb[:, :, 0]
        luma = float(cv2.mean(y_channel, mask=skin_mask)[0])
        if luma < 72:
            return "deep"
        if luma < 98:
            return "medium-deep"
        if luma < 126:
            return "medium"
        if luma < 156:
            return "light-medium"
        return "light"

    @staticmethod
    def _classify_hair_hsv(hue: float, sat: float, val: float) -> str:
        if val < 55:
            return "black"
        if sat < 30 and val > 165:
            return "gray or white"
        if (hue < 10 or hue > 165) and sat > 65:
            return "red or auburn"
        if 16 <= hue <= 35 and sat >= 60 and val >= 105:
            return "blonde"
        if 8 <= hue <= 28:
            return "brown"
        if sat < 55 and val < 140:
            return "dark brown"
        return "brown"

    @staticmethod
    def _estimate_hair_color(frame_bgr, x1: int, y1: int, x2: int, y2: int) -> str:
        if cv2 is None or frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return "uncertain"

        fh, fw = frame_bgr.shape[:2]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        pad_x = int(w * 0.14)
        hair_x1 = max(0, x1 - pad_x)
        hair_x2 = min(fw, x2 + pad_x)
        hair_y1 = max(0, y1 - int(h * 0.62))
        hair_y2 = min(fh, y1 + int(h * 0.18))
        if hair_x1 >= hair_x2 or hair_y1 >= hair_y2:
            return "uncertain"

        hair_roi = frame_bgr[hair_y1:hair_y2, hair_x1:hair_x2]
        if getattr(hair_roi, "size", 0) == 0:
            return "uncertain"

        hsv = cv2.cvtColor(hair_roi, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(hair_roi, cv2.COLOR_BGR2YCrCb)
        skin_mask = FaceTracker._skin_mask_ycrcb(ycrcb)

        # Prefer non-skin pixels to avoid forehead bias.
        use_mask = None
        if skin_mask is not None:
            non_skin_mask = cv2.bitwise_not(skin_mask)
            valid = int(cv2.countNonZero(non_skin_mask))
            area = max(1, int(hair_roi.shape[0] * hair_roi.shape[1]))
            if valid >= max(60, int(area * 0.07)):
                use_mask = non_skin_mask

        mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=use_mask)
        return FaceTracker._classify_hair_hsv(float(mean_h), float(mean_s), float(mean_v))

    @staticmethod
    def _classify_eye_hsv(hue: float, sat: float, val: float) -> str:
        if val < 45:
            return "very dark"
        if sat < 28:
            if val > 135:
                return "gray"
            return "dark brown"
        if 8 <= hue <= 30:
            return "brown or hazel"
        if 30 < hue <= 90:
            return "green or hazel"
        if 90 < hue <= 135:
            return "blue"
        return "brown"

    def _estimate_eye_color(self, face_bgr, face_gray) -> str:
        if (
            cv2 is None
            or face_bgr is None
            or face_gray is None
            or getattr(face_bgr, "size", 0) == 0
            or getattr(face_gray, "size", 0) == 0
        ):
            return "uncertain"

        fh, fw = face_gray.shape[:2]
        if fh < 36 or fw < 36:
            return "uncertain"

        eye_boxes: list[tuple[int, int, int, int]] = []
        for cascade in self._eye_cascades:
            try:
                found = cascade.detectMultiScale(
                    face_gray,
                    scaleFactor=1.12,
                    minNeighbors=5,
                    minSize=(10, 10),
                )
            except Exception:
                continue

            for (x, y, w, h) in found:
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                if w <= 0 or h <= 0:
                    continue
                # Eye candidates should stay in upper part of face.
                if y > int(fh * 0.62):
                    continue
                if h > int(fh * 0.45):
                    continue
                eye_boxes.append((x, y, w, h))

        eye_boxes = self._dedupe_faces(eye_boxes)
        if not eye_boxes:
            return "uncertain"

        # Use up to two largest eye candidates.
        eye_boxes = sorted(eye_boxes, key=lambda item: item[2] * item[3], reverse=True)[:2]
        weighted_h = 0.0
        weighted_s = 0.0
        weighted_v = 0.0
        total_weight = 0.0

        for x, y, w, h in eye_boxes:
            ix1 = max(0, x + int(w * 0.22))
            ix2 = min(fw, x + int(w * 0.78))
            iy1 = max(0, y + int(h * 0.30))
            iy2 = min(fh, y + int(h * 0.85))
            if ix1 >= ix2 or iy1 >= iy2:
                continue

            iris_roi = face_bgr[iy1:iy2, ix1:ix2]
            if getattr(iris_roi, "size", 0) == 0:
                continue

            hsv = cv2.cvtColor(iris_roi, cv2.COLOR_BGR2HSV)
            # Focus on darker/saturated iris-like pixels, ignore highlights/skin.
            mask = cv2.inRange(hsv, (0, 22, 18), (180, 255, 210))
            count = int(cv2.countNonZero(mask))
            if count < 8:
                continue

            mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=mask)
            weight = float(count)
            weighted_h += float(mean_h) * weight
            weighted_s += float(mean_s) * weight
            weighted_v += float(mean_v) * weight
            total_weight += weight

        if total_weight <= 0.0:
            return "uncertain"

        hue = weighted_h / total_weight
        sat = weighted_s / total_weight
        val = weighted_v / total_weight
        return self._classify_eye_hsv(hue, sat, val)

    def _face_descriptions(self, frame, gray, faces) -> tuple[str, ...]:
        if gray is None or frame is None or len(faces) == 0:
            return tuple()

        height, width = gray.shape[:2]
        frame_area = float(max(1, width * height))
        sorted_faces = sorted(faces, key=lambda item: item[0] + (item[2] * 0.5))

        descriptions: list[str] = []
        for idx, (x, y, w, h) in enumerate(sorted_faces, start=1):
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            if w <= 0 or h <= 0:
                continue

            x2 = min(width, x + w)
            y2 = min(height, y + h)
            x1 = max(0, x)
            y1 = max(0, y)
            if x1 >= x2 or y1 >= y2:
                continue

            cx_ratio = (x1 + ((x2 - x1) / 2.0)) / float(max(1, width))
            cy_ratio = (y1 + ((y2 - y1) / 2.0)) / float(max(1, height))
            area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area

            roi_gray = gray[y1:y2, x1:x2]
            roi_color = frame[y1:y2, x1:x2]
            face_light = self._face_light_label(float(roi_gray.mean())) if roi_gray.size > 0 else "unknown"
            expression = "neutral expression"
            if self._smile_cascade is not None and roi_gray.size > 0:
                try:
                    smiles = self._smile_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.7,
                        minNeighbors=20,
                        minSize=(18, 18),
                    )
                    if len(smiles) > 0:
                        expression = "possibly smiling"
                except Exception:
                    pass

            skin_tone = self._estimate_skin_tone(roi_color)
            hair_color = self._estimate_hair_color(frame, x1, y1, x2, y2)
            eye_color = self._estimate_eye_color(roi_color, roi_gray)

            descriptions.append(
                f"Face {idx}: {self._horizontal_label(cx_ratio)}, {self._vertical_label(cy_ratio)}, "
                f"{self._distance_label(area_ratio)}, {face_light} lighting, {expression}, "
                f"estimated skin tone {skin_tone}, hair color {hair_color}, eye color {eye_color}."
            )

        return tuple(descriptions)

    def _analyze_scene(self, frame, gray, face_count: int, motion: float, face_descriptions: tuple[str, ...]) -> dict:
        if cv2 is None:
            return {
                "people_count": face_count,
                "scene_label": "unknown",
                "scene_confidence": 0.0,
                "light_level": "unknown",
                "motion_level": "low",
                "dominant_color": "unknown",
                "objects": tuple(),
                "face_descriptions": tuple(),
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
        if face_descriptions:
            summary_bits.append("faces: " + " | ".join(face_descriptions[:3]))
        summary = "; ".join(summary_bits)

        return {
            "people_count": people_count,
            "scene_label": scene_label,
            "scene_confidence": max(0.0, min(1.0, scene_confidence)),
            "light_level": light_level,
            "motion_level": motion_level,
            "dominant_color": dominant_color,
            "objects": unique_objects,
            "face_descriptions": face_descriptions,
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
                face_descriptions=tuple(scene.get("face_descriptions", tuple())),
                summary=str(scene.get("summary", "")),
                timestamp=time.time(),
            )

    def _loop(self) -> None:
        if cv2 is None or self._capture is None or not self._cascades:
            return

        while not self._stop_event.is_set():
            self._frame_index += 1

            ok, frame = self._capture.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray_raw)

            motion = 0.0
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                motion = float(diff.mean()) / 255.0
            self._prev_gray = gray

            faces = self._detect_faces(gray)
            if not faces:
                faces = self._detect_faces(gray_raw)

            face_descriptions = self._face_descriptions(frame, gray_raw, faces)
            scene = self._analyze_scene(frame, gray_raw, len(faces), motion, face_descriptions)

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

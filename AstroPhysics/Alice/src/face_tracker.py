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
    hand_found: bool = False
    hand_count: int = 0
    hand_x: float = 0.0
    hand_y: float = 0.0
    hand_backend: str = "none"
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
        self._hand_backend = "none"
        self._mp_hands = None
        self._hand_x = 0.0
        self._hand_y = 0.0
        self._hand_visible = False

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
        self._setup_hand_detector()
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

    def _setup_hand_detector(self) -> None:
        self._hand_backend = "skin"
        self._mp_hands = None
        try:
            import mediapipe as mp  # type: ignore

            self._mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.45,
            )
            self._hand_backend = "mediapipe"
        except Exception:
            self._mp_hands = None
            self._hand_backend = "skin"

    @staticmethod
    def _normalize_point(cx: float, cy: float, frame_w: int, frame_h: int) -> tuple[float, float]:
        x_norm = ((cx / max(1, frame_w)) - 0.5) * 2.0
        y_norm = ((cy / max(1, frame_h)) - 0.5) * 2.0
        return (
            max(-1.0, min(1.0, x_norm)),
            max(-1.0, min(1.0, y_norm)),
        )

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

    def _detect_hands_mediapipe(
        self,
        frame: object,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, float, float]:
        if self._mp_hands is None or self._cv2 is None:
            return 0, 0.0, 0.0
        cv2 = self._cv2
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._mp_hands.process(rgb)
        except Exception:
            return 0, 0.0, 0.0

        multi = getattr(result, "multi_hand_landmarks", None)
        if not multi:
            return 0, 0.0, 0.0

        centers: list[tuple[float, float]] = []
        for hand_landmarks in multi[:2]:
            points = getattr(hand_landmarks, "landmark", None)
            if not points:
                continue
            xs = [float(p.x) for p in points]
            ys = [float(p.y) for p in points]
            if not xs or not ys:
                continue
            cx = max(0.0, min(1.0, sum(xs) / len(xs))) * frame_w
            cy = max(0.0, min(1.0, sum(ys) / len(ys))) * frame_h
            centers.append((cx, cy))

        if not centers:
            return 0, 0.0, 0.0

        cx = sum(x for x, _ in centers) / len(centers)
        cy = sum(y for _, y in centers) / len(centers)
        hx, hy = self._normalize_point(cx, cy, frame_w, frame_h)
        return len(centers), hx, hy

    def _detect_hands_skin(
        self,
        frame: object,
        frame_w: int,
        frame_h: int,
        face_box: tuple[int, int, int, int] | None,
    ) -> tuple[int, float, float]:
        if self._cv2 is None:
            return 0, 0.0, 0.0
        cv2 = self._cv2

        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        except Exception:
            return 0, 0.0, 0.0

        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin = cv2.GaussianBlur(skin, (5, 5), 0)
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours_result = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_result) == 2:
            contours = contours_result[0]
        else:
            contours = contours_result[1]

        if face_box is not None:
            fx, fy, fw, fh = face_box
            face_bounds = (
                fx - fw * 0.25,
                fy - fh * 0.25,
                fx + fw * 1.25,
                fy + fh * 1.25,
            )
        else:
            face_bounds = None

        candidates: list[tuple[float, float, float]] = []
        min_area = max(2200.0, frame_w * frame_h * 0.0075)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w * 0.5
            cy = y + h * 0.5
            if face_bounds is not None:
                left, top, right, bottom = face_bounds
                if left <= cx <= right and top <= cy <= bottom:
                    continue

            aspect = h / max(1.0, float(w))
            if aspect < 0.45 or aspect > 3.2:
                continue

            hull = cv2.convexHull(contour)
            hull_area = max(1.0, float(cv2.contourArea(hull)))
            solidity = area / hull_area
            if solidity > 0.97:
                continue

            candidates.append((area, cx, cy))

        if not candidates:
            return 0, 0.0, 0.0

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[:2]
        cx = sum(item[1] for item in selected) / len(selected)
        cy = sum(item[2] for item in selected) / len(selected)
        hx, hy = self._normalize_point(cx, cy, frame_w, frame_h)
        return len(selected), hx, hy

    def _detect_hands(
        self,
        frame: object,
        frame_w: int,
        frame_h: int,
        face_box: tuple[int, int, int, int] | None,
    ) -> tuple[int, float, float]:
        if self._hand_backend == "mediapipe":
            count, hx, hy = self._detect_hands_mediapipe(frame, frame_w, frame_h)
            if count > 0:
                return count, hx, hy
        return self._detect_hands_skin(frame, frame_w, frame_h, face_box)

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

                face_found = False
                x_norm = 0.0
                y_norm = 0.0
                if selected is None:
                    self._lost_frames += 1
                    if self._lost_frames > 80:
                        self._owner_signature = None
                else:
                    self._lost_frames = 0
                    face_found = True
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
                    x_norm, y_norm = self._normalize_point(cx, cy, frame_w, frame_h)

                hand_count, hand_x, hand_y = self._detect_hands(
                    frame,
                    frame_w,
                    frame_h,
                    selected,
                )
                if hand_count > 0:
                    if not self._hand_visible:
                        self._hand_x = hand_x
                        self._hand_y = hand_y
                    else:
                        alpha = 0.72
                        self._hand_x = self._hand_x * alpha + hand_x * (1.0 - alpha)
                        self._hand_y = self._hand_y * alpha + hand_y * (1.0 - alpha)
                    self._hand_visible = True
                else:
                    self._hand_x *= 0.85
                    self._hand_y *= 0.85
                    self._hand_visible = False

                self._set_latest(
                    FaceObservation(
                        found=face_found,
                        x=x_norm,
                        y=y_norm,
                        face_count=len(faces),
                        hand_found=hand_count > 0 and self._hand_visible,
                        hand_count=hand_count,
                        hand_x=self._hand_x if hand_count > 0 else 0.0,
                        hand_y=self._hand_y if hand_count > 0 else 0.0,
                        hand_backend=self._hand_backend,
                        owner_locked=self._owner_signature is not None,
                        owner_name=self.owner_name if self._owner_signature is not None else None,
                    )
                )
                time.sleep(0.015)
        finally:
            if self._mp_hands is not None:
                try:
                    self._mp_hands.close()
                except Exception:
                    pass
            cap.release()
            self._running = False

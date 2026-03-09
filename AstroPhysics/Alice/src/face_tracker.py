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
    eye_found: bool = False
    eye_x: float = 0.0
    eye_y: float = 0.0
    face_count: int = 0
    hand_found: bool = False
    hand_count: int = 0
    hand_x: float = 0.0
    hand_y: float = 0.0
    scene_brightness: float = 0.0
    scene_motion: float = 0.0
    scene_note: str = "unknown"
    hand_backend: str = "none"
    owner_locked: bool = False
    owner_name: str | None = None


class FaceTracker:
    def __init__(
        self,
        *,
        camera_index: int = 0,
        owner_name: str = "You",
        preview: bool = True,
        preview_title: str = "Alice Camera",
    ) -> None:
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
        self._hand_lost_frames = 0
        self._eye_x = 0.0
        self._eye_y = 0.0
        self._eye_visible = False
        self._eye_cascade = None
        self._last_small_gray = None
        self._scene_motion = 0.0
        self._preview_enabled = preview
        self._preview_title = preview_title
        self._preview_png: bytes | None = None
        self._preview_updated_at = 0.0

    @staticmethod
    def _scene_note(brightness: float, motion: float) -> str:
        if brightness < 0.30:
            light = "dim"
        elif brightness > 0.70:
            light = "bright"
        else:
            light = "balanced"

        if motion < 0.040:
            motion_desc = "still"
        elif motion < 0.090:
            motion_desc = "light movement"
        else:
            motion_desc = "active movement"
        return f"{light} lighting, {motion_desc}"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def preview_enabled(self) -> bool:
        return self._preview_enabled

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

    def get_preview_png(self) -> bytes | None:
        with self._lock:
            return self._preview_png

    def _set_latest(self, value: FaceObservation) -> None:
        with self._lock:
            self._latest = value

    def _set_preview_png(self, data: bytes | None) -> None:
        with self._lock:
            self._preview_png = data
            self._preview_updated_at = time.monotonic()

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

    def _detect_eyes(
        self,
        gray: object,
        face_box: tuple[int, int, int, int] | None,
        frame_w: int,
        frame_h: int,
    ) -> tuple[bool, float, float]:
        if self._eye_cascade is None or face_box is None or self._cv2 is None:
            return False, 0.0, 0.0
        cv2 = self._cv2
        x, y, w, h = face_box
        if w < 60 or h < 60:
            return False, 0.0, 0.0

        top = y + int(0.12 * h)
        bottom = y + int(0.62 * h)
        left = x + int(0.06 * w)
        right = x + int(0.94 * w)
        roi = gray[top:bottom, left:right]
        if roi is None or roi.size == 0:
            return False, 0.0, 0.0

        try:
            roi = cv2.equalizeHist(roi)
        except Exception:
            return False, 0.0, 0.0

        min_side = int(max(14, min(w, h) * 0.12))
        eyes = self._eye_cascade.detectMultiScale(
            roi,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(min_side, min_side),
            maxSize=(int(0.46 * w), int(0.42 * h)),
        )
        if len(eyes) == 0:
            return False, 0.0, 0.0

        eye_boxes = sorted(
            [(int(ex), int(ey), int(ew), int(eh)) for ex, ey, ew, eh in eyes],
            key=lambda item: item[2] * item[3],
            reverse=True,
        )[:2]

        centers_x: list[float] = []
        centers_y: list[float] = []
        for ex, ey, ew, eh in eye_boxes:
            centers_x.append(left + ex + ew * 0.5)
            centers_y.append(top + ey + eh * 0.5)
        if not centers_x:
            return False, 0.0, 0.0

        cx = sum(centers_x) / len(centers_x)
        cy = sum(centers_y) / len(centers_y)
        nx, ny = self._normalize_point(cx, cy, frame_w, frame_h)
        return True, nx, ny

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
        self._eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if self._eye_cascade.empty():
            self._eye_cascade = None

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
                    # Keep face lock briefly through hand motion or short occlusions.
                    if self._owner_signature is not None and self._lost_frames <= 14:
                        face_found = True
                        ox, oy, _, _ = self._owner_signature
                        x_norm, y_norm = self._normalize_point(ox, oy, frame_w, frame_h)
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
                    self._eye_visible = False

                hand_count, hand_x, hand_y = self._detect_hands(
                    frame,
                    frame_w,
                    frame_h,
                    selected,
                )
                if hand_count > 0:
                    self._hand_lost_frames = 0
                    if not self._hand_visible:
                        self._hand_x = hand_x
                        self._hand_y = hand_y
                    else:
                        alpha = 0.72
                        self._hand_x = self._hand_x * alpha + hand_x * (1.0 - alpha)
                        self._hand_y = self._hand_y * alpha + hand_y * (1.0 - alpha)
                    self._hand_visible = True
                else:
                    self._hand_lost_frames += 1
                    if self._hand_lost_frames <= 9:
                        self._hand_x *= 0.92
                        self._hand_y *= 0.92
                        self._hand_visible = True
                    else:
                        self._hand_x *= 0.85
                        self._hand_y *= 0.85
                        self._hand_visible = False

                brightness = float(gray.mean()) / 255.0
                small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (5, 5), 0)
                if self._last_small_gray is None:
                    instant_motion = 0.0
                else:
                    diff = cv2.absdiff(small, self._last_small_gray)
                    instant_motion = float(diff.mean()) / 255.0
                self._last_small_gray = small
                self._scene_motion = self._scene_motion * 0.84 + instant_motion * 0.16
                scene_note = self._scene_note(brightness, self._scene_motion)

                self._set_latest(
                    FaceObservation(
                        found=face_found,
                        x=x_norm,
                        y=y_norm,
                        eye_found=False,
                        eye_x=0.0,
                        eye_y=0.0,
                        face_count=len(faces),
                        hand_found=self._hand_visible,
                        hand_count=max(hand_count, 1) if self._hand_visible else 0,
                        hand_x=self._hand_x if self._hand_visible else 0.0,
                        hand_y=self._hand_y if self._hand_visible else 0.0,
                        scene_brightness=brightness,
                        scene_motion=self._scene_motion,
                        scene_note=scene_note,
                        hand_backend=self._hand_backend,
                        owner_locked=self._owner_signature is not None,
                        owner_name=self.owner_name if self._owner_signature is not None else None,
                    )
                )
                if self._preview_enabled:
                    try:
                        preview = frame.copy()
                        for idx, (fx, fy, fw, fh) in enumerate(faces):
                            color = (110, 90, 230)
                            thickness = 1
                            if selected is not None and (fx, fy, fw, fh) == selected:
                                color = (70, 220, 120)
                                thickness = 2
                            cv2.rectangle(preview, (fx, fy), (fx + fw, fy + fh), color, thickness)
                            if idx < 3:
                                cv2.putText(
                                    preview,
                                    f"face {idx + 1}",
                                    (fx, max(16, fy - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    color,
                                    1,
                                    cv2.LINE_AA,
                                )

                        if self._hand_visible:
                            hx = int((self._hand_x * 0.5 + 0.5) * frame_w)
                            hy = int((self._hand_y * 0.5 + 0.5) * frame_h)
                            cv2.circle(preview, (hx, hy), 11, (90, 215, 255), 2)
                            hand_text = f"hands: {max(1, hand_count)}"
                            cv2.putText(
                                preview,
                                hand_text,
                                (hx + 10, max(24, hy - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.46,
                                (90, 215, 255),
                                1,
                                cv2.LINE_AA,
                            )

                        status_lines = [
                            f"tracking: {self.owner_name if face_found else 'searching face'}",
                            f"scene: {scene_note}",
                            f"brightness: {brightness:.2f}  motion: {self._scene_motion:.2f}",
                        ]
                        y0 = 22
                        for line in status_lines:
                            cv2.putText(
                                preview,
                                line,
                                (12, y0),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.52,
                                (210, 230, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            y0 += 20

                        ok_png, encoded = cv2.imencode(".png", preview)
                        if ok_png:
                            self._set_preview_png(encoded.tobytes())
                    except Exception:
                        self._set_preview_png(None)
                time.sleep(0.015)
        finally:
            if self._mp_hands is not None:
                try:
                    self._mp_hands.close()
                except Exception:
                    pass
            cap.release()
            self._running = False

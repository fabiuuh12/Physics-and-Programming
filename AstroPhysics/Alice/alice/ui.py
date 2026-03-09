from __future__ import annotations

import math
import time
from typing import Optional

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - platform-specific import
    tk = None
    ttk = None


class AliceUI:
    def __init__(self) -> None:
        self._running = False
        self._state = "idle"
        self._status_text = "Online"

        self._blink_value = 1.0
        self._next_blink_at = time.monotonic() + 2.3
        self._blink_stage = 0
        self._blink_started_at = 0.0

        self._speak_phase = 0.0
        self._idle_phase = 0.0
        self._face_offset_y = 0.0
        self._mouth_open = 3.2

        self._last_frame_time = time.monotonic()

        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._gaze_vx = 0.0
        self._gaze_vy = 0.0

        self._target_gaze_x = 0.0
        self._target_gaze_y = 0.0

        self._smoothed_track_x = 0.0
        self._smoothed_track_y = 0.0
        self._track_blend = 0.0

        self._raw_face_x = 0.0
        self._raw_face_y = 0.0
        self._vision_found = False
        self._vision_face_count = 0
        self._vision_scene_label = "unknown"
        self._vision_scene_confidence = 0.0

        self._avatar_offset_y = 24.0
        self._head_x = 0.0
        self._head_y = 0.0
        self._head_tilt = 0.0

        self._emotion_name = "neutral"
        self._emotion_intensity = 0.0

        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._status_var: Optional[tk.StringVar] = None
        self._chat_text: Optional[tk.Text] = None

        self._chat_lines = 0

        self._aura_ring = None
        self._state_ring = None
        self._orbit_dot = None
        self._focus_text = None

        self._face_base = None
        self._face_shade = None

        self._left_eye_white = None
        self._right_eye_white = None
        self._left_iris = None
        self._right_iris = None
        self._left_pupil = None
        self._right_pupil = None
        self._left_highlight = None
        self._right_highlight = None

        self._left_upper_lid = None
        self._right_upper_lid = None
        self._left_lower_lid = None
        self._right_lower_lid = None
        self._left_brow = None
        self._right_brow = None

        self._nose_bridge = None
        self._nose_tip = None

        self._mouth_inner = None
        self._mouth_teeth = None
        self._mouth_upper = None
        self._mouth_lower = None
        self._lip_highlight = None

    def start(self) -> bool:
        if tk is None or ttk is None:
            self._running = False
            return False

        try:
            root = tk.Tk()
        except Exception:
            self._running = False
            return False

        self._root = root
        root.title("Alice")
        root.geometry("520x700")
        root.minsize(420, 620)
        root.configure(bg="#0c1117")
        root.protocol("WM_DELETE_WINDOW", self.stop)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Alice.TFrame", background="#0c1117")
        style.configure("AliceStatus.TLabel", background="#0c1117", foreground="#8db8ff")

        shell = ttk.Frame(root, style="Alice.TFrame", padding=12)
        shell.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(shell, width=460, height=430, bg="#0f1724", highlightthickness=0)
        self._canvas.pack(fill="x")

        self._status_var = tk.StringVar(value=self._status_text)
        status_label = ttk.Label(
            shell,
            textvariable=self._status_var,
            style="AliceStatus.TLabel",
            anchor="w",
            font=("Avenir Next", 12),
        )
        status_label.pack(fill="x", pady=(10, 6))

        self._chat_text = tk.Text(
            shell,
            height=12,
            bg="#101a29",
            fg="#dce8ff",
            insertbackground="#dce8ff",
            relief="flat",
            wrap="word",
            font=("Avenir Next", 12),
            padx=10,
            pady=10,
        )
        self._chat_text.pack(fill="both", expand=True)
        self._chat_text.configure(state="disabled")

        self._draw_static_face()
        self._draw_dynamic_face()
        self._running = True
        return True

    def _draw_static_face(self) -> None:
        if self._canvas is None:
            return

        c = self._canvas
        oy = self._avatar_offset_y

        c.create_oval(18, -42, 442, 382, fill="#101a29", outline="#1b2f4a", width=1)
        self._aura_ring = c.create_oval(36, -24 + oy, 424, 364 + oy, fill="", outline="#274365", width=1, dash=(2, 6))
        c.create_oval(48, -8 + oy, 412, 352 + oy, fill="", outline="#1e3455", width=1, dash=(3, 5))
        self._state_ring = c.create_oval(55, -1 + oy, 405, 349 + oy, outline="#4f7fb8", width=3)
        self._orbit_dot = c.create_oval(228, -16 + oy, 236, -8 + oy, fill="#7fbfff", outline="")

        self._face_base = c.create_oval(92, 38 + oy, 368, 318 + oy, fill="#0e2034", outline="#3f88b7", width=2)
        self._face_shade = c.create_oval(108, 56 + oy, 352, 300 + oy, fill="", outline="#6eb6de", width=1, dash=(4, 6))

        self._focus_text = c.create_text(230, 390, text="NO FACE DETECTED", fill="#95d7ff", font=("Avenir Next", 11))

    def _draw_dynamic_face(self) -> None:
        if self._canvas is None:
            return

        c = self._canvas
        oy = self._avatar_offset_y

        self._left_eye_white = c.create_oval(146, 122 + oy, 208, 176 + oy, fill="#0f2539", outline="#74c6f0", width=2)
        self._right_eye_white = c.create_oval(252, 122 + oy, 314, 176 + oy, fill="#0f2539", outline="#74c6f0", width=2)
        self._left_iris = c.create_oval(166, 136 + oy, 190, 160 + oy, fill="#63dcff", outline="")
        self._right_iris = c.create_oval(272, 136 + oy, 296, 160 + oy, fill="#63dcff", outline="")
        self._left_pupil = c.create_oval(174, 143 + oy, 182, 151 + oy, fill="#05121c", outline="")
        self._right_pupil = c.create_oval(280, 143 + oy, 288, 151 + oy, fill="#05121c", outline="")
        self._left_highlight = c.create_oval(177, 145 + oy, 180, 148 + oy, fill="#d9f6ff", outline="")
        self._right_highlight = c.create_oval(283, 145 + oy, 286, 148 + oy, fill="#d9f6ff", outline="")

        self._left_upper_lid = c.create_line(146, 131 + oy, 177, 126 + oy, 208, 131 + oy, fill="#7fcff1", width=2, smooth=True)
        self._right_upper_lid = c.create_line(252, 131 + oy, 283, 126 + oy, 314, 131 + oy, fill="#7fcff1", width=2, smooth=True)
        self._left_lower_lid = c.create_line(146, 167 + oy, 177, 171 + oy, 208, 167 + oy, fill="#5ca2ce", width=2, smooth=True)
        self._right_lower_lid = c.create_line(252, 167 + oy, 283, 171 + oy, 314, 167 + oy, fill="#5ca2ce", width=2, smooth=True)

        self._left_brow = c.create_line(142, 113 + oy, 176, 100 + oy, 212, 112 + oy, fill="#6eb6de", width=3, smooth=True)
        self._right_brow = c.create_line(248, 112 + oy, 284, 100 + oy, 318, 113 + oy, fill="#6eb6de", width=3, smooth=True)

        self._nose_bridge = c.create_line(230, 146 + oy, 230, 196 + oy, fill="#3d7fa9", width=2)
        self._nose_tip = c.create_line(220, 198 + oy, 230, 202 + oy, 240, 198 + oy, fill="#3d7fa9", width=2, smooth=True)

        self._mouth_inner = c.create_oval(192, 248 + oy, 268, 266 + oy, fill="#123553", outline="#4eb9e8", width=1)
        self._mouth_teeth = c.create_rectangle(200, 250 + oy, 260, 254 + oy, fill="#b7ecff", outline="")
        self._mouth_upper = c.create_line(176, 254 + oy, 204, 250 + oy, 230, 248 + oy, 256, 250 + oy, 284, 254 + oy, fill="#6ad8ff", width=3, smooth=True)
        self._mouth_lower = c.create_line(176, 254 + oy, 204, 260 + oy, 230, 264 + oy, 256, 260 + oy, 284, 254 + oy, fill="#3ea0d2", width=3, smooth=True)
        self._lip_highlight = c.create_line(186, 252 + oy, 214, 249 + oy, 230, 248 + oy, 246, 249 + oy, 274, 252 + oy, fill="#d6f7ff", width=1, smooth=True)

        self._sync_layer_order()

    def _sync_layer_order(self) -> None:
        if self._canvas is None:
            return

        for item in (
            self._face_base,
            self._face_shade,
            self._nose_bridge,
            self._nose_tip,
            self._left_eye_white,
            self._right_eye_white,
            self._left_iris,
            self._right_iris,
            self._left_pupil,
            self._right_pupil,
            self._left_highlight,
            self._right_highlight,
            self._left_upper_lid,
            self._right_upper_lid,
            self._left_lower_lid,
            self._right_lower_lid,
            self._mouth_inner,
            self._mouth_teeth,
            self._mouth_lower,
            self._mouth_upper,
            self._lip_highlight,
        ):
            if item is not None:
                self._canvas.tag_raise(item)

    def _ring_color_for_state(self) -> str:
        if self._state == "listening":
            return "#6eb8ff"
        if self._state == "thinking":
            return "#f0cd73"
        if self._state == "speaking":
            return "#7be4b7"
        if self._state == "error":
            return "#ff9393"
        if self._state == "offline":
            return "#66778f"
        emotion = self._emotion_name
        if emotion in {"joy", "gratitude", "content", "calm", "serenity", "affection"}:
            return "#68dca8"
        if emotion in {"curiosity", "focus", "anticipation", "determination", "alertness"}:
            return "#7bb8ff"
        if emotion in {"concern", "confusion", "uncertainty", "anxiety"}:
            return "#f2b56f"
        if emotion in {"sadness", "loneliness", "disappointment", "fatigue", "boredom"}:
            return "#90a4c8"
        if emotion in {"frustration", "anger", "overwhelm", "fear"}:
            return "#ff8f8f"
        return "#4f7fb8"

    def _emotion_profile(self) -> str:
        if self._emotion_name in {"joy", "gratitude", "content", "calm", "serenity", "affection", "amusement"}:
            return "positive"
        if self._emotion_name in {"curiosity", "focus", "anticipation", "determination", "alertness", "confidence"}:
            return "focused"
        if self._emotion_name in {"concern", "confusion", "uncertainty", "anxiety", "fear"}:
            return "concerned"
        if self._emotion_name in {"sadness", "loneliness", "disappointment", "fatigue", "boredom", "nostalgia"}:
            return "low"
        if self._emotion_name in {"frustration", "anger", "overwhelm"}:
            return "tense"
        return "neutral"

    def _pose_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            x + self._head_x,
            y + self._avatar_offset_y + self._head_y + self._face_offset_y + (x - 230.0) * self._head_tilt,
        )

    def _update_tracking_target(self) -> None:
        if self._vision_found:
            tx = self._raw_face_x * 0.90
            ty = self._raw_face_y * 0.78

            dx = tx - self._smoothed_track_x
            dy = ty - self._smoothed_track_y
            motion = math.hypot(dx, dy)
            alpha = 0.54 if motion > 0.35 else 0.30
            self._smoothed_track_x += dx * alpha
            self._smoothed_track_y += dy * alpha

            sx = self._smoothed_track_x + 0.14 * self._smoothed_track_x * abs(self._smoothed_track_x)
            sy = self._smoothed_track_y + 0.10 * self._smoothed_track_y * abs(self._smoothed_track_y)

            self._target_gaze_x = max(-1.0, min(1.0, sx))
            self._target_gaze_y = max(-1.0, min(1.0, sy))
            self._track_blend = min(1.0, self._track_blend + 0.14)
            return

        # No face lock: gentle fallback search motion.
        self._smoothed_track_x *= 0.90
        self._smoothed_track_y *= 0.90
        search_x = 0.26 * math.sin(self._idle_phase * 1.8)
        search_y = 0.14 * math.cos(self._idle_phase * 1.3)
        self._target_gaze_x = self._smoothed_track_x + search_x
        self._target_gaze_y = self._smoothed_track_y + search_y
        self._track_blend = max(0.0, self._track_blend - 0.08)

    def _animate_blink(self, now: float) -> None:
        if self._blink_stage == 0 and now >= self._next_blink_at:
            self._blink_stage = 1
            self._blink_started_at = now

        if self._blink_stage == 1:
            t = min((now - self._blink_started_at) / 0.08, 1.0)
            self._blink_value = 1.0 - t
            if t >= 1.0:
                self._blink_stage = 2
                self._blink_started_at = now
        elif self._blink_stage == 2:
            t = min((now - self._blink_started_at) / 0.10, 1.0)
            self._blink_value = t
            if t >= 1.0:
                self._blink_stage = 0
                self._next_blink_at = now + 2.3
        else:
            self._blink_value = 1.0

    def _layout_head_shell(self) -> None:
        if self._canvas is None:
            return

        breath = 1.0 + 0.006 * math.sin(self._idle_phase * 1.3)
        head_cx = 230.0 + self._head_x
        head_cy = 177.0 + self._avatar_offset_y + self._head_y + self._face_offset_y

        face_half_w = 138.0 * breath
        face_half_h = 140.0 * breath

        self._canvas.coords(
            self._face_base,
            head_cx - face_half_w,
            head_cy - face_half_h,
            head_cx + face_half_w,
            head_cy + face_half_h,
        )
        self._canvas.coords(
            self._face_shade,
            head_cx - 124.0 * breath,
            head_cy - 122.0 * breath,
            head_cx + 124.0 * breath,
            head_cy + 122.0 * breath,
        )

        self._canvas.coords(
            self._nose_bridge,
            *self._pose_point(230.0, 146.0),
            *self._pose_point(230.0, 196.0),
        )
        self._canvas.coords(
            self._nose_tip,
            *self._pose_point(220.0, 198.0),
            *self._pose_point(230.0, 202.0),
            *self._pose_point(240.0, 198.0),
        )

    def _layout_eyes(self, dt: float) -> None:
        if self._canvas is None:
            return

        err_x = self._target_gaze_x - self._gaze_x
        err_y = self._target_gaze_y - self._gaze_y
        err_mag = min(1.0, math.hypot(err_x, err_y))

        spring = (11.5 if self._vision_found else 7.2) + err_mag * (5.2 if self._vision_found else 2.0)
        damping = 6.2 if self._vision_found else 4.9

        self._gaze_vx += (err_x * spring - self._gaze_vx * damping) * dt
        self._gaze_vy += (err_y * spring - self._gaze_vy * damping) * dt

        if self._vision_found and err_mag > 0.34:
            boost = min(0.18, err_mag * 0.22)
            self._gaze_vx += err_x * boost
            self._gaze_vy += err_y * boost

        self._gaze_x += self._gaze_vx * dt
        self._gaze_y += self._gaze_vy * dt
        self._gaze_x = max(-1.2, min(1.2, self._gaze_x))
        self._gaze_y = max(-1.2, min(1.2, self._gaze_y))

        blink_h = max(2.0, 52.0 * self._blink_value)
        left_mid_x, left_mid_y = self._pose_point(177.0, 149.0)
        right_mid_x, right_mid_y = self._pose_point(283.0, 149.0)
        half_w = 31.0

        left_top = left_mid_y - blink_h / 2.0
        left_bottom = left_mid_y + blink_h / 2.0
        right_top = right_mid_y - blink_h / 2.0
        right_bottom = right_mid_y + blink_h / 2.0

        self._canvas.coords(self._left_eye_white, left_mid_x - half_w, left_top, left_mid_x + half_w, left_bottom)
        self._canvas.coords(self._right_eye_white, right_mid_x - half_w, right_top, right_mid_x + half_w, right_bottom)

        left_upper_lid_y = left_top + max(2.0, 0.18 * blink_h)
        left_lower_lid_y = left_bottom - max(2.0, 0.18 * blink_h)
        right_upper_lid_y = right_top + max(2.0, 0.18 * blink_h)
        right_lower_lid_y = right_bottom - max(2.0, 0.18 * blink_h)

        if left_lower_lid_y < left_upper_lid_y + 2.0:
            left_lower_lid_y = left_upper_lid_y + 2.0
        if right_lower_lid_y < right_upper_lid_y + 2.0:
            right_lower_lid_y = right_upper_lid_y + 2.0

        left_upper_peak = left_upper_lid_y - max(1.3, 0.12 * blink_h)
        left_lower_valley = left_lower_lid_y + max(1.3, 0.10 * blink_h)
        right_upper_peak = right_upper_lid_y - max(1.3, 0.12 * blink_h)
        right_lower_valley = right_lower_lid_y + max(1.3, 0.10 * blink_h)

        self._canvas.coords(self._left_upper_lid, left_mid_x - half_w, left_upper_lid_y + 0.8, left_mid_x, left_upper_peak, left_mid_x + half_w, left_upper_lid_y + 0.8)
        self._canvas.coords(self._right_upper_lid, right_mid_x - half_w, right_upper_lid_y + 0.8, right_mid_x, right_upper_peak, right_mid_x + half_w, right_upper_lid_y + 0.8)
        self._canvas.coords(self._left_lower_lid, left_mid_x - half_w, left_lower_lid_y - 0.8, left_mid_x, left_lower_valley, left_mid_x + half_w, left_lower_lid_y - 0.8)
        self._canvas.coords(self._right_lower_lid, right_mid_x - half_w, right_lower_lid_y - 0.8, right_mid_x, right_lower_valley, right_mid_x + half_w, right_lower_lid_y - 0.8)

        if blink_h < 8:
            for item in (
                self._left_iris,
                self._right_iris,
                self._left_pupil,
                self._right_pupil,
                self._left_highlight,
                self._right_highlight,
                self._left_lower_lid,
                self._right_lower_lid,
            ):
                self._canvas.itemconfigure(item, state="hidden")
            return

        for item in (
            self._left_iris,
            self._right_iris,
            self._left_pupil,
            self._right_pupil,
            self._left_highlight,
            self._right_highlight,
            self._left_lower_lid,
            self._right_lower_lid,
        ):
            self._canvas.itemconfigure(item, state="normal")

        if self._vision_found:
            micro_scale = 0.70 + 0.30 * (1.0 - self._track_blend)
            micro_x = micro_scale * (0.24 * math.sin(self._idle_phase * 10.4) + 0.16 * math.cos(self._idle_phase * 5.6))
            micro_y = micro_scale * (0.18 * math.cos(self._idle_phase * 8.7))
        else:
            micro_x = 0.8 * math.sin(self._idle_phase * 2.3)
            micro_y = 0.5 * math.cos(self._idle_phase * 1.9)

        px = self._gaze_x * 10.8 + micro_x
        py = self._gaze_y * 8.8 + micro_y
        convergence = 0.7 * self._track_blend

        li_x = left_mid_x + px + convergence
        li_y = left_mid_y + py
        ri_x = right_mid_x + px - convergence
        ri_y = right_mid_y + py

        eye_energy = min(1.0, abs(self._gaze_vx) + abs(self._gaze_vy))
        iris_r = 11.5 + 1.0 * (0.5 + 0.5 * math.sin(self._idle_phase * 2.8))
        if self._state == "thinking":
            iris_r -= 0.8
        pupil_r = 3.8 + 0.8 * eye_energy

        self._canvas.coords(self._left_iris, li_x - iris_r, li_y - iris_r, li_x + iris_r, li_y + iris_r)
        self._canvas.coords(self._right_iris, ri_x - iris_r, ri_y - iris_r, ri_x + iris_r, ri_y + iris_r)
        self._canvas.coords(self._left_pupil, li_x - pupil_r, li_y - pupil_r, li_x + pupil_r, li_y + pupil_r)
        self._canvas.coords(self._right_pupil, ri_x - pupil_r, ri_y - pupil_r, ri_x + pupil_r, ri_y + pupil_r)
        self._canvas.coords(self._left_highlight, li_x - 1, li_y - 1, li_x + 2, li_y + 2)
        self._canvas.coords(self._right_highlight, ri_x - 1, ri_y - 1, ri_x + 2, ri_y + 2)

    def _layout_brows(self) -> None:
        if self._canvas is None:
            return

        profile = self._emotion_profile()
        intensity = max(0.0, min(1.0, self._emotion_intensity))

        if self._state == "thinking" or profile == "focused":
            left = ((142.0, 110.0), (176.0, 104.0), (212.0, 114.0))
            right = ((248.0, 114.0), (284.0, 104.0), (318.0, 110.0))
        elif profile == "positive":
            lift = 3.0 + intensity * 2.5
            left = ((142.0, 115.0), (176.0, 102.0 - lift), (212.0, 114.0))
            right = ((248.0, 114.0), (284.0, 102.0 - lift), (318.0, 115.0))
        elif profile == "concerned":
            pinch = 3.0 + intensity * 3.0
            left = ((142.0, 113.0), (176.0, 102.0 + pinch), (212.0, 111.0))
            right = ((248.0, 111.0), (284.0, 102.0 + pinch), (318.0, 113.0))
        elif profile == "low":
            drop = 2.0 + intensity * 3.0
            left = ((142.0, 116.0 + drop), (176.0, 110.0 + drop), (212.0, 116.0 + drop))
            right = ((248.0, 116.0 + drop), (284.0, 110.0 + drop), (318.0, 116.0 + drop))
        elif profile == "tense":
            left = ((142.0, 118.0), (176.0, 121.0), (212.0, 110.0))
            right = ((248.0, 110.0), (284.0, 121.0), (318.0, 118.0))
        elif self._state == "speaking":
            left = ((142.0, 114.0), (176.0, 99.0), (212.0, 112.0))
            right = ((248.0, 112.0), (284.0, 99.0), (318.0, 114.0))
        elif self._state == "error":
            left = ((142.0, 116.0), (176.0, 120.0), (212.0, 112.0))
            right = ((248.0, 112.0), (284.0, 120.0), (318.0, 116.0))
        else:
            left = ((142.0, 113.0), (176.0, 100.0), (212.0, 112.0))
            right = ((248.0, 112.0), (284.0, 100.0), (318.0, 113.0))

        lx1, ly1 = self._pose_point(*left[0])
        lx2, ly2 = self._pose_point(*left[1])
        lx3, ly3 = self._pose_point(*left[2])
        rx1, ry1 = self._pose_point(*right[0])
        rx2, ry2 = self._pose_point(*right[1])
        rx3, ry3 = self._pose_point(*right[2])

        self._canvas.coords(self._left_brow, lx1, ly1, lx2, ly2, lx3, ly3)
        self._canvas.coords(self._right_brow, rx1, ry1, rx2, ry2, rx3, ry3)

    def _layout_mouth(self, dt: float) -> None:
        if self._canvas is None:
            return

        profile = self._emotion_profile()
        intensity = max(0.0, min(1.0, self._emotion_intensity))

        if self._state == "speaking":
            self._speak_phase += dt * 10.0
            target_open = 7.0 + 13.0 * (0.5 + 0.5 * math.sin(self._speak_phase))
            curve = 2.6
        elif self._state == "listening":
            target_open = 4.6
            curve = 1.1
        elif self._state == "thinking":
            target_open = 3.8
            curve = 0.6
        elif self._state == "error":
            target_open = 2.2
            curve = -2.8
        else:
            target_open = 2.8 + intensity * 0.8
            if profile == "positive":
                curve = 2.3 + intensity * 1.4
            elif profile == "focused":
                curve = 1.0 + intensity * 0.4
            elif profile == "concerned":
                curve = 0.2 - intensity * 1.2
            elif profile == "low":
                curve = -0.5 - intensity * 1.4
                target_open = max(2.0, target_open - 0.5)
            elif profile == "tense":
                curve = -1.4 - intensity * 1.3
                target_open = 2.2 + intensity * 0.4
            else:
                curve = 1.8 if self._vision_found else 1.1

        smoothing = min(1.0, dt * 12.0)
        self._mouth_open += (target_open - self._mouth_open) * smoothing
        open_val = max(1.8, self._mouth_open)

        y = 258.0 + 0.35 * self._head_y

        ux1, uy1 = self._pose_point(176.0, y)
        ux2, uy2 = self._pose_point(204.0, y - 4.0 - curve)
        ux3, uy3 = self._pose_point(230.0, y - 5.0 - curve * 0.45)
        ux4, uy4 = self._pose_point(256.0, y - 4.0 - curve)
        ux5, uy5 = self._pose_point(284.0, y)
        self._canvas.coords(self._mouth_upper, ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4, ux5, uy5)

        lx1, ly1 = self._pose_point(176.0, y)
        lx2, ly2 = self._pose_point(204.0, y + open_val * 0.62)
        lx3, ly3 = self._pose_point(230.0, y + open_val)
        lx4, ly4 = self._pose_point(256.0, y + open_val * 0.62)
        lx5, ly5 = self._pose_point(284.0, y)
        self._canvas.coords(self._mouth_lower, lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, lx5, ly5)

        hx1, hy1 = self._pose_point(186.0, y - 1.5)
        hx2, hy2 = self._pose_point(214.0, y - 4.5 - curve * 0.55)
        hx3, hy3 = self._pose_point(230.0, y - 4.2 - curve * 0.2)
        hx4, hy4 = self._pose_point(246.0, y - 4.5 - curve * 0.55)
        hx5, hy5 = self._pose_point(274.0, y - 1.5)
        self._canvas.coords(self._lip_highlight, hx1, hy1, hx2, hy2, hx3, hy3, hx4, hy4, hx5, hy5)

        inner_left, inner_right = 194.0, 266.0
        inner_top = y - 2.0
        inner_bottom = y + max(3.0, open_val * 0.95)

        if open_val < 3.1:
            self._canvas.itemconfigure(self._mouth_inner, state="hidden")
            self._canvas.itemconfigure(self._mouth_teeth, state="hidden")
        else:
            self._canvas.itemconfigure(self._mouth_inner, state="normal")
            self._canvas.coords(self._mouth_inner, *self._pose_point(inner_left, inner_top), *self._pose_point(inner_right, inner_bottom))

            if 3.1 <= open_val < 7.8:
                teeth_top = inner_top + 2.0
                teeth_bottom = teeth_top + min(3.0, open_val * 0.18)
                self._canvas.coords(self._mouth_teeth, *self._pose_point(inner_left + 6.0, teeth_top), *self._pose_point(inner_right - 6.0, teeth_bottom))
                self._canvas.itemconfigure(self._mouth_teeth, state="normal")
            else:
                self._canvas.itemconfigure(self._mouth_teeth, state="hidden")

    def _layout_orbit(self) -> None:
        if self._canvas is None or self._orbit_dot is None:
            return

        orbit_x = 230.0 + 186.0 * math.cos(self._idle_phase * 0.42)
        orbit_y = 170.0 + self._avatar_offset_y + 186.0 * math.sin(self._idle_phase * 0.42)
        self._canvas.coords(self._orbit_dot, orbit_x - 4.0, orbit_y - 4.0, orbit_x + 4.0, orbit_y + 4.0)

    def _update_vision_label(self) -> None:
        if self._canvas is None or self._focus_text is None:
            return

        if self._vision_found:
            faces = "face" if self._vision_face_count == 1 else "faces"
            scene = self._vision_scene_label.upper() if self._vision_scene_label else "UNKNOWN"
            text = f"I CAN SEE YOU ({self._vision_face_count} {faces}) | {scene}"
            self._canvas.itemconfigure(self._focus_text, text=text, fill="#95d7ff")
        else:
            scene = self._vision_scene_label.upper() if self._vision_scene_label else "UNKNOWN"
            text = f"NO FACE DETECTED | {scene}"
            self._canvas.itemconfigure(self._focus_text, text=text, fill="#6e87a5")

    def _frame(self) -> None:
        if self._canvas is None:
            return

        now = time.monotonic()
        dt = now - self._last_frame_time
        dt = min(max(dt, 1.0 / 240.0), 1.0 / 24.0)
        self._last_frame_time = now

        self._idle_phase += dt * (2.2 if self._state == "speaking" else 1.5)
        # Keep this oscillatory; never use a linear term here or the face will drift downward.
        self._face_offset_y = math.sin(self._idle_phase) * 0.8

        self._update_tracking_target()
        self._animate_blink(now)

        tracked = self._track_blend if self._vision_found else 0.0
        target_head_x = self._gaze_x * (7.2 if self._vision_found else 4.6) * tracked + 0.24 * math.sin(self._idle_phase * 0.9)
        target_head_y = self._gaze_y * (2.8 if self._vision_found else 1.7) * tracked + 0.35 * math.sin(self._idle_phase * 1.1)
        target_head_tilt = -0.058 * self._gaze_x * tracked + 0.010 * math.sin(self._idle_phase * 1.4)

        if self._state == "speaking":
            target_head_y += 1.2 * math.sin(self._idle_phase * 3.5)
            target_head_tilt += 0.010 * math.sin(self._idle_phase * 3.1)
        elif self._state == "thinking":
            target_head_tilt += 0.013 * math.sin(self._idle_phase * 2.6)

        blend = min(1.0, dt * 9.0)
        self._head_x += (target_head_x - self._head_x) * blend
        self._head_y += (target_head_y - self._head_y) * blend
        self._head_tilt += (target_head_tilt - self._head_tilt) * min(1.0, dt * 10.0)

        self._layout_head_shell()
        self._layout_eyes(dt)
        self._layout_brows()
        self._layout_mouth(dt)
        self._layout_orbit()
        self._update_vision_label()

        self._canvas.itemconfigure(self._state_ring, outline=self._ring_color_for_state())
        if self._state in {"listening", "speaking"}:
            ring_width = 3.0 + 1.1 * (0.5 + 0.5 * math.sin(self._idle_phase * 5.0))
        else:
            ring_width = 3.0
        self._canvas.itemconfigure(self._state_ring, width=ring_width)
        self._canvas.itemconfigure(self._aura_ring, dashoffset=int(self._idle_phase * 20))

        self._sync_layer_order()

    def pump(self) -> None:
        if not self._running or self._root is None:
            return

        self._frame()
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            self._running = False

    def stop(self) -> None:
        self._running = False
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None

    def running(self) -> bool:
        return self._running

    def set_state(self, state: str) -> None:
        self._state = state

    def set_status(self, status: str) -> None:
        self._status_text = status
        if self._status_var is not None:
            self._status_var.set(status)

    def add_message(self, speaker: str, text: str) -> None:
        if self._chat_text is None:
            return

        self._chat_text.configure(state="normal")
        self._chat_text.insert(tk.END, f"{speaker}: {text}\n")
        self._chat_lines += 1
        if self._chat_lines > 120:
            self._chat_text.delete("1.0", "3.0")
            self._chat_lines -= 2
        self._chat_text.see(tk.END)
        self._chat_text.configure(state="disabled")

    def set_face_target(
        self,
        x: float,
        y: float,
        found: bool,
        face_count: int = 0,
        scene_label: str = "unknown",
        scene_confidence: float = 0.0,
    ) -> None:
        self._vision_found = found
        self._vision_face_count = max(0, face_count)
        self._vision_scene_label = scene_label or "unknown"
        self._vision_scene_confidence = max(0.0, min(1.0, scene_confidence))

        if found:
            self._raw_face_x = max(-1.0, min(1.0, x))
            self._raw_face_y = max(-1.0, min(1.0, y))
        else:
            self._raw_face_x *= 0.9
            self._raw_face_y *= 0.9

    def set_emotion(self, name: str, intensity: float) -> None:
        self._emotion_name = (name or "neutral").strip().lower() or "neutral"
        self._emotion_intensity = max(0.0, min(1.0, intensity))

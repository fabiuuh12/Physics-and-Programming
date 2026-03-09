from __future__ import annotations

import math
import time
import tkinter as tk
from tkinter import ttk


class AliceFaceUI:
    def __init__(self, title: str = "Alice") -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("520x700")
        self.root.minsize(420, 620)
        self.root.configure(bg="#0c1117")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.running = True
        self.state = "idle"
        self.status_text = "Online"
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
        self._hand_found = False
        self._hand_count = 0
        self._avatar_offset_y = 24.0
        self._head_x = 0.0
        self._head_y = 0.0
        self._head_tilt = 0.0
        self._tracker = None
        self._face_found = False
        self._face_name = ""

        self._build_layout()
        self._draw_static_face()
        self._draw_dynamic_face()

    def attach_face_tracker(self, tracker: object) -> None:
        self._tracker = tracker
        self.focus_label.configure(text="Camera: active")

    def _on_close(self) -> None:
        self.running = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def close(self) -> None:
        self._on_close()

    def _build_layout(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Alice.TFrame", background="#0c1117")
        style.configure("AliceStatus.TLabel", background="#0c1117", foreground="#8db8ff")
        style.configure("AliceFocus.TLabel", background="#0c1117", foreground="#95d7ff")

        shell = ttk.Frame(self.root, style="Alice.TFrame", padding=12)
        shell.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            shell,
            width=460,
            height=430,
            bg="#0f1724",
            highlightthickness=0,
        )
        self.canvas.pack(fill="x")

        self.status_label = ttk.Label(
            shell,
            text=self.status_text,
            style="AliceStatus.TLabel",
            anchor="w",
            font=("Avenir Next", 12),
        )
        self.status_label.pack(fill="x", pady=(10, 2))

        self.focus_label = ttk.Label(
            shell,
            text="Camera: off",
            style="AliceFocus.TLabel",
            anchor="w",
            font=("Avenir Next", 11),
        )
        self.focus_label.pack(fill="x", pady=(0, 8))

        self.chat_box = tk.Text(
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
        self.chat_box.pack(fill="both", expand=True)
        self.chat_box.configure(state="disabled")

    def _draw_static_face(self) -> None:
        c = self.canvas
        oy = self._avatar_offset_y
        c.create_oval(18, -42, 442, 382, fill="#101a29", outline="#1b2f4a", width=1)
        self.aura_ring = c.create_oval(36, -24 + oy, 424, 364 + oy, fill="", outline="#274365", width=1, dash=(2, 6))
        c.create_oval(48, -8 + oy, 412, 352 + oy, fill="", outline="#1e3455", width=1, dash=(3, 5))
        self.state_ring = c.create_oval(55, -1 + oy, 405, 349 + oy, outline="#4f7fb8", width=3)
        self.orbit_dot = c.create_oval(228, -16 + oy, 236, -8 + oy, fill="#7fbfff", outline="")
        self.face_base = c.create_oval(92, 38 + oy, 368, 318 + oy, fill="#0e2034", outline="#3f88b7", width=2)
        self.face_shade = c.create_oval(108, 56 + oy, 352, 300 + oy, fill="", outline="#6eb6de", width=1, dash=(4, 6))

    def _draw_dynamic_face(self) -> None:
        c = self.canvas
        oy = self._avatar_offset_y
        self.left_eye_white = c.create_oval(146, 122 + oy, 208, 176 + oy, fill="#0f2539", outline="#74c6f0", width=2)
        self.right_eye_white = c.create_oval(252, 122 + oy, 314, 176 + oy, fill="#0f2539", outline="#74c6f0", width=2)
        self.left_iris = c.create_oval(166, 136 + oy, 190, 160 + oy, fill="#63dcff", outline="")
        self.right_iris = c.create_oval(272, 136 + oy, 296, 160 + oy, fill="#63dcff", outline="")
        self.left_pupil = c.create_oval(174, 143 + oy, 182, 151 + oy, fill="#05121c", outline="")
        self.right_pupil = c.create_oval(280, 143 + oy, 288, 151 + oy, fill="#05121c", outline="")
        self.left_highlight = c.create_oval(177, 145 + oy, 180, 148 + oy, fill="#d9f6ff", outline="")
        self.right_highlight = c.create_oval(283, 145 + oy, 286, 148 + oy, fill="#d9f6ff", outline="")
        self.left_upper_lid = c.create_line(146, 131 + oy, 177, 126 + oy, 208, 131 + oy, fill="#7fcff1", width=2, smooth=True)
        self.right_upper_lid = c.create_line(252, 131 + oy, 283, 126 + oy, 314, 131 + oy, fill="#7fcff1", width=2, smooth=True)
        self.left_lower_lid = c.create_line(146, 167 + oy, 177, 171 + oy, 208, 167 + oy, fill="#5ca2ce", width=2, smooth=True)
        self.right_lower_lid = c.create_line(252, 167 + oy, 283, 171 + oy, 314, 167 + oy, fill="#5ca2ce", width=2, smooth=True)

        self.nose_bridge = c.create_line(230, 146 + oy, 230, 196 + oy, fill="#3d7fa9", width=2)
        self.nose_tip = c.create_line(220, 198 + oy, 230, 202 + oy, 240, 198 + oy, fill="#3d7fa9", width=2, smooth=True)

        self.mouth_inner = c.create_oval(192, 248 + oy, 268, 266 + oy, fill="#123553", outline="#4eb9e8", width=1)
        self.mouth_teeth = c.create_rectangle(200, 250 + oy, 260, 254 + oy, fill="#b7ecff", outline="")
        self.mouth_upper = c.create_line(176, 254 + oy, 204, 250 + oy, 230, 248 + oy, 256, 250 + oy, 284, 254 + oy, fill="#6ad8ff", width=3, smooth=True)
        self.mouth_lower = c.create_line(176, 254 + oy, 204, 260 + oy, 230, 264 + oy, 256, 260 + oy, 284, 254 + oy, fill="#3ea0d2", width=3, smooth=True)
        self.lip_highlight = c.create_line(186, 252 + oy, 214, 249 + oy, 230, 248 + oy, 246, 249 + oy, 274, 252 + oy, fill="#d6f7ff", width=1, smooth=True)
        self._sync_layer_order()

    def _sync_layer_order(self) -> None:
        for item in (
            self.face_base,
            self.face_shade,
            self.nose_bridge,
            self.nose_tip,
            self.left_eye_white,
            self.right_eye_white,
            self.left_iris,
            self.right_iris,
            self.left_pupil,
            self.right_pupil,
            self.left_highlight,
            self.right_highlight,
            self.left_upper_lid,
            self.right_upper_lid,
            self.left_lower_lid,
            self.right_lower_lid,
            self.mouth_inner,
            self.mouth_teeth,
            self.mouth_lower,
            self.mouth_upper,
            self.lip_highlight,
        ):
            self.canvas.tag_raise(item)

    def _ring_color_for_state(self) -> str:
        if self.state == "listening":
            return "#6eb8ff"
        if self.state == "thinking":
            return "#f0cd73"
        if self.state == "speaking":
            return "#7be4b7"
        if self.state == "error":
            return "#ff9393"
        if self.state == "offline":
            return "#66778f"
        return "#4f7fb8"

    def _update_face_tracking(self) -> None:
        if self._tracker is None:
            self._target_gaze_x = 0.0
            self._target_gaze_y = 0.0
            self._face_found = False
            self._hand_found = False
            self._hand_count = 0
            return

        try:
            obs = self._tracker.get_latest()
        except Exception:
            self._target_gaze_x = 0.0
            self._target_gaze_y = 0.0
            self._face_found = False
            self._hand_found = False
            self._hand_count = 0
            return

        self._hand_found = bool(getattr(obs, "hand_found", False))
        self._hand_count = int(getattr(obs, "hand_count", 0) or 0)

        if obs.found:
            tx = obs.x * 0.90
            ty = obs.y * 0.78

            dx = tx - self._smoothed_track_x
            dy = ty - self._smoothed_track_y
            motion = math.hypot(dx, dy)
            alpha = 0.54 if motion > 0.35 else 0.30
            self._smoothed_track_x += dx * alpha
            self._smoothed_track_y += dy * alpha

            smooth_x = self._smoothed_track_x
            smooth_y = self._smoothed_track_y
            # Non-linear boost near edges keeps gaze expressive.
            smooth_x += 0.14 * smooth_x * abs(smooth_x)
            smooth_y += 0.10 * smooth_y * abs(smooth_y)

            self._target_gaze_x = max(-1.0, min(1.0, smooth_x))
            self._target_gaze_y = max(-1.0, min(1.0, smooth_y))
            self._face_found = True
            self._track_blend = min(1.0, self._track_blend + 0.14)
            self._face_name = obs.owner_name or "You"
            suffix = f" | hands: {self._hand_count}" if self._hand_found else ""
            self.focus_label.configure(text=f"Camera: tracking {self._face_name}{suffix}")
        elif self._hand_found:
            tx = getattr(obs, "hand_x", 0.0) * 0.65
            ty = getattr(obs, "hand_y", 0.0) * 0.58
            self._smoothed_track_x += (tx - self._smoothed_track_x) * 0.32
            self._smoothed_track_y += (ty - self._smoothed_track_y) * 0.32
            self._target_gaze_x = max(-1.0, min(1.0, self._smoothed_track_x))
            self._target_gaze_y = max(-1.0, min(1.0, self._smoothed_track_y))
            self._face_found = False
            self._track_blend = min(1.0, self._track_blend + 0.08)
            noun = "hand" if self._hand_count == 1 else "hands"
            self.focus_label.configure(text=f"Camera: tracking {self._hand_count} {noun}")
        else:
            self._smoothed_track_x *= 0.90
            self._smoothed_track_y *= 0.90
            self._target_gaze_x = self._smoothed_track_x
            self._target_gaze_y = self._smoothed_track_y
            self._face_found = False
            self._track_blend = max(0.0, self._track_blend - 0.08)
            self.focus_label.configure(text="Camera: searching...")

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
            t = min((now - self._blink_started_at) / 0.1, 1.0)
            self._blink_value = t
            if t >= 1.0:
                self._blink_stage = 0
                self._next_blink_at = now + 2.3
        else:
            self._blink_value = 1.0

    def _pose_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            x + self._head_x,
            y + self._avatar_offset_y + self._head_y + (x - 230.0) * self._head_tilt,
        )

    def _layout_head_shell(self) -> None:
        breath = 1.0 + 0.006 * math.sin(self._idle_phase * 1.3)
        head_cx = 230.0 + self._head_x
        head_cy = 177.0 + self._avatar_offset_y + self._head_y
        face_half_w = 138.0 * breath
        face_half_h = 140.0 * breath
        self.canvas.coords(
            self.face_base,
            head_cx - face_half_w,
            head_cy - face_half_h,
            head_cx + face_half_w,
            head_cy + face_half_h,
        )
        self.canvas.coords(
            self.face_shade,
            head_cx - 124.0 * breath,
            head_cy - 122.0 * breath,
            head_cx + 124.0 * breath,
            head_cy + 122.0 * breath,
        )

        self.canvas.coords(
            self.nose_bridge,
            *self._pose_point(230.0, 144.0),
            *self._pose_point(230.0, 198.0),
        )
        self.canvas.coords(
            self.nose_tip,
            *self._pose_point(220.0, 198.0),
            *self._pose_point(230.0, 202.0),
            *self._pose_point(240.0, 198.0),
        )

    def _layout_eyes(self, dt: float) -> None:
        err_x = self._target_gaze_x - self._gaze_x
        err_y = self._target_gaze_y - self._gaze_y
        err_mag = min(1.0, math.hypot(err_x, err_y))

        spring = (11.5 if self._face_found else 7.2) + err_mag * (5.2 if self._face_found else 2.0)
        damping = 6.2 if self._face_found else 4.9
        self._gaze_vx += (err_x * spring - self._gaze_vx * damping) * dt
        self._gaze_vy += (err_y * spring - self._gaze_vy * damping) * dt
        if self._face_found and err_mag > 0.34:
            boost = min(0.18, err_mag * 0.22)
            self._gaze_vx += err_x * boost
            self._gaze_vy += err_y * boost
        self._gaze_x += self._gaze_vx * dt
        self._gaze_y += self._gaze_vy * dt
        self._gaze_x = max(-1.2, min(1.2, self._gaze_x))
        self._gaze_y = max(-1.2, min(1.2, self._gaze_y))

        blink_h = max(2.0, 52.0 * self._blink_value)
        left_mid_x, left_mid_y = self._pose_point(177.0, 149.0 + self._face_offset_y)
        right_mid_x, right_mid_y = self._pose_point(283.0, 149.0 + self._face_offset_y)
        half_w = 31.0
        left_top = left_mid_y - blink_h / 2.0
        left_bottom = left_mid_y + blink_h / 2.0
        right_top = right_mid_y - blink_h / 2.0
        right_bottom = right_mid_y + blink_h / 2.0

        self.canvas.coords(
            self.left_eye_white,
            left_mid_x - half_w,
            left_top,
            left_mid_x + half_w,
            left_bottom,
        )
        self.canvas.coords(
            self.right_eye_white,
            right_mid_x - half_w,
            right_top,
            right_mid_x + half_w,
            right_bottom,
        )
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
        self.canvas.coords(
            self.left_upper_lid,
            left_mid_x - half_w,
            left_upper_lid_y + 0.8,
            left_mid_x,
            left_upper_peak,
            left_mid_x + half_w,
            left_upper_lid_y + 0.8,
        )
        self.canvas.coords(
            self.right_upper_lid,
            right_mid_x - half_w,
            right_upper_lid_y + 0.8,
            right_mid_x,
            right_upper_peak,
            right_mid_x + half_w,
            right_upper_lid_y + 0.8,
        )
        self.canvas.coords(
            self.left_lower_lid,
            left_mid_x - half_w,
            left_lower_lid_y - 0.8,
            left_mid_x,
            left_lower_valley,
            left_mid_x + half_w,
            left_lower_lid_y - 0.8,
        )
        self.canvas.coords(
            self.right_lower_lid,
            right_mid_x - half_w,
            right_lower_lid_y - 0.8,
            right_mid_x,
            right_lower_valley,
            right_mid_x + half_w,
            right_lower_lid_y - 0.8,
        )

        if blink_h < 8:
            self.canvas.itemconfigure(self.left_iris, state="hidden")
            self.canvas.itemconfigure(self.right_iris, state="hidden")
            self.canvas.itemconfigure(self.left_pupil, state="hidden")
            self.canvas.itemconfigure(self.right_pupil, state="hidden")
            self.canvas.itemconfigure(self.left_highlight, state="hidden")
            self.canvas.itemconfigure(self.right_highlight, state="hidden")
            self.canvas.itemconfigure(self.left_lower_lid, state="hidden")
            self.canvas.itemconfigure(self.right_lower_lid, state="hidden")
            return

        self.canvas.itemconfigure(self.left_iris, state="normal")
        self.canvas.itemconfigure(self.right_iris, state="normal")
        self.canvas.itemconfigure(self.left_pupil, state="normal")
        self.canvas.itemconfigure(self.right_pupil, state="normal")
        self.canvas.itemconfigure(self.left_highlight, state="normal")
        self.canvas.itemconfigure(self.right_highlight, state="normal")
        self.canvas.itemconfigure(self.left_lower_lid, state="normal")
        self.canvas.itemconfigure(self.right_lower_lid, state="normal")

        if self._face_found:
            micro_scale = 0.70 + 0.30 * (1.0 - self._track_blend)
            micro_x = micro_scale * (
                0.24 * math.sin(self._idle_phase * 10.4) + 0.16 * math.cos(self._idle_phase * 5.6)
            )
            micro_y = micro_scale * (0.18 * math.cos(self._idle_phase * 8.7))
        elif self._hand_found:
            micro_x = 0.34 * math.sin(self._idle_phase * 6.3)
            micro_y = 0.26 * math.cos(self._idle_phase * 5.4)
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
        if self.state == "thinking":
            iris_r -= 0.8
        pupil_r = 3.8 + 0.8 * eye_energy
        self.canvas.coords(self.left_iris, li_x - iris_r, li_y - iris_r, li_x + iris_r, li_y + iris_r)
        self.canvas.coords(self.right_iris, ri_x - iris_r, ri_y - iris_r, ri_x + iris_r, ri_y + iris_r)
        self.canvas.coords(self.left_pupil, li_x - pupil_r, li_y - pupil_r, li_x + pupil_r, li_y + pupil_r)
        self.canvas.coords(self.right_pupil, ri_x - pupil_r, ri_y - pupil_r, ri_x + pupil_r, ri_y + pupil_r)
        self.canvas.coords(self.left_highlight, li_x - 1, li_y - 1, li_x + 2, li_y + 2)
        self.canvas.coords(self.right_highlight, ri_x - 1, ri_y - 1, ri_x + 2, ri_y + 2)

    def _layout_brows(self) -> None:
        return

    def _layout_mouth(self, dt: float) -> None:
        if self.state == "speaking":
            self._speak_phase += dt * 10.0
            target_open = 7.0 + 13.0 * (0.5 + 0.5 * math.sin(self._speak_phase))
            curve = 2.6
        elif self.state == "listening":
            target_open = 4.6
            curve = 1.1
        elif self.state == "thinking":
            target_open = 3.8
            curve = 0.6
        elif self.state == "error":
            target_open = 2.2
            curve = -2.8
        else:
            target_open = 2.8
            curve = 1.8 if self._face_found or self._hand_found else 1.1

        smoothing = min(1.0, dt * 12.0)
        self._mouth_open += (target_open - self._mouth_open) * smoothing
        open_val = max(1.8, self._mouth_open)

        mid_x = 230.0
        left_x = 176.0
        right_x = 284.0
        y = 258.0 + self._face_offset_y + 0.35 * self._head_y

        ux1, uy1 = self._pose_point(left_x, y)
        ux2, uy2 = self._pose_point(204.0, y - 4.0 - curve)
        ux3, uy3 = self._pose_point(mid_x, y - 5.0 - curve * 0.45)
        ux4, uy4 = self._pose_point(256.0, y - 4.0 - curve)
        ux5, uy5 = self._pose_point(right_x, y)
        self.canvas.coords(
            self.mouth_upper,
            ux1,
            uy1,
            ux2,
            uy2,
            ux3,
            uy3,
            ux4,
            uy4,
            ux5,
            uy5,
        )
        lx1, ly1 = self._pose_point(left_x, y)
        lx2, ly2 = self._pose_point(204.0, y + open_val * 0.62)
        lx3, ly3 = self._pose_point(mid_x, y + open_val)
        lx4, ly4 = self._pose_point(256.0, y + open_val * 0.62)
        lx5, ly5 = self._pose_point(right_x, y)
        self.canvas.coords(
            self.mouth_lower,
            lx1,
            ly1,
            lx2,
            ly2,
            lx3,
            ly3,
            lx4,
            ly4,
            lx5,
            ly5,
        )
        hx1, hy1 = self._pose_point(186.0, y - 1.5)
        hx2, hy2 = self._pose_point(214.0, y - 4.5 - curve * 0.55)
        hx3, hy3 = self._pose_point(mid_x, y - 4.2 - curve * 0.2)
        hx4, hy4 = self._pose_point(246.0, y - 4.5 - curve * 0.55)
        hx5, hy5 = self._pose_point(274.0, y - 1.5)
        self.canvas.coords(
            self.lip_highlight,
            hx1,
            hy1,
            hx2,
            hy2,
            hx3,
            hy3,
            hx4,
            hy4,
            hx5,
            hy5,
        )

        inner_left = 194.0
        inner_right = 266.0
        inner_top = y - 2.0
        inner_bottom = y + max(3.0, open_val * 0.95)
        if open_val < 3.1:
            self.canvas.itemconfigure(self.mouth_inner, state="hidden")
            self.canvas.itemconfigure(self.mouth_teeth, state="hidden")
        else:
            self.canvas.itemconfigure(self.mouth_inner, state="normal")
            self.canvas.coords(
                self.mouth_inner,
                *self._pose_point(inner_left, inner_top),
                *self._pose_point(inner_right, inner_bottom),
            )
            if 3.1 <= open_val < 7.8:
                teeth_top = inner_top + 2.0
                teeth_bottom = teeth_top + min(3.0, open_val * 0.18)
                self.canvas.coords(
                    self.mouth_teeth,
                    *self._pose_point(inner_left + 6.0, teeth_top),
                    *self._pose_point(inner_right - 6.0, teeth_bottom),
                )
                self.canvas.itemconfigure(self.mouth_teeth, state="normal")
            else:
                self.canvas.itemconfigure(self.mouth_teeth, state="hidden")

    def _frame(self) -> None:
        now = time.monotonic()
        dt = now - self._last_frame_time
        dt = min(max(dt, 1.0 / 240.0), 1.0 / 24.0)
        self._last_frame_time = now
        self._idle_phase += dt * (2.2 if self.state == "speaking" else 1.5)
        self._face_offset_y = math.sin(self._idle_phase) * 0.8
        self._update_face_tracking()
        self._animate_blink(now)
        tracked = self._track_blend if (self._face_found or self._hand_found) else 0.0
        target_head_x = (
            self._gaze_x * (7.2 if self._face_found else 4.6) * tracked
            + 0.24 * math.sin(self._idle_phase * 0.9)
        )
        target_head_y = (
            self._gaze_y * (2.8 if self._face_found else 1.7) * tracked
            + 0.35 * math.sin(self._idle_phase * 1.1)
        )
        target_head_tilt = -0.058 * self._gaze_x * tracked + 0.010 * math.sin(self._idle_phase * 1.4)
        if self.state == "speaking":
            target_head_y += 1.2 * math.sin(self._idle_phase * 3.5)
            target_head_tilt += 0.010 * math.sin(self._idle_phase * 3.1)
        elif self.state == "thinking":
            target_head_tilt += 0.013 * math.sin(self._idle_phase * 2.6)

        blend = min(1.0, dt * 9.0)
        self._head_x += (target_head_x - self._head_x) * blend
        self._head_y += (target_head_y - self._head_y) * blend
        self._head_tilt += (target_head_tilt - self._head_tilt) * min(1.0, dt * 10.0)

        self._layout_head_shell()
        self._layout_eyes(dt)
        self._layout_mouth(dt)

        orbit_x = 230.0 + 186.0 * math.cos(self._idle_phase * 0.42)
        orbit_y = 170.0 + self._avatar_offset_y + 186.0 * math.sin(self._idle_phase * 0.42)
        self.canvas.coords(self.orbit_dot, orbit_x - 4.0, orbit_y - 4.0, orbit_x + 4.0, orbit_y + 4.0)
        self.canvas.itemconfigure(self.aura_ring, dashoffset=int(self._idle_phase * 20))
        self.canvas.itemconfigure(self.state_ring, outline=self._ring_color_for_state())
        if self.state in {"listening", "speaking"}:
            ring_width = 3.0 + 1.1 * (0.5 + 0.5 * math.sin(self._idle_phase * 5.0))
        else:
            ring_width = 3.0
        self.canvas.itemconfigure(self.state_ring, width=ring_width)
        self._sync_layer_order()

    def pump(self) -> None:
        if not self.running:
            return
        try:
            self._frame()
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.running = False

    def set_state(self, state: str) -> None:
        self.state = state

    def set_status(self, text: str) -> None:
        self.status_text = text
        if not self.running:
            return
        try:
            self.status_label.configure(text=text)
        except tk.TclError:
            self.running = False

    def add_message(self, speaker: str, text: str) -> None:
        if not self.running:
            return
        try:
            self.chat_box.configure(state="normal")
            self.chat_box.insert("end", f"{speaker}: {text}\n")
            self.chat_box.see("end")
            self.chat_box.configure(state="disabled")
        except tk.TclError:
            self.running = False

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
        c.create_oval(18, -42, 442, 382, fill="#101a29", outline="#1b2f4a", width=1)
        self.aura_ring = c.create_oval(36, -24, 424, 364, fill="", outline="#274365", width=1, dash=(2, 6))
        c.create_oval(48, -8, 412, 352, fill="", outline="#1e3455", width=1, dash=(3, 5))
        self.state_ring = c.create_oval(55, -1, 405, 349, outline="#4f7fb8", width=3)
        self.orbit_dot = c.create_oval(228, -16, 236, -8, fill="#7fbfff", outline="")
        self.face_base = c.create_oval(95, 36, 365, 318, fill="#f0dcc8", outline="#cfb5a0", width=2)
        self.face_shade = c.create_oval(108, 62, 352, 325, fill="", outline="#d8bca7", width=1)

        c.create_oval(112, 150, 136, 188, fill="#dcb8a2", outline="")
        c.create_oval(324, 150, 348, 188, fill="#dcb8a2", outline="")
        self.left_cheek = c.create_oval(132, 188, 188, 235, fill="#efb5ab", outline="")
        self.right_cheek = c.create_oval(272, 188, 328, 235, fill="#efb5ab", outline="")

    def _draw_dynamic_face(self) -> None:
        c = self.canvas
        self.left_eye_white = c.create_oval(146, 122, 208, 176, fill="#f8fcff", outline="#b8cadc", width=2)
        self.right_eye_white = c.create_oval(252, 122, 314, 176, fill="#f8fcff", outline="#b8cadc", width=2)
        self.left_iris = c.create_oval(166, 136, 190, 160, fill="#3a6ea5", outline="")
        self.right_iris = c.create_oval(272, 136, 296, 160, fill="#3a6ea5", outline="")
        self.left_pupil = c.create_oval(174, 143, 182, 151, fill="#0f1a27", outline="")
        self.right_pupil = c.create_oval(280, 143, 288, 151, fill="#0f1a27", outline="")
        self.left_highlight = c.create_oval(177, 145, 180, 148, fill="#edf5ff", outline="")
        self.right_highlight = c.create_oval(283, 145, 286, 148, fill="#edf5ff", outline="")
        self.left_upper_lid = c.create_line(146, 131, 177, 126, 208, 131, fill="#cf9a88", width=2, smooth=True)
        self.right_upper_lid = c.create_line(252, 131, 283, 126, 314, 131, fill="#cf9a88", width=2, smooth=True)
        self.left_lower_lid = c.create_line(146, 167, 177, 171, 208, 167, fill="#c0907f", width=2, smooth=True)
        self.right_lower_lid = c.create_line(252, 167, 283, 171, 314, 167, fill="#c0907f", width=2, smooth=True)
        self.left_brow = c.create_line(142, 113, 176, 100, 212, 112, fill="#644a3f", width=3, smooth=True)
        self.right_brow = c.create_line(248, 112, 284, 100, 318, 113, fill="#644a3f", width=3, smooth=True)

        self.nose_bridge = c.create_line(230, 148, 226, 182, 230, 194, fill="#c9a995", width=2, smooth=True)
        self.nose_tip = c.create_line(220, 198, 230, 202, 240, 198, fill="#c9a995", width=2, smooth=True)

        self.mouth_inner = c.create_oval(192, 248, 268, 266, fill="#7f4146", outline="")
        self.mouth_teeth = c.create_rectangle(200, 250, 260, 256, fill="#f8efe7", outline="")
        self.mouth_upper = c.create_line(176, 254, 204, 245, 230, 243, 256, 245, 284, 254, fill="#8a4a4c", width=3, smooth=True)
        self.mouth_lower = c.create_line(176, 254, 204, 262, 230, 266, 256, 262, 284, 254, fill="#8a4a4c", width=3, smooth=True)
        self.lip_highlight = c.create_line(186, 252, 214, 247, 230, 246, 246, 247, 274, 252, fill="#d97b81", width=1, smooth=True)
        self._sync_layer_order()

    def _sync_layer_order(self) -> None:
        for item in (
            self.face_base,
            self.face_shade,
            self.left_cheek,
            self.right_cheek,
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
            self.left_brow,
            self.right_brow,
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
            return

        try:
            obs = self._tracker.get_latest()
        except Exception:
            self._target_gaze_x = 0.0
            self._target_gaze_y = 0.0
            self._face_found = False
            return

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
            self.focus_label.configure(text=f"Camera: tracking {self._face_name}")
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
        left_mid_x, right_mid_x = 177.0, 283.0
        mid_y = 149.0 + self._face_offset_y
        half_w = 31.0
        top = mid_y - blink_h / 2.0
        bottom = mid_y + blink_h / 2.0

        self.canvas.coords(self.left_eye_white, left_mid_x - half_w, top, left_mid_x + half_w, bottom)
        self.canvas.coords(self.right_eye_white, right_mid_x - half_w, top, right_mid_x + half_w, bottom)
        upper_lid_y = top + max(2.0, 0.18 * blink_h)
        lower_lid_y = bottom - max(2.0, 0.18 * blink_h)
        if lower_lid_y < upper_lid_y + 2.0:
            lower_lid_y = upper_lid_y + 2.0
        upper_peak = upper_lid_y - max(1.3, 0.12 * blink_h)
        lower_valley = lower_lid_y + max(1.3, 0.10 * blink_h)
        self.canvas.coords(
            self.left_upper_lid,
            left_mid_x - half_w,
            upper_lid_y + 0.8,
            left_mid_x,
            upper_peak,
            left_mid_x + half_w,
            upper_lid_y + 0.8,
        )
        self.canvas.coords(
            self.right_upper_lid,
            right_mid_x - half_w,
            upper_lid_y + 0.8,
            right_mid_x,
            upper_peak,
            right_mid_x + half_w,
            upper_lid_y + 0.8,
        )
        self.canvas.coords(
            self.left_lower_lid,
            left_mid_x - half_w,
            lower_lid_y - 0.8,
            left_mid_x,
            lower_valley,
            left_mid_x + half_w,
            lower_lid_y - 0.8,
        )
        self.canvas.coords(
            self.right_lower_lid,
            right_mid_x - half_w,
            lower_lid_y - 0.8,
            right_mid_x,
            lower_valley,
            right_mid_x + half_w,
            lower_lid_y - 0.8,
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
        else:
            micro_x = 0.8 * math.sin(self._idle_phase * 2.3)
            micro_y = 0.5 * math.cos(self._idle_phase * 1.9)
        px = self._gaze_x * 10.8 + micro_x
        py = self._gaze_y * 8.8 + micro_y
        convergence = 0.7 * self._track_blend
        li_x = left_mid_x + px + convergence
        li_y = mid_y + py
        ri_x = right_mid_x + px - convergence
        ri_y = mid_y + py

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
        dy = self._face_offset_y * 0.8
        if self.state == "thinking":
            left = (142, 109 + dy, 176, 103 + dy, 212, 113 + dy)
            right = (248, 113 + dy, 284, 103 + dy, 318, 109 + dy)
        elif self.state == "speaking":
            left = (142, 115 + dy, 176, 98 + dy, 212, 111 + dy)
            right = (248, 111 + dy, 284, 98 + dy, 318, 115 + dy)
        elif self.state == "error":
            left = (142, 116 + dy, 176, 121 + dy, 212, 112 + dy)
            right = (248, 112 + dy, 284, 121 + dy, 318, 116 + dy)
        else:
            left = (142, 113 + dy, 176, 100 + dy, 212, 112 + dy)
            right = (248, 112 + dy, 284, 100 + dy, 318, 113 + dy)
        self.canvas.coords(self.left_brow, *left)
        self.canvas.coords(self.right_brow, *right)

    def _layout_mouth(self, dt: float) -> None:
        if self.state == "speaking":
            self._speak_phase += dt * 10.0
            target_open = 8.0 + 14.0 * (0.5 + 0.5 * math.sin(self._speak_phase))
            smile = 4.0
        elif self.state == "listening":
            target_open = 4.2
            smile = 1.0
        elif self.state == "thinking":
            target_open = 3.8
            smile = 0.5
        elif self.state == "error":
            target_open = 2.2
            smile = -3.5
        else:
            target_open = 2.8
            smile = 2.0

        smoothing = min(1.0, dt * 12.0)
        self._mouth_open += (target_open - self._mouth_open) * smoothing
        open_val = max(1.8, self._mouth_open)

        mid_x = 230.0
        left_x = 176.0
        right_x = 284.0
        y = 254.0 + self._face_offset_y

        self.canvas.coords(
            self.mouth_upper,
            left_x,
            y,
            204,
            y - 8 - smile,
            mid_x,
            y - 9 - smile * 0.35,
            256,
            y - 8 - smile,
            right_x,
            y,
        )
        self.canvas.coords(
            self.mouth_lower,
            left_x,
            y,
            204,
            y + open_val * 0.52,
            mid_x,
            y + open_val,
            256,
            y + open_val * 0.52,
            right_x,
            y,
        )
        self.canvas.coords(
            self.lip_highlight,
            186,
            y - 2,
            214,
            y - 7 - smile * 0.6,
            mid_x,
            y - 7 - smile * 0.3,
            246,
            y - 7 - smile * 0.6,
            274,
            y - 2,
        )

        inner_left = 194.0 - smile * 0.3
        inner_right = 266.0 + smile * 0.3
        inner_top = y - 2.0
        inner_bottom = y + max(3.0, open_val * 0.95)
        if open_val < 3.1:
            self.canvas.itemconfigure(self.mouth_inner, state="hidden")
            self.canvas.itemconfigure(self.mouth_teeth, state="hidden")
        else:
            self.canvas.itemconfigure(self.mouth_inner, state="normal")
            self.canvas.coords(self.mouth_inner, inner_left, inner_top, inner_right, inner_bottom)
            if 3.1 <= open_val < 7.8:
                teeth_top = inner_top + 2.0
                teeth_bottom = teeth_top + min(6.0, open_val * 0.40)
                self.canvas.coords(
                    self.mouth_teeth,
                    inner_left + 6.0,
                    teeth_top,
                    inner_right - 6.0,
                    teeth_bottom,
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
        self._layout_eyes(dt)
        self._layout_brows()
        self._layout_mouth(dt)

        orbit_x = 230.0 + 186.0 * math.cos(self._idle_phase * 0.42)
        orbit_y = 170.0 + 186.0 * math.sin(self._idle_phase * 0.42)
        self.canvas.coords(self.orbit_dot, orbit_x - 4.0, orbit_y - 4.0, orbit_x + 4.0, orbit_y + 4.0)
        self.canvas.itemconfigure(self.aura_ring, dashoffset=int(self._idle_phase * 20))

        blush_fill = "#efb5ab" if self.state in {"speaking", "listening"} else "#e8b2a8"
        self.canvas.itemconfigure(self.left_cheek, fill=blush_fill)
        self.canvas.itemconfigure(self.right_cheek, fill=blush_fill)
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

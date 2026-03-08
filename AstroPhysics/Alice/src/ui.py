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
        self._last_frame_time = time.monotonic()
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._target_gaze_x = 0.0
        self._target_gaze_y = 0.0
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
        c.create_oval(48, -8, 412, 352, fill="", outline="#1e3455", width=1, dash=(3, 5))
        self.state_ring = c.create_oval(55, -1, 405, 349, outline="#4f7fb8", width=3)
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
        self.left_lid = c.create_line(146, 149, 208, 149, fill="#d49f8d", width=2, smooth=True)
        self.right_lid = c.create_line(252, 149, 314, 149, fill="#d49f8d", width=2, smooth=True)
        self.left_brow = c.create_line(142, 113, 176, 100, 212, 112, fill="#644a3f", width=3, smooth=True)
        self.right_brow = c.create_line(248, 112, 284, 100, 318, 113, fill="#644a3f", width=3, smooth=True)

        self.nose_bridge = c.create_line(230, 148, 226, 182, 230, 194, fill="#c9a995", width=2, smooth=True)
        self.nose_tip = c.create_line(220, 198, 230, 202, 240, 198, fill="#c9a995", width=2, smooth=True)

        self.mouth_shadow = c.create_oval(186, 244, 274, 272, fill="#8d4f50", outline="")
        self.mouth_upper = c.create_line(176, 254, 204, 246, 230, 244, 256, 246, 284, 254, fill="#8a4a4c", width=3, smooth=True)
        self.mouth_lower = c.create_line(176, 254, 204, 262, 230, 266, 256, 262, 284, 254, fill="#8a4a4c", width=3, smooth=True)
        self.lip_highlight = c.create_line(186, 252, 214, 247, 230, 246, 246, 247, 274, 252, fill="#d97b81", width=1, smooth=True)

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
            self._target_gaze_x = max(-1.0, min(1.0, obs.x * 0.75))
            self._target_gaze_y = max(-1.0, min(1.0, obs.y * 0.65))
            self._face_found = True
            self._face_name = obs.owner_name or "You"
            self.focus_label.configure(text=f"Camera: tracking {self._face_name}")
        else:
            self._target_gaze_x *= 0.9
            self._target_gaze_y *= 0.9
            self._face_found = False
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

    def _layout_eyes(self) -> None:
        self._gaze_x += (self._target_gaze_x - self._gaze_x) * 0.15
        self._gaze_y += (self._target_gaze_y - self._gaze_y) * 0.15

        blink_h = max(2.0, 54.0 * self._blink_value)
        left_mid_x, right_mid_x = 177.0, 283.0
        mid_y = 149.0
        half_w = 31.0
        top = mid_y - blink_h / 2.0
        bottom = mid_y + blink_h / 2.0

        self.canvas.coords(self.left_eye_white, left_mid_x - half_w, top, left_mid_x + half_w, bottom)
        self.canvas.coords(self.right_eye_white, right_mid_x - half_w, top, right_mid_x + half_w, bottom)
        self.canvas.coords(self.left_lid, left_mid_x - half_w, mid_y, left_mid_x, mid_y - 2, left_mid_x + half_w, mid_y)
        self.canvas.coords(self.right_lid, right_mid_x - half_w, mid_y, right_mid_x, mid_y - 2, right_mid_x + half_w, mid_y)

        if blink_h < 5:
            self.canvas.itemconfigure(self.left_iris, state="hidden")
            self.canvas.itemconfigure(self.right_iris, state="hidden")
            self.canvas.itemconfigure(self.left_pupil, state="hidden")
            self.canvas.itemconfigure(self.right_pupil, state="hidden")
            self.canvas.itemconfigure(self.left_highlight, state="hidden")
            self.canvas.itemconfigure(self.right_highlight, state="hidden")
            return

        self.canvas.itemconfigure(self.left_iris, state="normal")
        self.canvas.itemconfigure(self.right_iris, state="normal")
        self.canvas.itemconfigure(self.left_pupil, state="normal")
        self.canvas.itemconfigure(self.right_pupil, state="normal")
        self.canvas.itemconfigure(self.left_highlight, state="normal")
        self.canvas.itemconfigure(self.right_highlight, state="normal")

        px = self._gaze_x * 10.0
        py = self._gaze_y * 8.0
        li_x = left_mid_x + px
        li_y = mid_y + py
        ri_x = right_mid_x + px
        ri_y = mid_y + py

        self.canvas.coords(self.left_iris, li_x - 12, li_y - 12, li_x + 12, li_y + 12)
        self.canvas.coords(self.right_iris, ri_x - 12, ri_y - 12, ri_x + 12, ri_y + 12)
        self.canvas.coords(self.left_pupil, li_x - 4, li_y - 4, li_x + 4, li_y + 4)
        self.canvas.coords(self.right_pupil, ri_x - 4, ri_y - 4, ri_x + 4, ri_y + 4)
        self.canvas.coords(self.left_highlight, li_x - 1, li_y - 1, li_x + 2, li_y + 2)
        self.canvas.coords(self.right_highlight, ri_x - 1, ri_y - 1, ri_x + 2, ri_y + 2)

    def _layout_brows(self) -> None:
        if self.state == "thinking":
            left = (142, 110, 176, 104, 212, 114)
            right = (248, 114, 284, 104, 318, 110)
        elif self.state == "speaking":
            left = (142, 114, 176, 99, 212, 112)
            right = (248, 112, 284, 99, 318, 114)
        elif self.state == "error":
            left = (142, 116, 176, 120, 212, 112)
            right = (248, 112, 284, 120, 318, 116)
        else:
            left = (142, 113, 176, 100, 212, 112)
            right = (248, 112, 284, 100, 318, 113)
        self.canvas.coords(self.left_brow, *left)
        self.canvas.coords(self.right_brow, *right)

    def _layout_mouth(self, dt: float) -> None:
        if self.state == "speaking":
            self._speak_phase += dt * 10.0
            open_val = 10.0 + 18.0 * (0.5 + 0.5 * math.sin(self._speak_phase))
            smile = 4.0
        elif self.state == "thinking":
            open_val = 5.0
            smile = 0.5
        elif self.state == "error":
            open_val = 3.0
            smile = -3.5
        else:
            open_val = 6.0
            smile = 2.0

        mid_x = 230.0
        left_x = 176.0
        right_x = 284.0
        y = 254.0

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
        self.canvas.coords(self.mouth_shadow, 186, y - 4, 274, y + open_val + 1)

    def _frame(self) -> None:
        now = time.monotonic()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        self._update_face_tracking()
        self._animate_blink(now)
        self._layout_eyes()
        self._layout_brows()
        self._layout_mouth(dt)

        blush_fill = "#efb5ab" if self.state in {"speaking", "listening"} else "#e8b2a8"
        self.canvas.itemconfigure(self.left_cheek, fill=blush_fill)
        self.canvas.itemconfigure(self.right_cheek, fill=blush_fill)
        self.canvas.itemconfigure(self.state_ring, outline=self._ring_color_for_state())

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

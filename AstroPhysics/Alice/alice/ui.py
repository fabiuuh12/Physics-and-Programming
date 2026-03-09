from __future__ import annotations

import math
import time
from typing import Optional

try:
    import tkinter as tk
except Exception:  # pragma: no cover - platform-specific import
    tk = None


class AliceUI:
    def __init__(self) -> None:
        self._running = False
        self._state = "idle"
        self._status_text = "Offline"
        self._start_ts = time.monotonic()
        self._face_x = 0.0
        self._face_y = 0.0

        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._status_var: Optional[tk.StringVar] = None
        self._chat_text: Optional[tk.Text] = None

        self._left_pupil = None
        self._right_pupil = None
        self._mouth = None
        self._chat_lines = 0

    def start(self) -> bool:
        if tk is None:
            self._running = False
            return False

        try:
            root = tk.Tk()
        except Exception:
            self._running = False
            return False

        root.title("Alice")
        root.geometry("540x700")
        root.configure(bg="#111622")
        root.protocol("WM_DELETE_WINDOW", self.stop)

        canvas = tk.Canvas(root, width=500, height=420, bg="#111622", highlightthickness=0)
        canvas.pack(pady=(14, 6))

        status_var = tk.StringVar(value=self._status_text)
        status = tk.Label(
            root,
            textvariable=status_var,
            bg="#111622",
            fg="#dce5ff",
            font=("Helvetica", 13, "bold"),
        )
        status.pack(pady=(0, 8))

        chat = tk.Text(
            root,
            height=12,
            width=64,
            bg="#0b1020",
            fg="#e8ecff",
            insertbackground="#e8ecff",
            relief=tk.FLAT,
            padx=12,
            pady=10,
            wrap=tk.WORD,
            font=("Menlo", 11),
        )
        chat.pack(padx=14, pady=(0, 10), fill=tk.BOTH, expand=True)
        chat.configure(state=tk.DISABLED)

        self._root = root
        self._canvas = canvas
        self._status_var = status_var
        self._chat_text = chat

        self._draw_face()
        self._running = True
        return True

    def _draw_face(self) -> None:
        if self._canvas is None:
            return

        c = self._canvas
        c.delete("all")

        c.create_oval(60, 20, 440, 400, fill="#f9dcb8", outline="#e0bc90", width=4)
        c.create_oval(140, 125, 230, 220, fill="#ffffff", outline="#d0d0d0", width=2)
        c.create_oval(270, 125, 360, 220, fill="#ffffff", outline="#d0d0d0", width=2)

        self._left_pupil = c.create_oval(175, 160, 195, 180, fill="#202640", outline="#202640")
        self._right_pupil = c.create_oval(305, 160, 325, 180, fill="#202640", outline="#202640")

        c.create_oval(235, 195, 265, 245, fill="#e2bb92", outline="#d29f72", width=2)
        self._mouth = c.create_line(190, 300, 310, 300, fill="#7b2f3d", width=8, capstyle=tk.ROUND)
        c.create_text(250, 52, text="ALICE", fill="#dce5ff", font=("Helvetica", 18, "bold"))

    def _animate_face(self) -> None:
        if self._canvas is None or self._mouth is None or self._left_pupil is None or self._right_pupil is None:
            return

        t = time.monotonic() - self._start_ts
        pupil_dx = max(-9.0, min(9.0, self._face_x * 9.0))
        pupil_dy = max(-7.0, min(7.0, self._face_y * 7.0))

        self._canvas.coords(self._left_pupil, 175 + pupil_dx, 160 + pupil_dy, 195 + pupil_dx, 180 + pupil_dy)
        self._canvas.coords(self._right_pupil, 305 + pupil_dx, 160 + pupil_dy, 325 + pupil_dx, 180 + pupil_dy)

        if self._state == "speaking":
            openness = 22 + 14 * (0.5 + 0.5 * math.sin(t * 11.0))
            self._canvas.coords(self._mouth, 190, 300 - openness / 2.0, 310, 300 + openness / 2.0)
        elif self._state == "listening":
            self._canvas.coords(self._mouth, 196, 306, 304, 306)
        elif self._state == "thinking":
            wobble = 3.0 * math.sin(t * 6.0)
            self._canvas.coords(self._mouth, 195, 300 + wobble, 305, 300 - wobble)
        elif self._state == "offline":
            self._canvas.coords(self._mouth, 198, 304, 302, 304)
        else:
            self._canvas.coords(self._mouth, 194, 302, 306, 302)

    def pump(self) -> None:
        if not self._running or self._root is None:
            return

        self._animate_face()
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
        self._chat_text.configure(state=tk.NORMAL)
        self._chat_text.insert(tk.END, f"{speaker}: {text}\n")
        self._chat_lines += 1
        if self._chat_lines > 120:
            self._chat_text.delete("1.0", "3.0")
            self._chat_lines -= 2
        self._chat_text.see(tk.END)
        self._chat_text.configure(state=tk.DISABLED)

    def set_face_target(self, x: float, y: float, found: bool, _face_count: int = 0) -> None:
        if found:
            self._face_x = max(-1.0, min(1.0, x))
            self._face_y = max(-1.0, min(1.0, y))
        else:
            self._face_x *= 0.9
            self._face_y *= 0.9

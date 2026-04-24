#!/usr/bin/env python3
"""
Animate Newton's method solving a user-entered function.

This is a first "math function solving itself" tool:
- the user types f(x)
- the program computes Newton iterations
- the animation shows the graph, tangent, x-axis intercept, and iteration history

Dependencies:
- numpy
- opencv-python
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


WINDOW_NAME = "Newton Function Animation"
WIDTH = 1380
HEIGHT = 860
GRAPH_LEFT = 60
GRAPH_TOP = 70
GRAPH_WIDTH = 860
GRAPH_HEIGHT = 720
PANEL_LEFT = 950
PANEL_TOP = 70
PANEL_WIDTH = 380
PANEL_HEIGHT = 720
BG = (11, 14, 24)
GRAPH_BG = (15, 20, 34)
PANEL_BG = (18, 24, 38)
GRID = (42, 52, 78)
AXIS = (118, 136, 180)
CURVE = (130, 222, 255)
POINT = (255, 194, 122)
NEXT_POINT = (170, 255, 176)
TEXT = (235, 239, 247)
MUTED = (156, 167, 186)
TANGENT = (255, 142, 142)
STEP_LINE = (250, 224, 146)


def _safe_names() -> dict[str, object]:
    return {
        "np": np,
        "math": math,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "pi": np.pi,
        "e": np.e,
    }


def compile_expression(expr: str):
    code = compile(expr, "<user-function>", "eval")

    def evaluate(x):
        scope = dict(_safe_names())
        scope["x"] = x
        return eval(code, {"__builtins__": {}}, scope)

    return evaluate


def numerical_derivative(fn, x: float) -> float:
    h = 1e-5 * max(1.0, abs(x))
    return float((fn(x + h) - fn(x - h)) / (2.0 * h))


@dataclass
class NewtonStep:
    index: int
    x: float
    fx: float
    slope: float
    next_x: float


def compute_newton_steps(fn, x0: float, max_iter: int = 8, tol: float = 1e-8) -> list[NewtonStep]:
    steps: list[NewtonStep] = []
    x = float(x0)

    for i in range(max_iter):
        fx = float(fn(x))
        if not np.isfinite(fx):
            raise ValueError("Function evaluation became non-finite. Try another expression or range.")

        slope = numerical_derivative(fn, x)
        if not np.isfinite(slope) or abs(slope) < 1e-10:
            raise ValueError("Derivative became too small for Newton's method. Try another initial guess.")

        next_x = float(x - fx / slope)
        steps.append(NewtonStep(index=i, x=x, fx=fx, slope=slope, next_x=next_x))
        x = next_x

        if abs(fx) < tol:
            break

    return steps


def auto_window(steps: list[NewtonStep]) -> tuple[float, float, float, float]:
    xs = [step.x for step in steps] + [step.next_x for step in steps]
    xmin = min(xs)
    xmax = max(xs)
    xpad = max(2.0, (xmax - xmin) * 0.45 + 1.0)
    xmin -= xpad
    xmax += xpad
    return xmin, xmax, -1.0, 1.0


def graph_samples(fn, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x_min, x_max, 1600, dtype=np.float64)
    ys = np.asarray(fn(xs), dtype=np.float64)
    finite = np.isfinite(ys)
    if not np.any(finite):
        raise ValueError("Function is non-finite across the current graph window.")

    clipped = ys[finite]
    y_low = float(np.percentile(clipped, 6))
    y_high = float(np.percentile(clipped, 94))
    span = max(2.0, y_high - y_low)
    center = 0.5 * (y_low + y_high)
    y_min = center - span * 0.8
    y_max = center + span * 0.8
    y_min = min(y_min, -1.0)
    y_max = max(y_max, 1.0)
    return xs, ys, y_min, y_max


def to_screen(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> tuple[int, int]:
    sx = int(GRAPH_LEFT + (x - x_min) / (x_max - x_min) * GRAPH_WIDTH)
    sy = int(GRAPH_TOP + GRAPH_HEIGHT - (y - y_min) / (y_max - y_min) * GRAPH_HEIGHT)
    return sx, sy


def draw_graph_frame(
    expr: str,
    xs: np.ndarray,
    ys: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    steps: list[NewtonStep],
    active_index: int,
    phase: float,
    paused: bool,
) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
    cv2.rectangle(image, (GRAPH_LEFT, GRAPH_TOP), (GRAPH_LEFT + GRAPH_WIDTH, GRAPH_TOP + GRAPH_HEIGHT), GRAPH_BG, -1)
    cv2.rectangle(image, (PANEL_LEFT, PANEL_TOP), (PANEL_LEFT + PANEL_WIDTH, PANEL_TOP + PANEL_HEIGHT), PANEL_BG, -1)
    cv2.rectangle(image, (GRAPH_LEFT, GRAPH_TOP), (GRAPH_LEFT + GRAPH_WIDTH, GRAPH_TOP + GRAPH_HEIGHT), AXIS, 1)
    cv2.rectangle(image, (PANEL_LEFT, PANEL_TOP), (PANEL_LEFT + PANEL_WIDTH, PANEL_TOP + PANEL_HEIGHT), AXIS, 1)

    for frac in np.linspace(0.0, 1.0, 11):
        x = int(GRAPH_LEFT + frac * GRAPH_WIDTH)
        y = int(GRAPH_TOP + frac * GRAPH_HEIGHT)
        cv2.line(image, (x, GRAPH_TOP), (x, GRAPH_TOP + GRAPH_HEIGHT), GRID, 1, cv2.LINE_AA)
        cv2.line(image, (GRAPH_LEFT, y), (GRAPH_LEFT + GRAPH_WIDTH, y), GRID, 1, cv2.LINE_AA)

    if x_min <= 0.0 <= x_max:
        x_axis = to_screen(0.0, 0.0, x_min, x_max, y_min, y_max)[0]
        cv2.line(image, (x_axis, GRAPH_TOP), (x_axis, GRAPH_TOP + GRAPH_HEIGHT), AXIS, 2, cv2.LINE_AA)
    if y_min <= 0.0 <= y_max:
        y_axis = to_screen(0.0, 0.0, x_min, x_max, y_min, y_max)[1]
        cv2.line(image, (GRAPH_LEFT, y_axis), (GRAPH_LEFT + GRAPH_WIDTH, y_axis), AXIS, 2, cv2.LINE_AA)

    pts = []
    finite = np.isfinite(ys)
    for xv, yv, ok in zip(xs, ys, finite):
        if not ok:
            if len(pts) > 1:
                cv2.polylines(image, [np.array(pts, dtype=np.int32)], False, CURVE, 2, cv2.LINE_AA)
            pts = []
            continue
        pts.append(to_screen(float(xv), float(yv), x_min, x_max, y_min, y_max))
    if len(pts) > 1:
        cv2.polylines(image, [np.array(pts, dtype=np.int32)], False, CURVE, 2, cv2.LINE_AA)

    for step in steps[:active_index]:
        a = to_screen(step.x, 0.0, x_min, x_max, y_min, y_max)
        b = to_screen(step.x, step.fx, x_min, x_max, y_min, y_max)
        c = to_screen(step.next_x, 0.0, x_min, x_max, y_min, y_max)
        cv2.line(image, a, b, (88, 124, 196), 1, cv2.LINE_AA)
        cv2.line(image, b, c, (112, 188, 160), 1, cv2.LINE_AA)
        cv2.circle(image, b, 4, POINT, -1, cv2.LINE_AA)
        cv2.circle(image, c, 4, NEXT_POINT, -1, cv2.LINE_AA)

    step = steps[min(active_index, len(steps) - 1)]
    current = to_screen(step.x, step.fx, x_min, x_max, y_min, y_max)
    root_guess = to_screen(step.x, 0.0, x_min, x_max, y_min, y_max)
    next_guess = to_screen(step.next_x, 0.0, x_min, x_max, y_min, y_max)

    vertical_t = min(1.0, phase / 0.33)
    tangent_t = np.clip((phase - 0.33) / 0.34, 0.0, 1.0)
    step_t = np.clip((phase - 0.67) / 0.33, 0.0, 1.0)

    interp_vertical = (
        int(root_guess[0] + (current[0] - root_guess[0]) * vertical_t),
        int(root_guess[1] + (current[1] - root_guess[1]) * vertical_t),
    )
    cv2.line(image, root_guess, interp_vertical, STEP_LINE, 2, cv2.LINE_AA)

    tangent_len = max(4.0, (x_max - x_min) * 0.28)
    line_x0 = step.x - tangent_len
    line_x1 = step.x + tangent_len
    line_y0 = step.fx + step.slope * (line_x0 - step.x)
    line_y1 = step.fx + step.slope * (line_x1 - step.x)
    p0 = to_screen(line_x0, line_y0, x_min, x_max, y_min, y_max)
    p1 = to_screen(line_x1, line_y1, x_min, x_max, y_min, y_max)
    tangent_mid = (
        int(p0[0] + (p1[0] - p0[0]) * tangent_t),
        int(p0[1] + (p1[1] - p0[1]) * tangent_t),
    )
    cv2.line(image, p0, tangent_mid, TANGENT, 2, cv2.LINE_AA)

    intercept = (
        int(current[0] + (next_guess[0] - current[0]) * step_t),
        int(current[1] + (next_guess[1] - current[1]) * step_t),
    )
    if step_t > 0.0:
        cv2.line(image, current, intercept, NEXT_POINT, 2, cv2.LINE_AA)

    cv2.circle(image, current, 7, POINT, -1, cv2.LINE_AA)
    if step_t > 0.0:
        cv2.circle(image, intercept, 6, NEXT_POINT, -1, cv2.LINE_AA)

    cv2.putText(image, "Newton Function Animation", (60, 42), cv2.FONT_HERSHEY_DUPLEX, 0.9, TEXT, 1, cv2.LINE_AA)
    cv2.putText(image, f"f(x) = {expr}", (60, 812), cv2.FONT_HERSHEY_DUPLEX, 0.62, TEXT, 1, cv2.LINE_AA)
    cv2.putText(image, "q quit   p pause   r restart   n new function", (60, 836), cv2.FONT_HERSHEY_DUPLEX, 0.52, MUTED, 1, cv2.LINE_AA)

    panel_lines = [
        "Current Step",
        f"iteration n = {step.index}",
        f"x_n = {step.x: .6f}",
        f"f(x_n) = {step.fx: .6f}",
        f"f'(x_n) = {step.slope: .6f}",
        f"x_(n+1) = {step.next_x: .6f}",
        "",
        "Update Rule",
        "x_(n+1) = x_n - f(x_n) / f'(x_n)",
        "",
        "Status",
        "PAUSED" if paused else "RUNNING",
    ]

    y = PANEL_TOP + 36
    for idx, line in enumerate(panel_lines):
        if not line:
            y += 12
            continue
        size = 0.72 if idx in (0, 7, 10) else 0.56
        color = TEXT if idx in (0, 7, 10) else MUTED
        if line in ("PAUSED", "RUNNING"):
            color = POINT if paused else NEXT_POINT
        cv2.putText(image, line, (PANEL_LEFT + 20, y), cv2.FONT_HERSHEY_DUPLEX, size, color, 1, cv2.LINE_AA)
        y += 30

    cv2.putText(image, "Iteration History", (PANEL_LEFT + 20, PANEL_TOP + 410), cv2.FONT_HERSHEY_DUPLEX, 0.72, TEXT, 1, cv2.LINE_AA)
    y = PANEL_TOP + 445
    for prev in steps[: min(len(steps), 7)]:
        color = TEXT if prev.index == step.index else MUTED
        line = f"{prev.index}: x={prev.x: .4f}  ->  {prev.next_x: .4f}"
        cv2.putText(image, line, (PANEL_LEFT + 20, y), cv2.FONT_HERSHEY_DUPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 28

    return image


def request_problem() -> tuple[str, float]:
    print("Enter a function in x. Example: x**3 - x - 2")
    expr = input("f(x) = ").strip() or "x**3 - x - 2"
    x0_raw = input("Initial guess x0 [default 1.5] = ").strip()
    x0 = float(x0_raw) if x0_raw else 1.5
    return expr, x0


def prepare_problem(expr: str, x0: float):
    fn = compile_expression(expr)
    steps = compute_newton_steps(fn, x0)
    x_min, x_max, _, _ = auto_window(steps)
    xs, ys, y_min, y_max = graph_samples(fn, x_min, x_max)
    return fn, steps, xs, ys, x_min, x_max, y_min, y_max


def main() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    while True:
        try:
            expr, x0 = request_problem()
            _, steps, xs, ys, x_min, x_max, y_min, y_max = prepare_problem(expr, x0)
        except Exception as exc:
            print(f"Problem setup failed: {exc}")
            continue

        paused = False
        frame_in_step = 0
        active_index = 0

        while True:
            phase = (frame_in_step % 90) / 89.0
            frame = draw_graph_frame(expr, xs, ys, x_min, x_max, y_min, y_max, steps, active_index, phase, paused)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyAllWindows()
                return
            if key in (ord("p"), ord(" ")):
                paused = not paused
            elif key == ord("r"):
                frame_in_step = 0
                active_index = 0
            elif key == ord("n"):
                break

            if not paused:
                frame_in_step += 1
                if frame_in_step >= 90:
                    frame_in_step = 0
                    if active_index < len(steps) - 1:
                        active_index += 1


if __name__ == "__main__":
    main()

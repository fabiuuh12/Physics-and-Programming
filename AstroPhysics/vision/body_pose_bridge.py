#!/usr/bin/env python3
"""
Full-body pose UDP bridge for the C++ exosuit scene.

Bridge transport:
- UDP localhost:50506
- ASCII CSV

Packet format:
timestamp,valid,
center_x,center_y,torso_h,shoulder_span,
head_x,head_y,
left_hand_x,left_hand_y,right_hand_x,right_hand_y,
left_foot_x,left_foot_y,right_foot_x,right_foot_y,
crouch,arms_up,lean
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

import cv2
import numpy as np

from body_exoskeleton_tracker import (
    EXO_BONES,
    FPSCounter,
    LandmarkSmoother,
    NOSE,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_FOOT_INDEX,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    POSE_MODEL_PATH,
    PoseMotionState,
    PoseTracker,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    WINDOW_NAME,
    _all_visible,
    _draw_head_helmet,
    _draw_hud,
    _draw_joint,
    _draw_segment,
    _draw_torso_plate,
    _movement_state,
    _visible,
    _xy,
)


UDP_HOST = "127.0.0.1"
UDP_PORT = 50506
WINDOW_TITLE = "Body Pose Bridge"


@dataclass
class BodyBridgeState:
    valid: bool = False
    center_x: float = 0.5
    center_y: float = 0.6
    torso_h: float = 0.28
    shoulder_span: float = 0.16
    head_x: float = 0.5
    head_y: float = 0.22
    left_hand_x: float = 0.42
    left_hand_y: float = 0.48
    right_hand_x: float = 0.58
    right_hand_y: float = 0.48
    left_foot_x: float = 0.46
    left_foot_y: float = 0.92
    right_foot_x: float = 0.54
    right_foot_y: float = 0.92
    crouch: int = 0
    arms_up: int = 0
    lean: int = 0


def _swap_left_right(state: BodyBridgeState) -> BodyBridgeState:
    return BodyBridgeState(
        valid=state.valid,
        center_x=state.center_x,
        center_y=state.center_y,
        torso_h=state.torso_h,
        shoulder_span=state.shoulder_span,
        head_x=state.head_x,
        head_y=state.head_y,
        left_hand_x=state.right_hand_x,
        left_hand_y=state.right_hand_y,
        right_hand_x=state.left_hand_x,
        right_hand_y=state.left_hand_y,
        left_foot_x=state.right_foot_x,
        left_foot_y=state.right_foot_y,
        right_foot_x=state.left_foot_x,
        right_foot_y=state.left_foot_y,
        crouch=state.crouch,
        arms_up=state.arms_up,
        lean=-state.lean,
    )


def _pick_xy(landmarks: np.ndarray, primary: int, fallbacks: tuple[int, ...]) -> tuple[float, float]:
    if _visible(landmarks, primary):
        return float(landmarks[primary, 0]), float(landmarks[primary, 1])
    for idx in fallbacks:
        if _visible(landmarks, idx):
            return float(landmarks[idx, 0]), float(landmarks[idx, 1])
    return 0.5, 0.5


def _compute_bridge_state(landmarks: np.ndarray, pose_state: PoseMotionState) -> BodyBridgeState:
    required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if not _all_visible(landmarks, required):
        return BodyBridgeState(valid=False)

    shoulder_center = 0.5 * (landmarks[LEFT_SHOULDER, :2] + landmarks[RIGHT_SHOULDER, :2])
    hip_center = 0.5 * (landmarks[LEFT_HIP, :2] + landmarks[RIGHT_HIP, :2])
    center = 0.5 * (shoulder_center + hip_center)
    torso_h = float(np.linalg.norm(shoulder_center - hip_center))
    shoulder_span = float(np.linalg.norm(landmarks[LEFT_SHOULDER, :2] - landmarks[RIGHT_SHOULDER, :2]))

    head_x, head_y = _pick_xy(landmarks, NOSE, (LEFT_SHOULDER, RIGHT_SHOULDER))
    left_hand_x, left_hand_y = _pick_xy(landmarks, LEFT_WRIST, (LEFT_ELBOW, LEFT_SHOULDER))
    right_hand_x, right_hand_y = _pick_xy(landmarks, RIGHT_WRIST, (RIGHT_ELBOW, RIGHT_SHOULDER))
    left_foot_x, left_foot_y = _pick_xy(landmarks, LEFT_FOOT_INDEX, (LEFT_ANKLE, LEFT_KNEE, LEFT_HIP))
    right_foot_x, right_foot_y = _pick_xy(landmarks, RIGHT_FOOT_INDEX, (RIGHT_ANKLE, RIGHT_KNEE, RIGHT_HIP))

    lean = 0
    if pose_state.lean_label == "left":
        lean = -1
    elif pose_state.lean_label == "right":
        lean = 1

    return BodyBridgeState(
        valid=True,
        center_x=float(center[0]),
        center_y=float(center[1]),
        torso_h=torso_h,
        shoulder_span=shoulder_span,
        head_x=head_x,
        head_y=head_y,
        left_hand_x=left_hand_x,
        left_hand_y=left_hand_y,
        right_hand_x=right_hand_x,
        right_hand_y=right_hand_y,
        left_foot_x=left_foot_x,
        left_foot_y=left_foot_y,
        right_foot_x=right_foot_x,
        right_foot_y=right_foot_y,
        crouch=int(pose_state.crouching),
        arms_up=int(pose_state.arms_up),
        lean=lean,
    )


def _send_bridge_packet(sock: socket.socket, state: BodyBridgeState, timestamp_s: float) -> None:
    packet = (
        f"{timestamp_s:.3f},"
        f"{1 if state.valid else 0},"
        f"{state.center_x:.5f},{state.center_y:.5f},{state.torso_h:.5f},{state.shoulder_span:.5f},"
        f"{state.head_x:.5f},{state.head_y:.5f},"
        f"{state.left_hand_x:.5f},{state.left_hand_y:.5f},{state.right_hand_x:.5f},{state.right_hand_y:.5f},"
        f"{state.left_foot_x:.5f},{state.left_foot_y:.5f},{state.right_foot_x:.5f},{state.right_foot_y:.5f},"
        f"{state.crouch},{state.arms_up},{state.lean}\n"
    )
    sock.sendto(packet.encode("ascii"), (UDP_HOST, UDP_PORT))


def _draw_bridge_hud(frame: np.ndarray, fps: float, mirror: bool, pose_state: PoseMotionState, tracked: bool) -> None:
    _draw_hud(frame, fps, mirror, pose_state, tracked)
    h, _ = frame.shape[:2]
    cv2.putText(
        frame,
        f"udp:{UDP_HOST}:{UDP_PORT}  model:{POSE_MODEL_PATH.name}",
        (14, h - 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (208, 220, 244),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (camera index 0).")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        pose_tracker = PoseTracker()
        pose_tracker.setup()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        cap.release()
        return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    fps_counter = FPSCounter()
    smoother = LandmarkSmoother()
    mirror = True
    prev_center: np.ndarray | None = None
    prev_state_ts = time.perf_counter()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 760)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: webcam frame read failed.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            fps = fps_counter.update()
            timestamp_ms = int(time.time() * 1000)
            now = time.perf_counter()
            dt = max(1.0e-4, now - prev_state_ts)
            prev_state_ts = now

            tracked = False
            pose_state = PoseMotionState()
            bridge_state = BodyBridgeState(valid=False)
            raw = pose_tracker.detect(frame, timestamp_ms)

            if raw is not None:
                landmarks = smoother.smooth(raw)
                tracked = True
                pose_state, prev_center = _movement_state(landmarks, prev_center, dt)
                bridge_state = _compute_bridge_state(landmarks, pose_state)
                if mirror:
                    bridge_state = _swap_left_right(bridge_state)

                _draw_torso_plate(frame, landmarks, frame_w, frame_h)
                _draw_head_helmet(frame, landmarks, frame_w, frame_h)
                for a_idx, b_idx in EXO_BONES:
                    if not _all_visible(landmarks, (a_idx, b_idx)):
                        continue
                    _draw_segment(frame, _xy(landmarks, a_idx, frame_w, frame_h), _xy(landmarks, b_idx, frame_w, frame_h), (116, 214, 255))
                for idx in (
                    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
                    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
                ):
                    if _visible(landmarks, idx):
                        _draw_joint(frame, _xy(landmarks, idx, frame_w, frame_h), 7, (118, 255, 216))
            else:
                smoother.reset()
                prev_center = None

            _send_bridge_packet(sock, bridge_state, time.time())
            _draw_bridge_hud(frame, fps, mirror, pose_state, tracked)
            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                mirror = not mirror
    finally:
        pose_tracker.close()
        cap.release()
        sock.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

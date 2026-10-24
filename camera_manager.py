"""Robust Camera Stream Manager with Windows Backend Detection and Virtual Demo Fallback.

Supports cv2.CAP_DSHOW, cv2.CAP_MSMF, and auto-detects physical cameras.
If no webcam is connected, provides a smooth animated virtual avatar camera
allowing immediate testing and demonstration without hardware requirements.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from emotion_detector import EMOTION_LABELS


class CameraStream:
    """Manages real webcam capture with fallback to high-quality virtual face simulation."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        force_demo: bool = False,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.is_demo = force_demo
        self.cap: cv2.VideoCapture | None = None
        self.sim_emotion: str = "happy"
        self.start_time = time.monotonic()
        self.sample_face_img: np.ndarray | None = None

        # Load sample face if available
        sample_path = Path("assets/test_faces/neutral_face.jpg")
        if sample_path.exists():
            img = cv2.imread(str(sample_path))
            if img is not None:
                self.sample_face_img = cv2.resize(img, (self.width, self.height))

        if not force_demo:
            self.cap = self._init_camera()
            if self.cap is None or not self.cap.isOpened():
                print("\n[!] No physical webcam detected. Switching to Manual Demo Mode.")
                print("[i] Demo labels are keyboard-selected; the expression model is not running.\n")
                self.is_demo = True

    def _init_camera(self) -> cv2.VideoCapture | None:
        """Attempts to open physical webcam with Windows-specific backends."""
        backends = []
        if sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        # Try specified camera index first, then 0, 1, 2
        indices_to_try = [self.camera_index] if self.camera_index != 0 else [0, 1, 2]

        for idx in indices_to_try:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        # Set resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        # Verify we can actually read a valid frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            print(f"[+] Webcam detected on index {idx} ({backend}).")
                            return cap
                        cap.release()
                except Exception:
                    continue
        return None

    def read(self) -> tuple[bool, np.ndarray, bool]:
        """Reads next frame.

        Returns:
            (success, frame, is_demo_mode)
        """
        if not self.is_demo and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                # Mirror frame horizontally for intuitive mirror experience
                frame = cv2.flip(frame, 1)
                return True, frame, False
            else:
                # Camera disconnected during execution
                print("[!] Webcam disconnected. Falling back to demo mode.")
                self.is_demo = True

        # Generate synthetic or sample face frame for demo mode
        frame = self._generate_simulated_frame()
        return True, frame, True

    def set_simulated_emotion(self, emotion: str) -> None:
        """Sets the active simulated expression in demo mode."""
        if emotion in EMOTION_LABELS:
            self.sim_emotion = emotion

    def _generate_simulated_frame(self) -> np.ndarray:
        """Generates a dynamic facial frame with animated breathing/eyeblink motion."""
        now = time.monotonic()
        t = now - self.start_time

        if self.sample_face_img is not None:
            frame = self.sample_face_img.copy()
            # Subtle breathing scale effect
            scale = 1.0 + 0.015 * math.sin(t * 1.8)
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.5 * math.sin(t * 0.9), scale)
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        else:
            # Procedural stylized avatar face canvas
            frame = np.full((self.height, self.width, 3), (35, 30, 42), dtype=np.uint8)
            cx, cy = self.width // 2, self.height // 2
            # Face contour
            cv2.ellipse(frame, (cx, cy), (110, 140), 0, 0, 360, (215, 190, 175), -1, cv2.LINE_AA)
            # Eyes
            cv2.circle(frame, (cx - 40, cy - 20), 12, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx + 40, cy - 20), 12, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx - 40, cy - 20), 6, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx + 40, cy - 20), 6, (40, 40, 40), -1, cv2.LINE_AA)
            # Mouth
            cv2.ellipse(frame, (cx, cy + 50), (35, 18), 0, 0, 180, (80, 50, 180), -1, cv2.LINE_AA)

        # Draw a translucent simulation banner
        cv2.putText(
            frame,
            f"MANUAL DEMO // SELECTED LABEL: {self.sim_emotion.upper()}",
            (18, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.48,
            (50, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Manual preview only: press 1-7 to change the label; no face inference here",
            (18, 52),
            cv2.FONT_HERSHEY_DUPLEX,
            0.38,
            (180, 180, 200),
            1,
            cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

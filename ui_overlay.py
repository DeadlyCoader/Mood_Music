"""Futuristic Cyberpunk / Glassmorphic HUD and UI Overlay for Real-Time Mood Music.

Renders translucent panels, live emotion breakdown bar graphs, face tracking reticles,
track recommendation cards, and interactive clickable buttons directly on the OpenCV canvas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from emotion_detector import EMOTION_DISPLAY, EMOTION_LABELS
from music_recommender import MoodProfile, ProviderType, Track


@dataclass
class ButtonRect:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    label: str

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class UIOverlay:
    """Renders high-tech overlays, charts, and interactive HUD buttons."""

    CANVAS_W = 1200
    CANVAS_H = 700

    def __init__(self) -> None:
        self.buttons: list[ButtonRect] = []
        self.hover_btn: str | None = None
        self.last_click_action: str | None = None

    def create_canvas(self, camera_frame: np.ndarray | None = None) -> np.ndarray:
        """Composes a high-resolution 1200x700 canvas with the camera stream centered."""
        canvas = np.zeros((self.CANVAS_H, self.CANVAS_W, 3), dtype=np.uint8)
        # Deep high-tech background gradient
        canvas[:, :] = (18, 16, 22)

        if camera_frame is not None and camera_frame.size > 0:
            fh, fw = camera_frame.shape[:2]
            # Target video display area: width=620, height=480 at (x=280, y=95)
            target_w, target_h = 620, 480
            aspect = fw / max(1, fh)

            if aspect >= (target_w / target_h):
                scaled_w = target_w
                scaled_h = int(target_w / aspect)
            else:
                scaled_h = target_h
                scaled_w = int(target_h * aspect)

            resized = cv2.resize(camera_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            offset_x = 280 + (target_w - scaled_w) // 2
            offset_y = 95 + (target_h - scaled_h) // 2
            canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = resized

            # Frame border around video
            cv2.rectangle(
                canvas,
                (offset_x - 2, offset_y - 2),
                (offset_x + scaled_w + 1, offset_y + scaled_h + 1),
                (60, 60, 75),
                1,
            )
            return canvas, (offset_x, offset_y, scaled_w, scaled_h, fw, fh)

        return canvas, (280, 95, 620, 480, 620, 480)

    def draw_alpha_rect(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        alpha: float = 0.6,
        border_color: tuple[int, int, int] | None = None,
    ) -> None:
        """Draws a semi-transparent glass panel."""
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return

        sub = img[y1:y2, x1:x2]
        overlay = np.full_like(sub, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)
        if border_color:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)

    def draw_text(
        self,
        img: np.ndarray,
        text: str,
        pos: tuple[int, int],
        font_scale: float = 0.52,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1,
        shadow: bool = True,
    ) -> None:
        """Renders crisp text with subtle drop-shadow for high legibility."""
        x, y = pos
        if shadow:
            cv2.putText(
                img,
                text,
                (x + 1, y + 1),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 1,
                cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_face_reticle(
        self,
        img: np.ndarray,
        bbox: tuple[int, int, int, int],
        emotion: str,
        confidence: float,
        video_meta: tuple[int, int, int, int, int, int],
    ) -> None:
        """Draws a futuristic corner-bracket face reticle and emotion badge."""
        vx, vy, vw, vh, orig_w, orig_h = video_meta
        x, y, w, h = bbox

        # Map original camera coords to canvas coords
        scale_x = vw / max(1, orig_w)
        scale_y = vh / max(1, orig_h)
        cx1 = int(vx + x * scale_x)
        cy1 = int(vy + y * scale_y)
        cx2 = int(cx1 + w * scale_x)
        cy2 = int(cy1 + h * scale_y)

        color = EMOTION_DISPLAY.get(emotion, ("Neutral", "😌", (200, 200, 200)))[2]
        corner_len = max(12, int(min(cx2 - cx1, cy2 - cy1) * 0.22))
        thickness = 2

        # Draw tech corner brackets
        # Top-Left
        cv2.line(img, (cx1, cy1), (cx1 + corner_len, cy1), color, thickness)
        cv2.line(img, (cx1, cy1), (cx1, cy1 + corner_len), color, thickness)
        # Top-Right
        cv2.line(img, (cx2, cy1), (cx2 - corner_len, cy1), color, thickness)
        cv2.line(img, (cx2, cy1), (cx2, cy1 + corner_len), color, thickness)
        # Bottom-Left
        cv2.line(img, (cx1, cy2), (cx1 + corner_len, cy2), color, thickness)
        cv2.line(img, (cx1, cy2), (cx1, cy2 - corner_len), color, thickness)
        # Bottom-Right
        cv2.line(img, (cx2, cy2), (cx2 - corner_len, cy2), color, thickness)
        cv2.line(img, (cx2, cy2), (cx2, cy2 - corner_len), color, thickness)

        # Subtle bounding box line
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (40, 40, 50), 1)

        # Emotion badge pill on top
        badge_text = f"{emotion.upper()} {confidence:.0%}"
        badge_w = max(110, len(badge_text) * 10 + 20)
        badge_y = max(vy + 5, cy1 - 28)
        self.draw_alpha_rect(img, cx1, badge_y, cx1 + badge_w, badge_y + 24, (20, 20, 25), 0.85, color)
        self.draw_text(img, badge_text, (cx1 + 8, badge_y + 17), 0.44, color, 1)

    def draw_hud(
        self,
        canvas: np.ndarray,
        scores: dict[str, float],
        current_emotion: str,
        confidence: float,
        profile: MoodProfile,
        recommendations: list[Track],
        provider: ProviderType,
        auto_play: bool,
        is_cooldown: bool,
        cooldown_remaining: float,
        status_message: str,
        fps: float,
        is_demo_mode: bool = False,
        model_name: str = "HSEmotion B0 ONNX",
    ) -> None:
        """Renders the complete futuristic glassmorphic UI interface."""
        self.buttons.clear()

        # 1. Top Header Bar
        self.draw_alpha_rect(canvas, 15, 12, 1185, 75, (25, 22, 32), 0.88, (70, 60, 90))
        self.draw_text(canvas, "MOOD MUSIC AI", (35, 48), 0.85, (255, 255, 255), 2)
        self.draw_text(canvas, "// REAL-TIME EMOTION WEB PLAYER", (245, 48), 0.45, (160, 160, 190), 1)

        # Header status pills: Model, Camera Mode, FPS
        model_badge = "MODEL: HSEmotion OFF" if is_demo_mode else f"MODEL: {model_name}"
        self.draw_alpha_rect(canvas, 665, 26, 810, 60, (30, 28, 40), 0.9, (200, 130, 255))
        self.draw_text(canvas, model_badge, (678, 49), 0.40, (220, 180, 255), 1)

        mode_text = "MANUAL DEMO" if is_demo_mode else "LIVE CAMERA"
        mode_color = (255, 170, 50) if is_demo_mode else (80, 230, 120)
        self.draw_alpha_rect(canvas, 820, 26, 960, 60, (30, 28, 40), 0.9, mode_color)
        self.draw_text(canvas, mode_text, (832, 49), 0.42, mode_color, 1)

        fps_text = f"FPS: {fps:.1f}"
        self.draw_alpha_rect(canvas, 975, 26, 1075, 60, (30, 28, 40), 0.9, (180, 180, 200))
        self.draw_text(canvas, fps_text, (988, 49), 0.42, (200, 200, 220), 1)

        # 2. Left Panel: Emotion Metrics
        self.draw_alpha_rect(canvas, 15, 88, 265, 590, (22, 20, 28), 0.85, (50, 48, 65))
        self.draw_text(canvas, "EMOTION METRICS", (30, 118), 0.58, (230, 230, 240), 2)
        self.draw_text(canvas, "MODEL SCORES (NOT CERTAINTY)", (30, 136), 0.35, (140, 140, 160), 1)

        start_y = 175
        for emotion in EMOTION_LABELS:
            score = scores.get(emotion, 0.0)
            is_winner = emotion == current_emotion
            name, emoji, color = EMOTION_DISPLAY.get(emotion, (emotion, "", (180, 180, 180)))

            # Indicator dot / arrow
            label_color = color if is_winner else (180, 180, 190)
            prefix = "▶ " if is_winner else "  "
            self.draw_text(canvas, f"{prefix}{name.capitalize()}", (28, start_y), 0.48, label_color, 2 if is_winner else 1)
            self.draw_text(canvas, f"{score:.0%}", (210, start_y), 0.44, (200, 200, 220), 1)

            # Progress bar background
            bar_y = start_y + 8
            self.draw_alpha_rect(canvas, 30, bar_y, 245, bar_y + 10, (40, 38, 50), 0.9, None)
            # Active filled bar
            fill_w = int(215 * min(1.0, max(0.0, score)))
            if fill_w > 0:
                bar_color = color if is_winner else (100, 95, 120)
                self.draw_alpha_rect(canvas, 30, bar_y, 30 + fill_w, bar_y + 10, bar_color, 0.95, None)

            start_y += 54

        # 3. Right Panel: Music Recommendations
        self.draw_alpha_rect(canvas, 915, 88, 1185, 590, (22, 20, 28), 0.85, (50, 48, 65))
        self.draw_text(canvas, "MUSIC RADAR", (930, 118), 0.58, (230, 230, 240), 2)
        self.draw_text(canvas, "ONLINE STREAMING ENGINE", (930, 136), 0.38, (140, 140, 160), 1)

        # Mood Card Banner
        mood_color = profile.color_bgr
        self.draw_alpha_rect(canvas, 930, 155, 1170, 225, (30, 28, 40), 0.92, mood_color)
        mood_title = f"{current_emotion.upper()}: {profile.display_name.upper()}"
        self.draw_text(canvas, mood_title[:24], (945, 185), 0.54, mood_color, 2)
        self.draw_text(canvas, profile.vibe_summary[:40] + "...", (945, 208), 0.36, (180, 180, 195), 1)

        # Provider Selector Pill Button
        prov_rect = ButtonRect("cycle_provider", 930, 236, 1170, 268, f"PROVIDER: {provider.upper()} (CLICK / M)")
        self.buttons.append(prov_rect)
        self.draw_alpha_rect(canvas, prov_rect.x1, prov_rect.y1, prov_rect.x2, prov_rect.y2, (40, 38, 55), 0.9, (100, 90, 130))
        self.draw_text(canvas, prov_rect.label, (945, 258), 0.40, (220, 220, 240), 1)

        # Track Recommendations List (3 tracks)
        track_y = 282
        for idx, track in enumerate(recommendations[:3]):
            t_rect = ButtonRect(f"play_track_{idx}", 930, track_y, 1170, track_y + 58, track.title)
            self.buttons.append(t_rect)
            self.draw_alpha_rect(canvas, t_rect.x1, t_rect.y1, t_rect.x2, t_rect.y2, (30, 28, 38), 0.9, (65, 60, 80))
            
            # Number circle & title
            self.draw_text(canvas, f"[{idx + 1}] {track.title[:22]}", (945, track_y + 24), 0.46, (255, 255, 255), 1)
            self.draw_text(canvas, f"{track.artist} | {track.genre[:18]}", (945, track_y + 44), 0.36, (150, 150, 175), 1)
            self.draw_text(canvas, "▶", (1145, track_y + 35), 0.55, mood_color, 1)

            track_y += 66

        # Interactive Play Action Buttons
        play_btn = ButtonRect("play_top", 930, 492, 1170, 532, "▶ PLAY NOW (SPACE)")
        self.buttons.append(play_btn)
        self.draw_alpha_rect(canvas, play_btn.x1, play_btn.y1, play_btn.x2, play_btn.y2, (35, 120, 75), 0.95, (80, 240, 140))
        self.draw_text(canvas, play_btn.label, (955, 519), 0.50, (255, 255, 255), 2)

        auto_label = "⚙ AUTO-PLAY: ON (A)" if auto_play else "⚙ AUTO-PLAY: OFF (A)"
        auto_color = (130, 95, 30) if auto_play else (45, 42, 58)
        auto_border = (255, 190, 60) if auto_play else (90, 85, 110)
        auto_btn = ButtonRect("toggle_auto", 930, 542, 1170, 578, auto_label)
        self.buttons.append(auto_btn)
        self.draw_alpha_rect(canvas, auto_btn.x1, auto_btn.y1, auto_btn.x2, auto_btn.y2, auto_color, 0.9, auto_border)
        self.draw_text(canvas, auto_btn.label, (960, 566), 0.44, (240, 240, 240), 1)

        # 4. Bottom HUD Status & Legend Bar
        self.draw_alpha_rect(canvas, 15, 600, 1185, 685, (22, 20, 28), 0.9, (60, 55, 75))

        # Status Line
        status_color = (255, 200, 80) if is_cooldown else (100, 235, 150)
        self.draw_text(canvas, "STATUS:", (30, 630), 0.48, (180, 180, 200), 1)
        self.draw_text(canvas, status_message[:70], (105, 630), 0.48, status_color, 1)

        # Cooldown progress pill
        if is_cooldown:
            cd_text = f"COOLDOWN: {cooldown_remaining:.0f}s"
            self.draw_alpha_rect(canvas, 960, 612, 1170, 642, (45, 35, 20), 0.9, (255, 180, 50))
            self.draw_text(canvas, cd_text, (985, 633), 0.44, (255, 200, 80), 1)

        # Keyboard Guide Legend
        legend = "KEYS: [SPACE: Play]  [1-3: Pick Track]  [A: Auto-Play]  [M: Provider]  [T: Switch Model]  [Q: Quit]"
        self.draw_text(canvas, legend, (30, 665), 0.42, (150, 150, 175), 1)

    def handle_click(self, x: int, y: int) -> str | None:
        """Finds if a mouse click hit any button and returns the action identifier."""
        for btn in self.buttons:
            if btn.contains(x, y):
                return btn.id
        return None

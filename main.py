"""Real-time facial-expression music recommender using YuNet and HSEmotion ONNX."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2

from camera_manager import CameraStream
from emotion_detector import EMOTION_LABELS, EmotionDetector, EmotionSmoother
from music_recommender import MusicRecommender
from ui_overlay import UIOverlay
from web_player import WebMusicPlayer

# Safe terminal encoding on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class MoodMusicApp:
    """Coordinates camera capture, emotion inference, recommendations, HUD, and web playback."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.window_name = "MOOD MUSIC AI // Real-Time Online Music Player"

        print("\n========================================================")
        print("    MOOD MUSIC AI // REAL-TIME EMOTION WEB PLAYER       ")
        print("========================================================")
        print("[*] Initializing HSEmotion ONNX expression model + YuNet face detector...")

        self.detector = EmotionDetector()
        # Keep UI changes responsive while requiring two matching frames before autoplay.
        self.smoother = EmotionSmoother(window_size=3, stability_threshold=2)
        self.recommender = MusicRecommender()
        self.player = WebMusicPlayer(
            default_provider=args.provider,
            cooldown_seconds=args.cooldown,
            enable_automation=not args.no_auto_key,
            open_browser=not args.no_browser,
        )
        if args.autoplay:
            self.player.toggle_auto_play()

        self.overlay = UIOverlay()
        self.camera = CameraStream(
            camera_index=args.camera,
            width=640,
            height=480,
            force_demo=args.demo,
        )

        self.current_emotion = "neutral"
        self.confidence = 0.0
        self.scores: dict[str, float] = {e: 1.0 / len(EMOTION_LABELS) for e in EMOTION_LABELS}
        self.current_bbox: tuple[int, int, int, int] | None = None
        self.pending_click_action: str | None = None
        # Sample emotion twice per second for faster feedback without running
        # the classifier on every camera frame.
        self.analysis_interval = 0.5
        self.last_emotion_sample = 0.0

        # Frame rate calculation
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.monotonic()

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handles mouse clicks on interactive UI buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            action = self.overlay.handle_click(x, y)
            if action:
                self.pending_click_action = action

    def process_action(self, action: str) -> None:
        """Executes a UI or keyboard triggered action."""
        tracks = self.recommender.get_recommendations(self.current_emotion, limit=3)
        if action == "play_top":
            if tracks:
                self.player.play_track(tracks[0], self.current_emotion, force=True)
        elif action.startswith("play_track_"):
            try:
                idx = int(action.split("_")[-1])
                if 0 <= idx < len(tracks):
                    self.player.play_track(tracks[idx], self.current_emotion, force=True)
            except Exception:
                pass
        elif action == "toggle_auto":
            self.player.toggle_auto_play()
        elif action == "cycle_provider":
            self.player.cycle_provider()

    def run(self) -> int:
        """Main real-time application loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, UIOverlay.CANVAS_W, UIOverlay.CANVAS_H)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        print("[+] Application running successfully.")
        print("    [SPACE / Click Play] : Play current mood music online")
        print("    [1, 2, 3]            : Play specific recommended track")
        print("    [A]                  : Toggle Hands-Free Auto-Play mode")
        print("    [M]                  : Switch provider (YouTube -> Spotify -> YT Music)")
        print("    [P]                  : Browser Media Play/Pause via PyAutoGUI")
        print("    [1-8]                : Manually change the label in Demo Mode only")
        print("    [Q / ESC]            : Quit application\n")

        try:
            while True:
                ok, camera_frame, is_demo = self.camera.read()
                if not ok or camera_frame is None:
                    print("[!] Failed to acquire frame. Exiting.")
                    break

                # 1. Track the face every frame, but run emotion inference twice
                # per second. The sampled frame remains in memory only.
                if not is_demo:
                    faces = self.detector.detect_faces(camera_frame)
                    if faces is not None and len(faces) > 0:
                        # Process the most prominent face (largest area)
                        primary_face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, width, height = (int(round(float(v))) for v in primary_face[:4])
                        self.current_bbox = (max(0, x), max(0, y), width, height)

                        now = time.monotonic()
                        if now - self.last_emotion_sample >= self.analysis_interval:
                            self.last_emotion_sample = now
                            raw_scores, _, _, self.current_bbox = self.detector.analyze_face(
                                camera_frame, primary_face
                            )
                            (
                                self.scores,
                                self.current_emotion,
                                self.confidence,
                                is_stable,
                            ) = self.smoother.update(raw_scores)

                            # Hands-free Auto-play detection
                            if is_stable and self.player.can_auto_trigger(
                                self.current_emotion, self.confidence
                            ):
                                tracks = self.recommender.get_recommendations(self.current_emotion)
                                if tracks:
                                    self.player.play_track(tracks[0], self.current_emotion)
                    else:
                        self.current_bbox = None
                        # Drop stale scores when the face leaves view; autoplay
                        # must wait for a fresh stable expression sequence.
                        self.smoother.clear()
                        self.last_emotion_sample = 0.0
                        self.confidence = 0.0
                        self.scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
                        self.current_emotion = "neutral"
                else:
                    # In demo mode, use the simulation emotion
                    demo_emotion = self.camera.sim_emotion
                    self.current_emotion = demo_emotion
                    self.confidence = 0.92
                    # Create synthetic distribution peaking at simulated emotion
                    self.scores = {
                        e: (0.92 if e == demo_emotion else 0.015) for e in EMOTION_LABELS
                    }
                    self.current_bbox = (220, 140, 200, 220)

                    # Check auto-trigger in demo mode as well
                    if self.player.can_auto_trigger(self.current_emotion, self.confidence):
                        tracks = self.recommender.get_recommendations(self.current_emotion)
                        if tracks:
                            self.player.play_track(tracks[0], self.current_emotion)

                # 2. Process Pending Mouse Clicks
                if self.pending_click_action:
                    self.process_action(self.pending_click_action)
                    self.pending_click_action = None

                # 3. Recommendations & UI Rendering
                profile = self.recommender.get_profile(self.current_emotion)
                recommendations = self.recommender.get_recommendations(self.current_emotion, limit=3)

                canvas, video_meta = self.overlay.create_canvas(camera_frame)

                if self.current_bbox is not None:
                    self.overlay.draw_face_reticle(
                        canvas,
                        self.current_bbox,
                        self.current_emotion,
                        self.confidence,
                        video_meta,
                    )

                self.overlay.draw_hud(
                    canvas=canvas,
                    scores=self.scores,
                    current_emotion=self.current_emotion,
                    confidence=self.confidence,
                    profile=profile,
                    recommendations=recommendations,
                    provider=self.player.current_provider,
                    auto_play=self.player.auto_play,
                    is_cooldown=self.player.is_cooling_down(),
                    cooldown_remaining=self.player.remaining_cooldown(),
                    status_message=self.player.status_message,
                    fps=self.fps,
                    is_demo_mode=is_demo,
                    model_name=self.detector.model_name,
                )

                # 4. Display Window
                cv2.imshow(self.window_name, canvas)

                # 5. FPS Calculation
                self.frame_count += 1
                now = time.monotonic()
                if now - self.fps_timer >= 1.0:
                    self.fps = self.frame_count / (now - self.fps_timer)
                    self.frame_count = 0
                    self.fps_timer = now

                # 6. Keyboard Control Handling
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # Q or ESC
                    print("\n[+] Exiting application on user request.")
                    break
                elif key in (ord(" "), ord("p"), ord("P")):
                    # Play top track
                    self.process_action("play_top")
                elif key in (ord("1"), ord("2"), ord("3")):
                    idx = key - ord("1")
                    if is_demo:
                        # In demo mode, keys 1-8 switch emotions
                        demo_map = {0: "happy", 1: "sad", 2: "angry"}
                        if idx in demo_map:
                            self.camera.set_simulated_emotion(demo_map[idx])
                            self.player.set_status(f"Switched demo emotion to: {demo_map[idx].upper()}")
                    else:
                        self.process_action(f"play_track_{idx}")
                elif is_demo and key in (ord("4"), ord("5"), ord("6"), ord("7"), ord("8")):
                    demo_map = {
                        ord("4"): "surprised",
                        ord("5"): "fearful",
                        ord("6"): "disgust",
                        ord("7"): "neutral",
                        ord("8"): "contempt",
                    }
                    if key in demo_map:
                        self.camera.set_simulated_emotion(demo_map[key])
                        self.player.set_status(f"Switched demo emotion to: {demo_map[key].upper()}")
                elif key in (ord("a"), ord("A")):
                    self.process_action("toggle_auto")
                elif key in (ord("m"), ord("M")):
                    self.process_action("cycle_provider")
                elif key in (ord("k"), ord("K")):
                    self.player.toggle_playback_keys()
                elif key in (ord("+"), ord("=")):
                    self.player.volume_up()
                elif key in (ord("-"), ord("_")):
                    self.player.volume_down()
                elif key in (ord("d"), ord("D")):
                    self.camera.is_demo = not self.camera.is_demo
                    state = "ENABLED" if self.camera.is_demo else "DISABLED"
                    self.player.set_status(f"Demo Mode {state}")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            print("\nSession Complete.")
            print(f"Total online tracks triggered: {len(self.player.history)}")
            for event in self.player.history[-5:]:
                print(f" - [{event.mood.upper()}] {event.track_title} ({event.provider.upper()})")

        return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend music from a facial-expression estimate using HSEmotion ONNX."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--provider",
        choices=["youtube", "spotify", "ytmusic"],
        default="youtube",
        help="Default online streaming provider (default: youtube).",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Enable hands-free auto-play mode on startup.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=45.0,
        help="Seconds of cooldown between auto-play launches to prevent tab spamming (default: 45.0).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force interactive simulation/demo mode without requiring a physical webcam.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Disable actual browser launching (useful for dry runs and headless tests).",
    )
    parser.add_argument(
        "--no-auto-key",
        action="store_true",
        help="Disable automated PyAutoGUI keystroke injection.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Analyze one face image and print expression scores without opening the webcam.",
    )
    return parser.parse_args()


def inspect_image(image_path: Path) -> int:
    """Run the same YuNet + HSEmotion pipeline on one still image for a quick check."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not open image: {image_path}")
        return 2
    try:
        detector = EmotionDetector()
        faces = detector.detect_faces(image)
        if faces is None or len(faces) == 0:
            print("No face found. Use a clear, front-facing photo with good lighting.")
            return 1
        face = max(faces, key=lambda item: float(item[2] * item[3]))
        scores, top, _, _ = detector.analyze_face(image, face)
    except (FileNotFoundError, RuntimeError, cv2.error, ValueError) as error:
        print(f"Could not analyze image: {error}")
        return 2

    print(f"Top expression estimate: {top}")
    print("Scores:")
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  {label:10s} {score:.1%}")
    return 0


def main() -> int:
    args = parse_arguments()
    if args.image:
        return inspect_image(args.image)
    try:
        app = MoodMusicApp(args)
        return app.run()
    except (FileNotFoundError, RuntimeError, cv2.error) as error:
        print(f"Startup failed: {error}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

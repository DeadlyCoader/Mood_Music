"""Local facial-expression inference with YuNet and HSEmotion ONNX.

This estimates visible expression patterns; it cannot determine a person's
actual mood. Camera frames and face crops stay in memory and are not saved.
"""

from __future__ import annotations

import urllib.request  # Ensure HSEmotion's first-run downloader has this submodule.
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EMOTION_LABELS = [
    "angry",
    "contempt",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

EMOTION_DISPLAY = {
    "happy": ("Happy", "😃", (40, 215, 120)),
    "contempt": ("Contempt", "😒", (120, 150, 210)),
    "sad": ("Sad", "😢", (255, 140, 60)),
    "angry": ("Angry", "😡", (50, 50, 255)),
    "surprised": ("Surprised", "😲", (255, 230, 80)),
    "fearful": ("Fearful", "😨", (80, 140, 255)),
    "disgust": ("Disgust", "🤢", (180, 80, 180)),
    "neutral": ("Neutral", "😌", (200, 200, 200)),
}

# HSEmotion's eight-class model output order.
HSEMOTION_OUTPUT_LABELS = (
    "angry",
    "contempt",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)


def _find_yunet(filename: str, explicit_path: str | Path | None = None) -> Path:
    """Find the YuNet face detector beside the app or in its models folder."""
    project_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())
    candidates.extend(
        (
            project_dir / "models" / filename,
            project_dir / filename,
            project_dir.parent / "models" / filename,
            Path.cwd() / "models" / filename,
            Path.cwd() / filename,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    expected = project_dir / "models" / filename
    raise FileNotFoundError(
        f"Required face-detector model is missing: {filename}\n"
        f"Put it in {expected.parent}."
    )


class EmotionDetector:
    """Detect faces with YuNet and classify expressions with HSEmotion ONNX."""

    def __init__(
        self,
        yunet_path: str | Path | None = None,
        score_threshold: float = 0.50,
        nms_threshold: float = 0.30,
    ) -> None:
        self.yunet_path = _find_yunet("face_detection_yunet_2023mar.onnx", yunet_path)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.model_name = "HSEmotion B2 ONNX"

        detector_factory = getattr(cv2, "FaceDetectorYN_create", None)
        if detector_factory is None:
            raise RuntimeError(
                "Your OpenCV is too old for YuNet. Install the project requirements "
                "with: python -m pip install -r requirements.txt"
            )

        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        except ImportError as error:
            raise RuntimeError(
                "HSEmotion is not installed. Run: "
                "python -m pip install -r requirements.txt"
            ) from error

        try:
            self.detector = detector_factory(
                str(self.yunet_path),
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                5000,
            )
            # Download the pretrained B2 ONNX classifier on first run and
            # reuse it from the user's local model cache afterward.
            self.classifier: Any = HSEmotionRecognizer(
                model_name="enet_b2_8"
            )
        except Exception as error:
            raise RuntimeError(
                "Could not initialize HSEmotion. Check the internet connection on "
                f"first run and reinstall requirements if needed. Details: {error}"
            ) from error

        self._input_size: tuple[int, int] | None = None

    def detect_faces(self, frame: np.ndarray) -> np.ndarray | None:
        """Return YuNet face rows for a BGR frame."""
        if frame is None or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        size = (width, height)
        if size != self._input_size:
            self.detector.setInputSize(size)
            self._input_size = size
        _, faces = self.detector.detect(frame)
        return faces

    @staticmethod
    def _square_face_crop(
        frame: np.ndarray, face_data: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Make a lightly padded, square BGR crop around the detected face."""
        frame_h, frame_w = frame.shape[:2]
        x, y, width, height = (float(value) for value in face_data[:4])
        if not np.isfinite([x, y, width, height]).all() or width < 2 or height < 2:
            raise ValueError("YuNet returned an invalid face box.")

        side = max(32, int(round(max(width, height) * 1.20)))
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        x0 = int(round(center_x - side / 2.0))
        y0 = int(round(center_y - side / 2.0))
        x1, y1 = x0 + side, y0 + side

        left, top = max(0, x0), max(0, y0)
        right, bottom = min(frame_w, x1), min(frame_h, y1)
        if right <= left or bottom <= top:
            raise ValueError("Detected face box is outside the camera frame.")

        crop = frame[top:bottom, left:right]
        crop = cv2.copyMakeBorder(
            crop,
            max(0, -y0),
            max(0, y1 - frame_h),
            max(0, -x0),
            max(0, x1 - frame_w),
            cv2.BORDER_REPLICATE,
        )

        # YuNet returns both eye landmarks. Correct small head roll so the
        # expression classifier sees a consistently upright face.
        if face_data.size >= 8:
            eyes = np.asarray(face_data[4:8], dtype=np.float32).reshape(2, 2)
            if np.isfinite(eyes).all():
                left_eye, right_eye = eyes[np.argsort(eyes[:, 0])]
                eye_dx = float(right_eye[0] - left_eye[0])
                eye_dy = float(right_eye[1] - left_eye[1])
                if eye_dx > 1.0:
                    angle = float(np.degrees(np.arctan2(eye_dy, eye_dx)))
                    angle = float(np.clip(angle, -25.0, 25.0))
                    if abs(angle) >= 1.0:
                        rotation = cv2.getRotationMatrix2D(
                            (crop.shape[1] / 2.0, crop.shape[0] / 2.0), angle, 1.0
                        )
                        crop = cv2.warpAffine(
                            crop,
                            rotation,
                            (crop.shape[1], crop.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE,
                        )

        bbox = (
            max(0, int(round(x))),
            max(0, int(round(y))),
            max(0, int(round(width))),
            max(0, int(round(height))),
        )
        return crop, bbox

    @staticmethod
    def scores_from_output(output: np.ndarray) -> dict[str, float]:
        """Map HSEmotion's eight probabilities without merging classes."""
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size != len(HSEMOTION_OUTPUT_LABELS):
            raise RuntimeError(
                f"Expected 8 HSEmotion scores, received {values.size}. "
                "Check the installed HSEmotion model."
            )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise RuntimeError("HSEmotion returned invalid class scores.")

        # Contempt and disgust are separate model classes; keep both separate.
        scores = {
            "angry": float(values[0]),
            "contempt": float(values[1]),
            "disgust": float(values[2]),
            "fearful": float(values[3]),
            "happy": float(values[4]),
            "neutral": float(values[5]),
            "sad": float(values[6]),
            "surprised": float(values[7]),
        }
        # Improve recall for disgust when the model's happy score is weak.
        # A strong happiness score is left untouched to protect clear smiles.
        if scores["happy"] < 0.40 and scores["disgust"] >= 0.10:
            scores["disgust"] *= 1.4
            total = sum(scores.values())
            if total > 0.0:
                scores = {label: score / total for label, score in scores.items()}
        return scores

    def analyze_face(
        self, frame: np.ndarray, face_data: np.ndarray
    ) -> tuple[dict[str, float], str, float, tuple[int, int, int, int]]:
        """Return class scores, top expression, its score, and the face rectangle."""
        crop, bbox = self._square_face_crop(frame, face_data)
        # OpenCV frames are BGR; convert the crop to the model's RGB input order.
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        _, output = self.classifier.predict_emotions(crop_rgb, logits=False)
        scores = self.scores_from_output(output)
        top_emotion = max(scores, key=scores.get)
        return scores, top_emotion, scores[top_emotion], bbox


class EmotionSmoother:
    """Smooth sampled predictions and require repeated results for autoplay."""

    def __init__(self, window_size: int = 3, stability_threshold: int = 2) -> None:
        if window_size < 1 or stability_threshold < 1:
            raise ValueError("Smoothing sizes must be positive integers.")
        self.window_size = window_size
        self.stability_threshold = min(stability_threshold, window_size)
        self.history: deque[dict[str, float]] = deque(maxlen=window_size)
        self.dominant_history: deque[str] = deque(maxlen=window_size)
        self.last_stable_emotion: str | None = None

    def update(self, scores: dict[str, float]) -> tuple[dict[str, float], str, float, bool]:
        self.history.append(
            {name: max(0.0, float(scores.get(name, 0.0))) for name in EMOTION_LABELS}
        )

        weights = np.arange(1, len(self.history) + 1, dtype=np.float32)
        weight_total = float(weights.sum())
        smoothed = {
            emotion: float(
                sum(weight * sample[emotion] for weight, sample in zip(weights, self.history))
                / weight_total
            )
            for emotion in EMOTION_LABELS
        }

        top_emotion = max(smoothed, key=smoothed.get)
        confidence = smoothed[top_emotion]
        self.dominant_history.append(top_emotion)
        recent_count = sum(emotion == top_emotion for emotion in self.dominant_history)
        is_stable = (
            len(self.dominant_history) >= self.stability_threshold
            and recent_count >= self.stability_threshold
        )
        if is_stable:
            self.last_stable_emotion = top_emotion
        return smoothed, top_emotion, confidence, is_stable

    def clear(self) -> None:
        self.history.clear()
        self.dominant_history.clear()
        self.last_stable_emotion = None

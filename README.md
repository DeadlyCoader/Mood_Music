# Mood Music

A small Python desktop app that estimates facial-expression cues from a webcam and recommends music. It uses **HSEmotion EfficientNet-B2 ONNX** with OpenCV YuNet face detection. It aligns and pads the face crop, supplies RGB input to the classifier, samples one frame every 0.5 seconds, and smooths scores across recent samples. Disgust gets a modest sensitivity adjustment when happiness is not strong; a strong happy score is left alone. Frames are processed locally in memory and are not saved.

The model estimates visible expression patterns; it cannot know someone's actual mood. Its scores can still be wrong, especially for subtle or mixed expressions. Contempt and disgust remain separate labels because they are separate model outputs. Music recommendations for contempt use the neutral/chill profile.

## Project files

```text
mood-music/
├── models/
│   └── face_detection_yunet_2023mar.onnx
├── camera_manager.py
├── emotion_detector.py
├── main.py
├── mood_music.py
├── music_recommender.py
├── ui_overlay.py
├── web_player.py
├── requirements.txt
└── README.md
```

The YuNet face detector is included in `models`. The HSEmotion classifier downloads its pretrained ONNX weights on first launch and caches them on your computer; the first launch needs an internet connection.

## Install

Use Python 3.10 or newer. Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

## Run

For live webcam use:

```bash
python main.py
```

To check a still photo and print every expression score:

```bash
python main.py --image path/to/face.jpg
```

Use a clear, front-facing photo with good lighting. Hold a strong expression for two or three seconds when using the camera so the app can sample and smooth several frames.

If no camera is available, the app opens **Manual Demo Mode**. That mode previews the music mapping with number keys and does not run expression recognition:

```bash
python main.py --demo --no-browser
```

## Controls

| Key | Action |
|---|---|
| `Space` | Open the top music recommendation in your browser |
| `1`, `2`, `3` | Open recommendation 1, 2, or 3; in Manual Demo Mode these choose happy, sad, or angry |
| `4`–`8` | In Manual Demo Mode, choose surprised, fearful, disgust, neutral, or contempt |
| `A` | Toggle hands-free autoplay; off by default |
| `M` | Cycle YouTube, Spotify, and YouTube Music |
| `P` or `K` | Send a play/pause key to the browser |
| `+` / `-` | Change system volume |
| `D` | Switch to or from Manual Demo Mode |
| `Q` or `Esc` | Quit |

To choose a different camera, run `python main.py --camera 1`. Autoplay can be enabled at startup with `python main.py --autoplay`; it is off by default.

## Improve recognition conditions

- Keep your full face in view and look toward the camera.
- Use even light from the front; avoid strong backlighting and shadows.
- Keep the camera steady and hold each expression for a few seconds.
- Use `--image` to inspect the score breakdown on a clear photo.

The model's classes are expression estimates, not dependable readings of inner feelings. For personalized results, the model would need examples of your expressions and a separate calibration step.

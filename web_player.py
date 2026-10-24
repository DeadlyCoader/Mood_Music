"""Online Web Music Player with PyAutoGUI Automation and Anti-Spam Cooldown.

Handles web playback via default browser (webbrowser) and optional keyboard
automation via pyautogui for seamless hands-free playback.
"""

from __future__ import annotations

import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from typing import Literal

from music_recommender import ProviderType, Track

# Optional PyAutoGUI import with graceful fallback
try:
    import pyautogui
    # Disable pyautogui failsafe pause delays for instant responsiveness
    pyautogui.PAUSE = 0.05
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False


@dataclass
class PlaybackEvent:
    track_title: str
    artist: str
    mood: str
    url: str
    provider: ProviderType
    timestamp: float


class WebMusicPlayer:
    """Manages online music playback via browser and PyAutoGUI automation."""

    PROVIDERS: list[ProviderType] = ["youtube", "spotify", "ytmusic"]

    def __init__(
        self,
        default_provider: ProviderType = "youtube",
        cooldown_seconds: float = 45.0,
        enable_automation: bool = True,
        open_browser: bool = True,
    ) -> None:
        self.current_provider: ProviderType = default_provider
        self.cooldown_seconds = cooldown_seconds
        self.enable_automation = enable_automation and PYAUTOGUI_AVAILABLE
        self.open_browser = open_browser

        self.auto_play: bool = False
        self.last_played: PlaybackEvent | None = None
        self.last_trigger_time: float = 0.0
        self.history: list[PlaybackEvent] = []
        self.status_message: str = "Ready. Press SPACE or Click Play."
        self.status_time: float = time.monotonic()

    def cycle_provider(self) -> ProviderType:
        """Cycles to the next online streaming provider."""
        current_idx = self.PROVIDERS.index(self.current_provider)
        self.current_provider = self.PROVIDERS[(current_idx + 1) % len(self.PROVIDERS)]
        self.set_status(f"Provider switched to: {self.current_provider.upper()}")
        return self.current_provider

    def toggle_auto_play(self) -> bool:
        """Toggles hands-free auto playback mode."""
        self.auto_play = not self.auto_play
        state = "ENABLED" if self.auto_play else "DISABLED"
        self.set_status(f"Auto-Play {state}")
        return self.auto_play

    def is_cooling_down(self) -> bool:
        """Returns True if the auto-play cooldown period is still active."""
        return (time.monotonic() - self.last_trigger_time) < self.cooldown_seconds

    def remaining_cooldown(self) -> float:
        """Returns the number of seconds remaining in the cooldown window."""
        elapsed = time.monotonic() - self.last_trigger_time
        return max(0.0, self.cooldown_seconds - elapsed)

    def can_auto_trigger(self, emotion: str, confidence: float) -> bool:
        """Determines if the detected emotion should trigger an automatic playback."""
        if not self.auto_play:
            return False
        if confidence < 0.50:
            return False
        if self.is_cooling_down():
            return False
        if self.last_played and self.last_played.mood == emotion:
            return False
        return True

    def set_status(self, message: str) -> None:
        self.status_message = message
        self.status_time = time.monotonic()

    def play_track(
        self,
        track: Track,
        mood: str,
        force: bool = False,
    ) -> bool:
        """Launches the track's web URL in the browser and triggers automation."""
        now = time.monotonic()
        if not force and self.is_cooling_down():
            self.set_status(f"Cooldown active: {self.remaining_cooldown():.0f}s remaining.")
            return False

        url = track.get_url(self.current_provider)
        self.last_trigger_time = now
        event = PlaybackEvent(
            track_title=track.title,
            artist=track.artist,
            mood=mood,
            url=url,
            provider=self.current_provider,
            timestamp=now,
        )
        self.last_played = event
        self.history.append(event)

        self.set_status(f"Playing online: {track.title} ({self.current_provider.upper()})")

        if self.open_browser:
            # Open URL in background thread to avoid blocking OpenCV render loop
            threading.Thread(
                target=self._launch_web_playback,
                args=(url, self.enable_automation),
                daemon=True,
            ).start()
        return True

    def play_url(self, url: str, title: str = "Custom Mix", mood: str = "custom") -> None:
        """Directly plays an arbitrary web URL."""
        self.last_trigger_time = time.monotonic()
        event = PlaybackEvent(
            track_title=title,
            artist="Web Stream",
            mood=mood,
            url=url,
            provider=self.current_provider,
            timestamp=self.last_trigger_time,
        )
        self.last_played = event
        self.history.append(event)
        self.set_status(f"Opening: {title}")

        if self.open_browser:
            threading.Thread(
                target=self._launch_web_playback,
                args=(url, self.enable_automation),
                daemon=True,
            ).start()

    def _launch_web_playback(self, url: str, automate: bool) -> None:
        """Worker thread to open browser and execute pyautogui key presses."""
        try:
            webbrowser.open_new_tab(url)
        except Exception as err:
            self.set_status(f"Browser launch error: {err}")
            return

        if automate and PYAUTOGUI_AVAILABLE:
            try:
                # Wait 2.2 seconds for the webpage and video player to initialize
                time.sleep(2.2)
                # Send 'k' (standard YouTube play/pause shortcut) or 'space'
                pyautogui.press("k")
            except Exception:
                # Failsafe ignore if user navigated away or locked screen
                pass

    def toggle_playback_keys(self) -> None:
        """Uses PyAutoGUI to send a play/pause media key or spacebar."""
        if not PYAUTOGUI_AVAILABLE:
            self.set_status("PyAutoGUI not available.")
            return
        try:
            pyautogui.press("space")
            self.set_status("Sent [SPACE] Play/Pause to browser.")
        except Exception as err:
            self.set_status(f"Key error: {err}")

    def volume_up(self) -> None:
        if PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.press("volumeup")
            except Exception:
                pass

    def volume_down(self) -> None:
        if PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.press("volumedown")
            except Exception:
                pass

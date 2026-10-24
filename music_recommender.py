"""Load mood-based music recommendations from a neighboring JSON file."""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ProviderType = Literal["youtube", "spotify", "ytmusic"]
DEFAULT_CATALOG_PATH = Path(__file__).resolve().with_name("mood_catalog.json")


@dataclass(frozen=True)
class Track:
    title: str
    artist: str
    genre: str
    description: str
    youtube_url: str
    spotify_url: str
    ytmusic_url: str
    tags: frozenset[str] = field(default_factory=frozenset)

    def get_url(self, provider: ProviderType = "youtube") -> str:
        if provider == "spotify":
            return self.spotify_url
        if provider == "ytmusic":
            return self.ytmusic_url
        return self.youtube_url


@dataclass(frozen=True)
class MoodProfile:
    mood: str
    display_name: str
    emoji: str
    color_bgr: tuple[int, int, int]
    vibe_summary: str
    default_search_query: str
    tracks: list[Track]


def _required_text(data: dict, key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: '{key}' must be a non-empty string.")
    return value


def load_mood_catalog(
    json_path: str | Path | None = None,
) -> dict[str, MoodProfile]:
    """Read and validate mood profiles from JSON.

    By default, ``mood_catalog.json`` must be beside this Python file. A custom
    path can be supplied when constructing ``MusicRecommender``.
    """
    path = Path(json_path).expanduser() if json_path is not None else DEFAULT_CATALOG_PATH
    try:
        raw_catalog = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Mood catalog not found: {path}. Keep mood_catalog.json beside "
            "music_recommender.py."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path} at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error

    if not isinstance(raw_catalog, dict) or not raw_catalog:
        raise ValueError("The mood catalog JSON must be a non-empty object.")

    catalog: dict[str, MoodProfile] = {}
    for mood_key, raw_profile in raw_catalog.items():
        if not isinstance(mood_key, str) or not mood_key.strip():
            raise ValueError("Every mood in the JSON catalog must have a text key.")
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Mood '{mood_key}' must contain a JSON object.")

        mood = mood_key.strip().lower()
        context = f"Mood '{mood}'"
        raw_color = raw_profile.get("color_bgr")
        if (
            not isinstance(raw_color, list)
            or len(raw_color) != 3
            or any(not isinstance(value, int) or isinstance(value, bool) for value in raw_color)
        ):
            raise ValueError(f"{context}: 'color_bgr' must be a list of three integers.")

        raw_tracks = raw_profile.get("tracks")
        if not isinstance(raw_tracks, list):
            raise ValueError(f"{context}: 'tracks' must be a JSON list.")

        tracks: list[Track] = []
        for index, raw_track in enumerate(raw_tracks, start=1):
            track_context = f"{context}, track {index}"
            if not isinstance(raw_track, dict):
                raise ValueError(f"{track_context} must be a JSON object.")
            raw_tags = raw_track.get("tags", [])
            if not isinstance(raw_tags, list) or any(not isinstance(tag, str) for tag in raw_tags):
                raise ValueError(f"{track_context}: 'tags' must be a list of strings.")
            tracks.append(
                Track(
                    title=_required_text(raw_track, "title", track_context),
                    artist=_required_text(raw_track, "artist", track_context),
                    genre=_required_text(raw_track, "genre", track_context),
                    description=_required_text(raw_track, "description", track_context),
                    youtube_url=_required_text(raw_track, "youtube_url", track_context),
                    spotify_url=_required_text(raw_track, "spotify_url", track_context),
                    ytmusic_url=_required_text(raw_track, "ytmusic_url", track_context),
                    tags=frozenset(raw_tags),
                )
            )

        catalog[mood] = MoodProfile(
            mood=_required_text(raw_profile, "mood", context).strip().lower(),
            display_name=_required_text(raw_profile, "display_name", context),
            emoji=_required_text(raw_profile, "emoji", context),
            color_bgr=tuple(raw_color),
            vibe_summary=_required_text(raw_profile, "vibe_summary", context),
            default_search_query=_required_text(
                raw_profile, "default_search_query", context
            ),
            tracks=tracks,
        )

    if "neutral" not in catalog:
        raise ValueError("The mood catalog JSON must include a 'neutral' profile.")
    return catalog


# Keep the existing module-level name available for code that imports it.
MOOD_CATALOG: dict[str, MoodProfile] = load_mood_catalog()


class MusicRecommender:
    """Select tracks from a JSON-backed mood catalog and build provider searches."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self.catalog = load_mood_catalog(catalog_path) if catalog_path else MOOD_CATALOG

    def get_profile(self, emotion: str) -> MoodProfile:
        emotion_key = emotion.strip().lower()
        return self.catalog.get(emotion_key, self.catalog["neutral"])

    def get_recommendations(self, emotion: str, limit: int = 3) -> list[Track]:
        """Return up to ``limit`` tracks for the detected expression."""
        profile = self.get_profile(emotion)
        return profile.tracks[:limit]

    def build_search_url(
        self, emotion: str, provider: ProviderType = "youtube"
    ) -> str:
        """Build a provider search URL using the mood query stored in JSON."""
        profile = self.get_profile(emotion)
        encoded = urllib.parse.quote_plus(profile.default_search_query)
        if provider == "spotify":
            return f"https://open.spotify.com/search/{encoded}"
        if provider == "ytmusic":
            return f"https://music.youtube.com/search?q={encoded}"
        return f"https://www.youtube.com/results?search_query={encoded}"

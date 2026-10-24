"""Music Recommender Engine for Facial Emotion Recognition.

Maps 7 basic emotions to curated mood profiles with high-quality tracks and playlists
across YouTube, Spotify, and YouTube Music with direct web playback links.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from typing import Literal

ProviderType = Literal["youtube", "spotify", "ytmusic"]


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
        elif provider == "ytmusic":
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


MOOD_CATALOG: dict[str, MoodProfile] = {
    "happy": MoodProfile(
        mood="happy",
        display_name="Joyful & Upbeat",
        emoji="😃",
        color_bgr=(40, 215, 120),  # Emerald Green
        vibe_summary="Infectious, upbeat anthems to amplify positive vibrations and energy.",
        default_search_query="feel good happy upbeat pop dance music",
        tracks=[
            Track(
                title="Happy",
                artist="Pharrell Williams",
                genre="Upbeat Pop / Soul",
                description="Clap along feel-good anthem full of sunny positive energy.",
                youtube_url="https://www.youtube.com/watch?v=ZbZSe6N_BXs",
                spotify_url="https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH",
                ytmusic_url="https://music.youtube.com/watch?v=ZbZSe6N_BXs",
                tags=frozenset({"happy", "upbeat", "pop", "dance", "bright"}),
            ),
            Track(
                title="Can't Stop the Feeling!",
                artist="Justin Timberlake",
                genre="Disco Pop / Funk",
                description="High-tempo sunshine groove guaranteed to get you moving.",
                youtube_url="https://www.youtube.com/watch?v=ru0K8uYEZWw",
                spotify_url="https://open.spotify.com/track/1WkMMJnEmMw0JnuQU96G4K",
                ytmusic_url="https://music.youtube.com/watch?v=ru0K8uYEZWw",
                tags=frozenset({"dance", "happy", "party", "energetic"}),
            ),
            Track(
                title="Uptown Funk",
                artist="Mark Ronson ft. Bruno Mars",
                genre="Funk / Pop",
                description="Iconic brass hooks and vintage swagger with irresistible groove.",
                youtube_url="https://www.youtube.com/watch?v=OPf0YbXqDm0",
                spotify_url="https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS",
                ytmusic_url="https://music.youtube.com/watch?v=OPf0YbXqDm0",
                tags=frozenset({"funk", "happy", "dance", "upbeat"}),
            ),
            Track(
                title="Sugar",
                artist="Maroon 5",
                genre="Pop Rock / Dance",
                description="Sweet, melodic pop confection with sparkling rhythm.",
                youtube_url="https://www.youtube.com/watch?v=09R8_2nJtjg",
                spotify_url="https://open.spotify.com/track/2iuZJX9KnUPgbrXTrPLQ0R",
                ytmusic_url="https://music.youtube.com/watch?v=09R8_2nJtjg",
                tags=frozenset({"pop", "happy", "upbeat"}),
            ),
        ],
    ),
    "sad": MoodProfile(
        mood="sad",
        display_name="Melancholy & Reflective",
        emoji="😢",
        color_bgr=(255, 140, 60),  # Sky Blue
        vibe_summary="Gentle acoustic chords and cathartic melodies for emotional comfort.",
        default_search_query="sad emotional piano comforting acoustic songs",
        tracks=[
            Track(
                title="Someone Like You",
                artist="Adele",
                genre="Emotional Piano Ballad",
                description="Deeply emotive, raw acoustic vocal performance.",
                youtube_url="https://www.youtube.com/watch?v=hLQl3WQQoQ0",
                spotify_url="https://open.spotify.com/track/1k1o9Su249wJkLNaA0z8Z9",
                ytmusic_url="https://music.youtube.com/watch?v=hLQl3WQQoQ0",
                tags=frozenset({"sad", "piano", "reflective", "acoustic", "slow"}),
            ),
            Track(
                title="Fix You",
                artist="Coldplay",
                genre="Alternative Rock / Ballad",
                description="Soaring emotional crescendo offering hope and healing.",
                youtube_url="https://www.youtube.com/watch?v=k4V3Mo61fJM",
                spotify_url="https://open.spotify.com/track/7LVHVU3tWfcxj5aiP2YW45",
                ytmusic_url="https://music.youtube.com/watch?v=k4V3Mo61fJM",
                tags=frozenset({"sad", "reflective", "healing", "slow"}),
            ),
            Track(
                title="Mad World",
                artist="Gary Jules",
                genre="Slow Melancholic Piano",
                description="Intimate and haunting arrangement for quiet contemplation.",
                youtube_url="https://www.youtube.com/watch?v=4N3N1MlvVc4",
                spotify_url="https://open.spotify.com/track/3JOVTQ5hQgMLi2iBhsz2i1",
                ytmusic_url="https://music.youtube.com/watch?v=4N3N1MlvVc4",
                tags=frozenset({"melancholy", "piano", "acoustic", "slow"}),
            ),
            Track(
                title="See You Again",
                artist="Wiz Khalifa ft. Charlie Puth",
                genre="Tribute Ballad / Pop",
                description="Heartfelt tribute melody with uplifting emotional resonance.",
                youtube_url="https://www.youtube.com/watch?v=RgKAFK5djSk",
                spotify_url="https://open.spotify.com/track/2JzZzNuYYfl0ZqNzqq4L6v",
                ytmusic_url="https://music.youtube.com/watch?v=RgKAFK5djSk",
                tags=frozenset({"sad", "reflective", "piano"}),
            ),
        ],
    ),
    "angry": MoodProfile(
        mood="angry",
        display_name="Tension Release & Calming",
        emoji="😡",
        color_bgr=(50, 50, 255),  # Crimson Red
        vibe_summary="Dual option: High-energy rock for catharsis, or calming ambient stress release.",
        default_search_query="deep stress relief tension calming music",
        tracks=[
            Track(
                title="Numb",
                artist="Linkin Park",
                genre="Cathartic Rock Anthem",
                description="Release tension with soaring guitars and raw, honest vocals.",
                youtube_url="https://www.youtube.com/watch?v=kXYiU_JCYtU",
                spotify_url="https://open.spotify.com/track/2nLtzopw4rPResBhoR9Nk5",
                ytmusic_url="https://music.youtube.com/watch?v=kXYiU_JCYtU",
                tags=frozenset({"intense", "rock", "cathartic", "energy"}),
            ),
            Track(
                title="Weightless (Tension Relief)",
                artist="Marconi Union",
                genre="Ambient Therapy / Sound Science",
                description="Scientifically crafted to slow heart rate and eliminate acute stress.",
                youtube_url="https://www.youtube.com/watch?v=UfcAVejslrU",
                spotify_url="https://open.spotify.com/track/6kkwzB6hXLsqDRwh0vFaxq",
                ytmusic_url="https://music.youtube.com/watch?v=UfcAVejslrU",
                tags=frozenset({"calm", "relax", "ambient", "stress-relief"}),
            ),
            Track(
                title="In The End",
                artist="Linkin Park",
                genre="Alternative Rock",
                description="Dynamic rhythms and driving energy to transform frustration into momentum.",
                youtube_url="https://www.youtube.com/watch?v=eVTXPUF4Oz4",
                spotify_url="https://open.spotify.com/track/60a0Rd6pj0xtZHEx2Yik5w",
                ytmusic_url="https://music.youtube.com/watch?v=eVTXPUF4Oz4",
                tags=frozenset({"rock", "energy", "cathartic"}),
            ),
        ],
    ),
    "surprised": MoodProfile(
        mood="surprised",
        display_name="Excited & Dynamic",
        emoji="😲",
        color_bgr=(255, 230, 80),  # Electric Cyan / Gold
        vibe_summary="Electrifying tempo, unexpected drops, and dynamic synth rhythms.",
        default_search_query="high energy electronic dance music hype",
        tracks=[
            Track(
                title="Levels",
                artist="Avicii",
                genre="Euphoric Progressive House",
                description="Legendary electronic anthem with boundless energy and joy.",
                youtube_url="https://www.youtube.com/watch?v=_ovdm2yX4MA",
                spotify_url="https://open.spotify.com/track/5UqCQwD7z4iY3qMmYl5R5Y",
                ytmusic_url="https://music.youtube.com/watch?v=_ovdm2yX4MA",
                tags=frozenset({"edm", "exciting", "electronic", "dance"}),
            ),
            Track(
                title="Blinding Lights",
                artist="The Weeknd",
                genre="Synthwave / Retro Pop",
                description="Fast-pulsing 80s synthesizers and high-octane drive.",
                youtube_url="https://www.youtube.com/watch?v=4NRXx6U8ABQ",
                spotify_url="https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b",
                ytmusic_url="https://music.youtube.com/watch?v=4NRXx6U8ABQ",
                tags=frozenset({"synthwave", "fast", "energetic", "pop"}),
            ),
            Track(
                title="Eye of the Tiger",
                artist="Survivor",
                genre="Classic Arena Rock",
                description="Instant hype riff that turns surprise into unstoppable focus.",
                youtube_url="https://www.youtube.com/watch?v=btPJPFnesV4",
                spotify_url="https://open.spotify.com/track/2KH16WveTQWT6KOG9Rg6e2",
                ytmusic_url="https://music.youtube.com/watch?v=btPJPFnesV4",
                tags=frozenset({"rock", "hype", "energy"}),
            ),
        ],
    ),
    "fearful": MoodProfile(
        mood="fearful",
        display_name="Grounding & Reassuring",
        emoji="😨",
        color_bgr=(80, 140, 255),  # Warm Amber / Coral
        vibe_summary="Harmonic sound baths, 432Hz ambient tones, and soft acoustic grounding.",
        default_search_query="432Hz deep anxiety relief calming meditation music",
        tracks=[
            Track(
                title="Weightless (Anxiety Relief)",
                artist="Marconi Union",
                genre="Therapeutic Soundscape",
                description="Decreases cortisol levels and calms the autonomic nervous system.",
                youtube_url="https://www.youtube.com/watch?v=UfcAVejslrU",
                spotify_url="https://open.spotify.com/track/6kkwzB6hXLsqDRwh0vFaxq",
                ytmusic_url="https://music.youtube.com/watch?v=UfcAVejslrU",
                tags=frozenset({"calm", "soothing", "ambient", "healing"}),
            ),
            Track(
                title="Deep 432Hz Calming Frequencies",
                artist="Meditative Mind",
                genre="Binaural Ambient / Meditation",
                description="Resonant healing frequencies for immediate relaxation and safety.",
                youtube_url="https://www.youtube.com/watch?v=1ZYbU82GVz4",
                spotify_url="https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO",
                ytmusic_url="https://music.youtube.com/watch?v=1ZYbU82GVz4",
                tags=frozenset({"meditation", "grounding", "chill"}),
            ),
            Track(
                title="Spiegel im Spiegel",
                artist="Arvo Pärt",
                genre="Minimalist Piano & Violin",
                description="Gentle, slow, orderly chords creating an oasis of stillness.",
                youtube_url="https://www.youtube.com/watch?v=TJ6Mzvh3XCc",
                spotify_url="https://open.spotify.com/track/25qBwJbJbOa45a6Wz1p3yA",
                ytmusic_url="https://music.youtube.com/watch?v=TJ6Mzvh3XCc",
                tags=frozenset({"classical", "peaceful", "soft"}),
            ),
        ],
    ),
    "disgust": MoodProfile(
        mood="disgust",
        display_name="Refreshing & Cleansing",
        emoji="🤢",
        color_bgr=(180, 80, 180),  # Soft Purple / Lavender
        vibe_summary="Brisk, crisp acoustic melodies and uplifting nature-infused sounds.",
        default_search_query="refreshing clean acoustic morning guitar instrumental",
        tracks=[
            Track(
                title="Spring (The Four Seasons)",
                artist="Antonio Vivaldi",
                genre="Classical Baroque Revival",
                description="Brisk, refreshing strings evoking clean air and new beginnings.",
                youtube_url="https://www.youtube.com/watch?v=mFWQgxXM_b8",
                spotify_url="https://open.spotify.com/track/598hpXp5o5gGk97k9s6wP1",
                ytmusic_url="https://music.youtube.com/watch?v=mFWQgxXM_b8",
                tags=frozenset({"classical", "refreshing", "bright"}),
            ),
            Track(
                title="Morning Acoustic Coffee Vibes",
                artist="Acoustic Morning Ensemble",
                genre="Clean Fingerstyle Acoustic",
                description="Crisp wooden acoustic resonance that resets your sensory palette.",
                youtube_url="https://www.youtube.com/watch?v=d-diB65scQU",
                spotify_url="https://open.spotify.com/playlist/37i9dQZF1DX4E3UdUs7fUx",
                ytmusic_url="https://music.youtube.com/watch?v=d-diB65scQU",
                tags=frozenset({"acoustic", "clean", "calm", "morning"}),
            ),
        ],
    ),
    "neutral": MoodProfile(
        mood="neutral",
        display_name="Chill & Focused",
        emoji="😌",
        color_bgr=(200, 200, 200),  # Silver / Slate
        vibe_summary="Smooth Lo-Fi hip hop, relaxing study beats, and atmospheric ambient chill.",
        default_search_query="lofi hip hop radio beats to relax study to",
        tracks=[
            Track(
                title="Lofi Girl: Beats to Relax/Study To",
                artist="Lofi Records",
                genre="Lo-Fi Hip Hop / Chillhop",
                description="24/7 world-famous relaxing beats for reading, coding, and chilling.",
                youtube_url="https://www.youtube.com/watch?v=jfKfPfyJRdk",
                spotify_url="https://open.spotify.com/playlist/0vvXsWCC9xrXsKd4FyS8kM",
                ytmusic_url="https://music.youtube.com/watch?v=jfKfPfyJRdk",
                tags=frozenset({"lofi", "chill", "focus", "study", "ambient"}),
            ),
            Track(
                title="Synthwave / Chillwave Radio",
                artist="Lofi Girl / ChilledCow",
                genre="Chillwave / Downtempo",
                description="Spacious analog synths and mellow dreamscapes.",
                youtube_url="https://www.youtube.com/watch?v=4xDzrJKXOOY",
                spotify_url="https://open.spotify.com/playlist/37i9dQZF1DXdLEN7aqioXM",
                ytmusic_url="https://music.youtube.com/watch?v=4xDzrJKXOOY",
                tags=frozenset({"synthwave", "chill", "relax", "instrumental"}),
            ),
            Track(
                title="Peaceful Guitar & Soft Piano",
                artist="Ambient Chill Collective",
                genre="Acoustic Ambient",
                description="Warm, unobtrusive melodies designed to enhance concentration.",
                youtube_url="https://www.youtube.com/watch?v=WPni755-Krg",
                spotify_url="https://open.spotify.com/playlist/37i9dQZF1DX8Uebhn9wzrS",
                ytmusic_url="https://music.youtube.com/watch?v=WPni755-Krg",
                tags=frozenset({"calm", "ambient", "focus", "piano"}),
            ),
        ],
    ),
}


class MusicRecommender:
    """Manages mood-based song recommendations and web search query generation."""

    def __init__(self) -> None:
        self.catalog = MOOD_CATALOG

    def get_profile(self, emotion: str) -> MoodProfile:
        emotion_key = emotion.strip().lower()
        return self.catalog.get(emotion_key, self.catalog["neutral"])

    def get_recommendations(
        self, emotion: str, limit: int = 3
    ) -> list[Track]:
        """Returns top tracks for the given emotion."""
        profile = self.get_profile(emotion)
        return profile.tracks[:limit]

    def build_search_url(
        self, emotion: str, provider: ProviderType = "youtube"
    ) -> str:
        """Constructs an online web search URL for the detected mood."""
        profile = self.get_profile(emotion)
        encoded = urllib.parse.quote_plus(profile.default_search_query)
        if provider == "spotify":
            return f"https://open.spotify.com/search/{encoded}"
        elif provider == "ytmusic":
            return f"https://music.youtube.com/search?q={encoded}"
        return f"https://www.youtube.com/results?search_query={encoded}"

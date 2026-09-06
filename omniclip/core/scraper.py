"""Link ingestion.

Takes any video URL and returns its metadata plus a timed transcript, which is
everything the script brain needs to rebuild an equivalent video: what was said,
when it was said, and how long the result has to run.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_ID_PATTERNS = (
    re.compile(r"(?:v=|/shorts/|/embed/|youtu\.be/)([0-9A-Za-z_-]{11})"),
    re.compile(r"^([0-9A-Za-z_-]{11})$"),
)


class IngestError(RuntimeError):
    """Raised when a source video cannot be read well enough to clone."""


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class SourceVideo:
    url: str
    video_id: str | None
    title: str
    description: str
    uploader: str
    duration_seconds: float
    tags: list[str] = field(default_factory=list)
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None
    auto_generated: bool = True

    @property
    def transcript(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def word_count(self) -> int:
        return len(self.transcript.split())

    @property
    def spoken_wpm(self) -> float:
        """Speaking rate of the original, used to match its pacing."""
        minutes = self.duration_seconds / 60
        return self.word_count / minutes if minutes > 0 else 0.0

    def target_word_count(self, wpm: float | None = None) -> int:
        """How many words the rewritten script needs to fill the same runtime."""
        rate = wpm or self.spoken_wpm or 145
        return int((self.duration_seconds / 60) * rate)


def extract_video_id(url: str) -> str | None:
    for pattern in _ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


# YouTube blocks yt-dlp's default web client on a growing share of videos,
# reporting them as simply "not available". Other player clients still serve
# the same video, so each is tried in turn before giving up.
PLAYER_CLIENTS = (None, "android", "ios", "tv")


def ydl_options(client: str | None, **extra) -> dict:
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "noprogress": True,
        **extra,
    }
    if client:
        options["extractor_args"] = {"youtube": {"player_client": [client]}}

    # yt-dlp needs ffmpeg to merge streams or cut a partial download, and ours
    # ships inside the virtualenv rather than on PATH.
    try:
        from ..utils.media import ffmpeg_dir

        options.setdefault("ffmpeg_location", ffmpeg_dir())
    except Exception:
        pass
    return options


def fetch_metadata(url: str) -> dict:
    """Pull title/duration/tags without downloading the media itself."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise IngestError("yt-dlp is not installed; run pip install -r requirements.txt") from exc

    errors = []
    for client in PLAYER_CLIENTS:
        try:
            with yt_dlp.YoutubeDL(ydl_options(client, skip_download=True)) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            errors.append(f"{client or 'default'}: {str(exc).splitlines()[0]}")
            continue

        if info.get("_type") == "playlist" and info.get("entries"):
            info = info["entries"][0]
        return info

    raise IngestError(f"Could not read metadata for {url}:\n  " + "\n  ".join(errors))


def _fetch_via_transcript_api(
    video_id: str, languages: list[str]
) -> tuple[list[dict], str | None, bool]:
    """youtube-transcript-api changed its interface in 1.0; support both."""
    from youtube_transcript_api import YouTubeTranscriptApi

    if hasattr(YouTubeTranscriptApi, "fetch"):
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=languages)
        return fetched.to_raw_data(), fetched.language_code, fetched.is_generated

    listing = YouTubeTranscriptApi.list_transcripts(video_id)
    try:
        transcript = listing.find_manually_created_transcript(languages)
    except Exception:
        transcript = listing.find_transcript(languages)
    return transcript.fetch(), transcript.language_code, transcript.is_generated


def _fetch_via_ytdlp_captions(
    info: dict, languages: list[str]
) -> tuple[list[dict], str | None, bool]:
    """Fallback path: read the caption tracks yt-dlp already discovered.

    Needed because youtube-transcript-api is frequently blocked from datacenter
    and VPN addresses, while yt-dlp's caption URLs usually still resolve.
    """
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    for source, generated in ((manual, False), (auto, True)):
        for lang in [*languages, *source.keys()]:
            tracks = source.get(lang)
            if not tracks:
                continue
            track = next(
                (t for t in tracks if t.get("ext") == "json3"),
                tracks[0],
            )
            if track.get("ext") != "json3":
                continue
            try:
                req = urllib.request.Request(
                    track["url"],
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    payload = json.load(response)
            except Exception:
                continue
            segments = []
            for event in payload.get("events", []):
                text = "".join(seg.get("utf8", "") for seg in event.get("segs", []))
                if not text.strip():
                    continue
                segments.append(
                    {
                        "text": text.strip(),
                        "start": event.get("tStartMs", 0) / 1000,
                        "duration": event.get("dDurationMs", 0) / 1000,
                    }
                )
            if segments:
                return segments, lang, generated
    return [], None, True


def fetch_transcript(
    video_id: str | None, info: dict, languages: list[str]
) -> tuple[list[TranscriptSegment], str | None, bool]:
    raw: list[dict] = []
    language: str | None = None
    generated = True

    if video_id:
        try:
            raw, language, generated = _fetch_via_transcript_api(video_id, languages)
        except Exception:
            raw = []

    if not raw:
        raw, language, generated = _fetch_via_ytdlp_captions(info, languages)

    segments = [
        TranscriptSegment(
            text=item["text"],
            start=float(item.get("start", 0.0)),
            duration=float(item.get("duration", 0.0)),
        )
        for item in raw
        if item.get("text", "").strip()
    ]
    return segments, language, generated


def scrape(
    url: str,
    languages: list[str] | None = None,
    transcriber=None,
) -> SourceVideo:
    """Resolve a link into a SourceVideo ready for the script brain.

    `transcriber` is an optional callback used when the video carries no
    captions at all, which is common for Shorts. See core.transcriber.
    """
    languages = languages or ["en"]
    info = fetch_metadata(url)
    video_id = info.get("id") or extract_video_id(url)
    segments, language, generated = fetch_transcript(video_id, info, languages)

    if not segments and transcriber:
        segments, language, generated = transcriber(url, info)

    if not segments:
        raise IngestError(
            f"No transcript or captions available for {url}, and no "
            "transcriber was supplied to fall back on."
        )

    return SourceVideo(
        url=url,
        video_id=video_id,
        title=info.get("title") or "Untitled",
        description=info.get("description") or "",
        uploader=info.get("uploader") or info.get("channel") or "",
        duration_seconds=float(info.get("duration") or segments[-1].end),
        tags=list(info.get("tags") or []),
        segments=segments,
        language=language,
        auto_generated=generated,
    )

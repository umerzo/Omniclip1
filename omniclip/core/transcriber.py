"""Audio transcription fallback.

Plenty of videos — Shorts especially — ship no captions at all. When that
happens the audio is downloaded and transcribed with Whisper, so the pipeline
still gets a timed transcript to work from.

Whisper runs remotely through the same OpenAI-compatible endpoint used for
scripting, which keeps it fast and avoids a local model download.
"""

from __future__ import annotations

from pathlib import Path

from ..utils.media import probe_duration, run
from .scraper import PLAYER_CLIENTS, TranscriptSegment, ydl_options

# Remote Whisper endpoints cap upload size; 24 MB keeps a margin under 25.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

# Speech stays perfectly legible at 32 kbps mono, which fits roughly 100
# minutes inside the upload cap. Anything longer is split.
SPEECH_BITRATE = "32k"
SEGMENT_SECONDS = 600


class TranscribeError(RuntimeError):
    """Raised when audio could not be fetched or transcribed."""


def download_audio(url: str, out_dir: str | Path) -> Path:
    """Grab the audio track only — no need for video we are going to discard."""
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    for client in PLAYER_CLIENTS:
        options = ydl_options(
            client,
            format="bestaudio/best",
            outtmpl=str(out_dir / "%(id)s.%(ext)s"),
        )
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                if info.get("_type") == "playlist" and info.get("entries"):
                    info = info["entries"][0]
                path = Path(ydl.prepare_filename(info))
        except Exception as exc:
            errors.append(f"{client or 'default'}: {str(exc).splitlines()[0]}")
            continue

        if not path.exists():
            matches = list(out_dir.glob(f"{info.get('id', '*')}.*"))
            if not matches:
                errors.append(f"{client or 'default'}: no file produced")
                continue
            path = matches[0]
        return path

    raise TranscribeError(f"Audio download failed for {url}:\n  " + "\n  ".join(errors))


def transcribe_file(
    audio_path: str | Path,
    api_key: str,
    base_url: str,
    model: str = "whisper-large-v3",
    translate: bool = False,
) -> tuple[list[TranscriptSegment], str | None]:
    """Transcribe audio to timed segments.

    With `translate`, Whisper's translation endpoint is used instead, which
    returns English regardless of the language spoken in the source.
    """
    from openai import OpenAI

    audio_path = Path(audio_path)
    size = audio_path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise TranscribeError(
            f"{audio_path.name} is {size/1e6:.0f} MB, over the "
            f"{MAX_UPLOAD_BYTES/1e6:.0f} MB upload limit. Split it first."
        )

    client = OpenAI(base_url=base_url, api_key=api_key)
    with audio_path.open("rb") as handle:
        payload = handle.read()

    if translate:
        response = client.audio.translations.create(
            file=(audio_path.name, payload),
            model=model,
            response_format="verbose_json",
        )
    else:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, payload),
            model=model,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    raw = getattr(response, "segments", None) or []
    segments = []
    for item in raw:
        get = item.get if isinstance(item, dict) else lambda k, d=None: getattr(item, k, d)
        text = (get("text") or "").strip()
        if not text:
            continue
        start = float(get("start") or 0.0)
        end = float(get("end") or start)
        segments.append(TranscriptSegment(text=text, start=start, duration=max(0.0, end - start)))

    if not segments:
        raise TranscribeError(f"Whisper returned no segments for {audio_path.name}")
    return segments, getattr(response, "language", None)


def compress_speech(source: str | Path, dest: str | Path) -> Path:
    """Re-encode to 32 kbps mono opus — small enough that most sources fit."""
    source, dest = Path(source), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["-i", str(source), "-vn", "-ac", "1", "-c:a", "libopus",
         "-b:a", SPEECH_BITRATE, str(dest)], label="compress speech")
    return dest


def split_audio(
    source: str | Path,
    out_dir: str | Path,
    segment_seconds: int = SEGMENT_SECONDS,
) -> list[tuple[Path, float]]:
    """Cut audio into chunks, each paired with its offset in the original.

    The offsets come from measuring the pieces rather than assuming they are
    exactly `segment_seconds` long — the splitter cuts on packet boundaries, so
    assuming would slowly drift the transcript out of sync on a long file.
    """
    source, out_dir = Path(source), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part_*.ogg"):
        stale.unlink()

    run(["-i", str(source), "-f", "segment",
         "-segment_time", str(segment_seconds), "-c", "copy",
         str(out_dir / "part_%03d.ogg")], label="split audio")

    parts, offset = [], 0.0
    for path in sorted(out_dir.glob("part_*.ogg")):
        parts.append((path, offset))
        offset += probe_duration(path)
    if not parts:
        raise TranscribeError(f"Splitting produced no parts for {source.name}")
    return parts


def transcribe_long(
    audio_path: str | Path,
    api_key: str,
    base_url: str,
    model: str = "whisper-large-v3",
    translate: bool = False,
    progress=None,
) -> tuple[list[TranscriptSegment], str | None]:
    """Transcribe audio of any length, compressing and splitting as needed."""
    audio_path = Path(audio_path)
    work = audio_path.parent / f"{audio_path.stem}_speech"

    small = compress_speech(audio_path, work.with_suffix(".ogg"))
    if small.stat().st_size <= MAX_UPLOAD_BYTES:
        if progress:
            progress(1, 1)
        return transcribe_file(small, api_key, base_url, model, translate)

    parts = split_audio(small, work)
    segments: list[TranscriptSegment] = []
    language = None

    for index, (part, offset) in enumerate(parts, start=1):
        if progress:
            progress(index, len(parts))
        part_segments, part_language = transcribe_file(
            part, api_key, base_url, model, translate
        )
        language = language or part_language
        segments.extend(
            TranscriptSegment(s.text, s.start + offset, s.duration)
            for s in part_segments
        )

    return segments, language


def make_transcriber(api_key: str, base_url: str, cache_dir: str | Path,
                     model: str = "whisper-large-v3", translate: bool = False,
                     progress=None):
    """Build the callback the scraper uses when a video has no captions."""

    def transcriber(url: str, info: dict):
        audio = download_audio(url, cache_dir)
        segments, language = transcribe_long(
            audio, api_key, base_url, model, translate, progress
        )
        return segments, language, True

    return transcriber

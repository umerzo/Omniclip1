"""Vocalist.

Synthesises narration with edge-tts and captures word-level timings as it goes.

edge-tts emits a WordBoundary event per spoken word, so the subtitle timings
come free with the audio — no forced alignment pass needed. faster-whisper is
only required when narration comes from somewhere else.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import re
from dataclasses import dataclass
from pathlib import Path

# edge-tts reports offsets in 100-nanosecond ticks.
_TICKS_PER_SECOND = 10_000_000


@functools.lru_cache(maxsize=1)
def _boundary_kwargs() -> dict:
    """edge-tts 7.x defaults to SentenceBoundary, which is useless for captions.

    Older releases have no `boundary` parameter and emit word events anyway, so
    the argument is only passed when the installed version accepts it.
    """
    import edge_tts

    params = inspect.signature(edge_tts.Communicate.__init__).parameters
    return {"boundary": "WordBoundary"} if "boundary" in params else {}


class SpeechError(RuntimeError):
    """Raised when narration could not be synthesised."""


@dataclass(frozen=True)
class WordTiming:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def _restore_punctuation(words: list[WordTiming], source: str) -> list[WordTiming]:
    """Put punctuation back onto the timed words.

    WordBoundary events carry the bare spoken word, so "home." arrives as
    "home". Captions need the punctuation to break sentences sensibly, and it is
    recoverable by walking the timed words against the text that was submitted.
    """
    tokens = source.split()
    keys = [re.sub(r"[^\w']", "", t).casefold() for t in tokens]
    restored: list[WordTiming] = []
    cursor = 0

    for word in words:
        key = re.sub(r"[^\w']", "", word.text).casefold()
        match = next(
            (i for i in range(cursor, min(cursor + 4, len(tokens))) if keys[i] == key),
            None,
        )
        if match is None:
            restored.append(word)
            continue
        restored.append(WordTiming(tokens[match], word.start, word.duration))
        cursor = match + 1

    return restored


@dataclass
class SpeechResult:
    audio_path: Path
    words: list[WordTiming]
    offset: float = 0.0

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0

    def shifted_words(self) -> list[WordTiming]:
        """Word timings rebased onto the full timeline."""
        return [
            WordTiming(w.text, w.start + self.offset, w.duration) for w in self.words
        ]


async def _synthesize_async(
    text: str,
    out_path: Path,
    voice: str,
    rate: str,
    volume: str,
) -> SpeechResult:
    import edge_tts

    communicate = edge_tts.Communicate(
        text, voice, rate=rate, volume=volume, **_boundary_kwargs()
    )
    words: list[WordTiming] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append(
                    WordTiming(
                        text=chunk["text"],
                        start=chunk["offset"] / _TICKS_PER_SECOND,
                        duration=chunk["duration"] / _TICKS_PER_SECOND,
                    )
                )

    if not words:
        raise SpeechError(f"edge-tts returned no word timings for {out_path.name}")
    return SpeechResult(audio_path=out_path, words=_restore_punctuation(words, text))


def synthesize(
    text: str,
    out_path: str | Path,
    voice: str = "en-US-AndrewNeural",
    rate: str = "+0%",
    volume: str = "+0%",
) -> SpeechResult:
    """Render one block of narration to mp3 and return its word timings."""
    if not text.strip():
        raise SpeechError("Refusing to synthesise empty narration")
    return asyncio.run(_synthesize_async(text, Path(out_path), voice, rate, volume))


def _pad_to(path: Path, seconds: float) -> Path:
    """Extend a track with trailing silence so it fills its slot."""
    from ..utils.media import run

    padded = path.with_name(f"{path.stem}_padded.mp3")
    run(["-i", str(path), "-af", f"apad=whole_dur={seconds:.3f}",
         "-t", f"{seconds:.3f}", "-c:a", "libmp3lame", str(padded)],
        label="pad narration")
    padded.replace(path)
    return path


def _silence(path: Path, seconds: float) -> Path:
    """A silent track of a given length, for a shot with no narration."""
    from ..utils.media import run

    path.parent.mkdir(parents=True, exist_ok=True)
    run(["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
         "-t", f"{max(0.1, seconds):.3f}", "-c:a", "libmp3lame", str(path)],
        label="silent track")
    return path


def list_voices(language: str | None = None) -> list[dict]:
    """Available edge-tts voices, optionally filtered by language prefix."""
    import edge_tts

    voices = asyncio.run(edge_tts.list_voices())
    if language:
        voices = [v for v in voices if v["ShortName"].lower().startswith(language.lower())]
    return sorted(voices, key=lambda v: v["ShortName"])


def narrate_plan(
    plan,
    out_dir: str | Path,
    voice: str = "en-US-AndrewNeural",
    rate: str = "+0%",
    volume: str = "+0%",
    progress=None,
    preserve_durations: bool = False,
) -> list[SpeechResult]:
    """Narrate every scene and set the timeline from the result.

    By default a scene lasts exactly as long as its narration. With
    `preserve_durations` the scene keeps the length it already had — the length
    of the source shot it was built from — and short narration is padded with
    silence to fill it. Without that, a shot given only a few words collapses to
    a one-second scene and the rebuilt video stops cutting like the original.
    """
    from ..utils.media import MediaError, probe_duration

    out_dir = Path(out_dir)
    results: list[SpeechResult] = []
    cursor = 0.0

    for scene in plan.scenes:
        if progress:
            progress(scene.index + 1, len(plan.scenes))

        path = out_dir / f"scene_{scene.index:04d}.mp3"
        if not scene.narration.strip():
            # A shot with nothing to say still has to occupy its slot, or the
            # audio and video tracks drift apart from here on.
            result = SpeechResult(audio_path=_silence(path, scene.duration), words=[])
            result.offset = cursor
            scene.start = cursor
            cursor += scene.duration
            results.append(result)
            continue

        result = synthesize(
            scene.narration,
            path,
            voice=voice,
            rate=rate,
            volume=volume,
        )

        # The last word ends before the file does; using the word time would
        # accumulate a gap on every scene and drift captions out of sync.
        try:
            length = probe_duration(result.audio_path)
        except MediaError:
            length = result.duration

        if preserve_durations and scene.duration > length + 0.05:
            # Hold the source shot's length; speech cannot be stretched, so the
            # remainder becomes silence rather than a shortened cut.
            _pad_to(result.audio_path, scene.duration)
            length = scene.duration

        result.offset = cursor
        scene.start = cursor
        scene.duration = length
        cursor += length
        results.append(result)

    return results

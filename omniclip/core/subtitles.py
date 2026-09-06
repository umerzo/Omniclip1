"""Subtitle aligner.

Turns word-level timings into caption files. SRT for CapCut and generic
players; ASS when the animated word-by-word highlight is wanted, which is the
caption style the reference videos use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .tts_engine import WordTiming


@dataclass
class Cue:
    words: list[WordTiming] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def group_words(
    words: list[WordTiming],
    max_words: int = 5,
    max_duration: float = 3.0,
    max_chars: int = 30,
    pause_split: float = 0.6,
) -> list[Cue]:
    """Break a word stream into readable caption cues.

    A cue ends when it gets too long to read, or when the speaker pauses —
    pauses are where a human editor would cut, so they make the cleanest breaks.
    """
    cues: list[Cue] = []
    current = Cue()

    for word in words:
        if current.words:
            gap = word.start - current.words[-1].end
            too_long = (
                len(current.words) >= max_words
                or len(current.text) + len(word.text) + 1 > max_chars
                or word.end - current.start > max_duration
            )
            if too_long or gap >= pause_split:
                cues.append(current)
                current = Cue()
        current.words.append(word)

        # A finished sentence is always a clean place to cut.
        if word.text.rstrip("\"')]}").endswith((".", "!", "?", "…")):
            cues.append(current)
            current = Cue()

    if current.words:
        cues.append(current)
    return cues


def even_words(text: str, start: float, duration: float) -> list[WordTiming]:
    """Spread a line evenly across a scene.

    Used when the captions are not in the spoken language: an English word has
    no position inside an Urdu sentence, so real word timings do not exist.
    Distributing them across the scene keeps the highlight moving in time with
    the delivery without pretending to a precision that is not there.
    """
    words = [w for w in text.split() if w]
    if not words or duration <= 0:
        return []
    share = duration / len(words)
    return [
        WordTiming(word, start + index * share, share)
        for index, word in enumerate(words)
    ]


def _srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _ass_timestamp(seconds: float) -> str:
    cs = int(round(seconds * 100))
    hours, cs = divmod(cs, 360_000)
    minutes, cs = divmod(cs, 6_000)
    secs, cs = divmod(cs, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def write_srt(cues: list[Cue], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, cue in enumerate(cues, start=1):
        lines.append(str(index))
        lines.append(f"{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}")
        lines.append(cue.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_ASS_HEADER_STANDARD = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: OmniClip,{font},{size},&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_ASS_HEADER_KINETIC = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: OmniClip,{font},{size},&H00FFFFFF,&H0000E5FF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,{outline},3,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


SIDE_MARGIN = 60


def caption_metrics(width: int, height: int, style: str = "kinetic") -> dict:
    """Type size and line length tailored to frame and subtitle style."""
    if style == "kinetic":
        # Viral pop-in: bold, larger text, placed closer to optical center
        size = max(36, round(min(width, height) / 12))
        outline = max(3, round(size / 12))
        margin_v = round(height * (0.28 if height > width else 0.12))
        max_chars = 18
    else:
        # Standard: classic bottom captions
        size = max(28, round(min(width, height) / 15))
        outline = max(2, round(size / 18))
        margin_v = round(height * (0.15 if height > width else 0.08))
        usable = width - 2 * SIDE_MARGIN
        max_chars = max(16, int(usable / (size * 0.5) * 0.9))
    return {
        "size": size,
        "outline": outline,
        "margin_v": margin_v,
        "max_chars": max_chars,
    }


def write_ass(
    cues: list[Cue],
    path: str | Path,
    width: int = 1080,
    height: int = 1920,
    font: str = "Arial",
    size: int | None = None,
    outline: int | None = None,
    margin_v: int | None = None,
    style: str = "kinetic",
) -> Path:
    """Write ASS subtitle file supporting standard or kinetic pop-in animations."""
    metrics = caption_metrics(width, height, style=style)
    size = size or metrics["size"]
    outline = outline or metrics["outline"]
    margin_v = margin_v if margin_v is not None else metrics["margin_v"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header_template = _ASS_HEADER_KINETIC if style == "kinetic" else _ASS_HEADER_STANDARD
    lines = [
        header_template.format(
            width=width, height=height, font=font,
            size=size, outline=outline, margin_v=margin_v,
        )
    ]
    for cue in cues:
        parts = []
        for word in cue.words:
            centiseconds = max(1, int(round(word.duration * 100)))
            tag = f"{{\\kf{centiseconds}}}" if style == "kinetic" else f"{{\\k{centiseconds}}}"
            parts.append(f"{tag}{word.text}")

        # In kinetic mode, add subtle scale pop-in on entrance
        anim_prefix = "{\\t(0,60,\\fscx110\\fscy110)\\t(60,120,\\fscx100\\fscy100)}" if style == "kinetic" else ""
        text_content = f"{anim_prefix}{' '.join(parts)}"

        lines.append(
            f"Dialogue: 0,{_ass_timestamp(cue.start)},{_ass_timestamp(cue.end)},"
            f"OmniClip,,0,0,0,,{text_content}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_from_lines(lines, out_dir, basename="captions",
                     width=1080, height=1920, font="Arial", style: str = "kinetic", **grouping):
    """Captions from per-scene lines rather than from spoken word timings."""
    metrics = caption_metrics(width, height, style=style)
    grouping.setdefault("max_chars", metrics["max_chars"])
    if style == "kinetic":
        grouping.setdefault("max_words", 3)
        grouping.setdefault("max_duration", 1.8)

    cues: list[Cue] = []
    for text, start, duration in lines:
        timed = even_words(text, start, duration)
        if timed:
            cues.extend(group_words(timed, **grouping))

    out_dir = Path(out_dir)
    return {
        "srt": write_srt(cues, out_dir / f"{basename}.srt"),
        "ass": write_ass(cues, out_dir / f"{basename}.ass", width, height, font, style=style),
        "cues": cues,
    }


def build_subtitles(
    results,
    out_dir: str | Path,
    basename: str = "captions",
    width: int = 1080,
    height: int = 1920,
    font: str = "Arial",
    style: str = "kinetic",
    **grouping,
) -> dict[str, Path]:
    """Write SRT + ASS for narration run, respecting standard vs kinetic pop-in style."""
    metrics = caption_metrics(width, height, style=style)
    grouping.setdefault("max_chars", metrics["max_chars"])
    if style == "kinetic":
        grouping.setdefault("max_words", 3)
        grouping.setdefault("max_duration", 1.8)

    cues: list[Cue] = []
    for result in results:
        cues.extend(group_words(result.shifted_words(), **grouping))

    out_dir = Path(out_dir)
    return {
        "srt": write_srt(cues, out_dir / f"{basename}.srt"),
        "ass": write_ass(cues, out_dir / f"{basename}.ass", width, height, font, style=style),
        "cues": cues,
    }

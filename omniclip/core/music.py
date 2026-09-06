"""Background music, generated to match the donor's mood.

Generating rather than sourcing avoids the licensing question entirely: the
track is original, so there is nothing to attribute and nothing to clear. The
model returns roughly half-minute pieces, so a longer video is filled by
crossfading several together.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import httpx

from ..utils.media import probe_duration, run

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class MusicError(RuntimeError):
    """Raised when a music bed could not be produced."""


# Appended to every brief, whatever its source. A bed with singing in it reads
# as a second narrator talking over the first, which is what it sounded like.
NO_VOCALS = ("purely instrumental, no vocals, no singing, no voice, no words, "
             "no choir, no spoken audio")


def instrumental_brief(mood: str) -> str:
    """A musical brief that cannot come back with words in it.

    Applied here rather than trusted to the wording of each brief, because the
    briefs come from several places -- a table of defaults, a model that writes
    one per video, and callers -- and every one of them has to be instrumental.
    """
    mood = (mood or "").strip().rstrip(".")
    if not mood:
        return NO_VOCALS
    return f"{mood}. {NO_VOCALS}"


def generate_clip(prompt: str, api_key: str, dest: Path,
                  model: str = "google/lyria-3-clip-preview",
                  timeout: float = 300.0) -> Path:
    """One generated piece, streamed and written to disk.

    The audio only arrives when an explicit output format is requested and the
    response is streamed; without both the model replies with a text marker and
    no sound at all.
    """
    body = {
        "model": model,
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    chunks: list[str] = []
    with httpx.stream(
        "POST", ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=body, timeout=timeout,
    ) as response:
        if response.status_code >= 400:
            raise MusicError(f"{response.status_code}: {response.read()[:200]}")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in parsed.get("choices") or []:
                audio = (choice.get("delta") or {}).get("audio")
                if isinstance(audio, dict) and audio.get("data"):
                    chunks.append(audio["data"])
                elif isinstance(audio, str) and audio:
                    chunks.append(audio)

    if not chunks:
        raise MusicError("no audio returned")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode("".join(chunks)))
    return dest


def build_bed(prompt: str, api_key: str, seconds: float, work_dir: Path,
              model: str = "google/lyria-3-clip-preview",
              crossfade: float = 2.0, progress=None,
              library: Path | None = None, pieces_wanted: int = 3) -> Path:
    """A music bed long enough to sit under the whole video.

    Generation is billed per clip and each clip is only about half a minute, so
    filling a ten minute video from scratch costs real money every time. A small
    number of pieces per mood is generated once into a shared library and reused
    across videos; longer beds repeat that set rather than buying more. Under
    narration at low volume the repetition is not noticeable, and the cost stops
    scaling with runtime.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    store = Path(library) if library else work_dir
    store.mkdir(parents=True, exist_ok=True)
    # Whatever brief arrived, the request that goes out is instrumental.
    prompt = instrumental_brief(prompt)
    mood_id = hashlib.sha1(prompt.encode()).hexdigest()[:12]

    # Only ever buy enough pieces to avoid an obvious short loop.
    needed = min(pieces_wanted, max(1, int(seconds // 25) + 1))
    pieces: list[Path] = []
    for number in range(1, needed + 1):
        piece = store / f"{mood_id}_{number:02d}.wav"
        if not piece.exists():
            try:
                generate_clip(prompt, api_key, piece, model)
            except MusicError:
                break
        try:
            probe_duration(piece)
        except Exception:
            break
        pieces.append(piece)
        if progress:
            progress(len(pieces), needed)

    if not pieces:
        raise MusicError("could not generate any music")

    bed = work_dir / "bed.m4a"

    # One passage built from the available pieces, then repeated to length.
    # Looping is what decouples cost from runtime: a ten minute video costs the
    # same few clips as a one minute one.
    if len(pieces) == 1:
        passage = pieces[0]
    else:
        passage = store / f"{mood_id}_passage.m4a"
        if not passage.exists():
            inputs: list[str] = []
            for piece in pieces:
                inputs += ["-i", str(piece)]
            filters = []
            label = "0:a"
            for i in range(1, len(pieces)):
                out = f"x{i}"
                filters.append(
                    f"[{label}][{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri[{out}]"
                )
                label = out
            run([*inputs, "-filter_complex", ";".join(filters),
                 "-map", f"[{label}]", "-c:a", "aac", "-b:a", "160k",
                 str(passage)], label="music passage")

    run(["-stream_loop", "-1", "-i", str(passage), "-t", f"{seconds:.3f}",
         "-c:a", "aac", "-b:a", "160k", str(bed)], label="music bed")
    return bed


def mix_under(narration: Path, bed: Path, dest: Path,
              music_db: float = -20.0) -> Path:
    """Lay the music under the voice, quiet enough to stay out of its way."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["-i", str(narration), "-i", str(bed),
         "-filter_complex",
         f"[1:a]volume={music_db}dB[m];[0:a][m]amix=inputs=2:duration=first:"
         f"dropout_transition=0:normalize=0[a]",
         "-map", "[a]", "-c:a", "aac", "-b:a", "192k", str(dest)],
        label="mix music")
    return dest

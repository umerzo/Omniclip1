"""ffmpeg helpers.

The binary ships with imageio-ffmpeg inside the virtualenv, so nothing has to be
installed system-wide. A system ffmpeg on PATH is used instead when present.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path

_DURATION = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")


class MediaError(RuntimeError):
    """Raised when an ffmpeg operation fails."""


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - dependency guard
        raise MediaError(
            "ffmpeg not found. Install it, or pip install imageio-ffmpeg."
        ) from exc


@functools.lru_cache(maxsize=1)
def ffmpeg_dir() -> str:
    """A directory containing a binary literally named ffmpeg.

    yt-dlp searches its ffmpeg_location for "ffmpeg.exe" by name, but
    imageio-ffmpeg ships a versioned filename like
    ffmpeg-win-x86_64-v7.1.exe, so pointing straight at that folder fails.
    A correctly named copy is made once and reused.
    """
    exe = Path(ffmpeg_exe())
    if exe.stem.lower() == "ffmpeg":
        return str(exe.parent)

    from .config import PROJECT_ROOT

    target_dir = PROJECT_ROOT / "assets" / "cache" / "bin"
    target = target_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, target)
    return str(target_dir)


def run(args: list[str], label: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Run ffmpeg, raising with the tail of its log if it fails."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        creationflags=creationflags,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise MediaError(f"{label} failed:\n  " + "\n  ".join(tail))
    return proc


def probe_duration(path: str | Path) -> float:
    """Real duration of a media file in seconds."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        creationflags=creationflags,
    )
    match = _DURATION.search(proc.stderr or "")
    if not match:
        raise MediaError(f"Could not read duration of {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def escape_filter_path(path: str | Path) -> str:
    """Quote a Windows path so ffmpeg's filter parser accepts it."""
    text = str(Path(path).resolve()).replace("\\", "/")
    return text.replace(":", r"\:")

"""Stock video providers for Pexels and Pixabay."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import httpx

from .base import VideoGenerationProvider, VisualAsset


class RateLimiter:
    """Spaces calls evenly so requests never trip provider thresholds."""

    def __init__(self, requests_per_minute: float):
        self._interval = 60.0 / max(requests_per_minute, 0.1)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = max(now, self._next) + self._interval


def _cache_name(source: str, *parts, suffix: str = ".mp4") -> str:
    material = ":".join(str(p) for p in (source, *parts))
    digest = hashlib.sha1(material.encode()).hexdigest()[:16]
    return f"{source}_{digest}{suffix}"


class PexelsProvider(VideoGenerationProvider):
    """Stock footage from Pexels API."""

    name = "pexels"
    SEARCH_URL = "https://api.pexels.com/videos/search"

    def __init__(self, api_key: str, portrait: bool = True, rpm: float = 60):
        self.api_key = api_key
        self.portrait = portrait
        self.limiter = RateLimiter(rpm)

    def available(self) -> bool:
        return bool(self.api_key)

    def _pick_file(self, video: dict) -> dict | None:
        files = [f for f in video.get("video_files", []) if f.get("link")]
        if not files:
            return None
        wanted = [
            f for f in files
            if (f.get("height") or 0) >= (1280 if self.portrait else 720)
            and ((f.get("height") or 0) > (f.get("width") or 0)) == self.portrait
        ]
        pool = wanted or files
        return min(pool, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))

    def fetch(
        self,
        query: str,
        min_duration: float,
        cache_dir: Path,
        variant: int = 0,
        **kwargs,
    ) -> VisualAsset | None:
        if not self.api_key:
            return None
        self.limiter.wait()

        response = httpx.get(
            self.SEARCH_URL,
            headers={"Authorization": self.api_key},
            params={
                "query": query,
                "orientation": "portrait" if self.portrait else "landscape",
                "per_page": 10,
            },
            timeout=30,
        )
        response.raise_for_status()
        videos = response.json().get("videos") or []
        if not videos:
            return None

        long_enough = [v for v in videos if (v.get("duration") or 0) >= min_duration]
        pool = long_enough or videos
        video = pool[variant % len(pool)]
        chosen = self._pick_file(video)
        if not chosen:
            return None

        path = cache_dir / _cache_name(self.name, video["id"])
        if not path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            with httpx.stream("GET", chosen["link"], timeout=120, follow_redirects=True) as stream:
                stream.raise_for_status()
                with path.open("wb") as handle:
                    for block in stream.iter_bytes(65536):
                        handle.write(block)

        return VisualAsset(
            path=path,
            source=self.name,
            query=query,
            width=chosen.get("width") or 0,
            height=chosen.get("height") or 0,
            duration=float(video.get("duration") or 0),
        )


class PixabayProvider(VideoGenerationProvider):
    """Stock footage from Pixabay API."""

    name = "pixabay"
    SEARCH_URL = "https://pixabay.com/api/videos/"

    def __init__(self, api_key: str, portrait: bool = True, rpm: float = 80):
        self.api_key = api_key
        self.portrait = portrait
        self.limiter = RateLimiter(rpm)

    def available(self) -> bool:
        return bool(self.api_key)

    def _pick_file(self, hit: dict) -> dict | None:
        files = [
            dict(v, quality=name)
            for name, v in (hit.get("videos") or {}).items()
            if v.get("url")
        ]
        if not files:
            return None
        wanted = [f for f in files if (f.get("height") or 0) >= 1080]
        pool = wanted or files
        return min(pool, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))

    def fetch(
        self,
        query: str,
        min_duration: float,
        cache_dir: Path,
        variant: int = 0,
        **kwargs,
    ) -> VisualAsset | None:
        if not self.api_key:
            return None
        self.limiter.wait()

        response = httpx.get(
            self.SEARCH_URL,
            params={"key": self.api_key, "q": query, "per_page": 10},
            timeout=30,
        )
        response.raise_for_status()
        hits = response.json().get("hits") or []
        if not hits:
            return None

        long_enough = [h for h in hits if (h.get("duration") or 0) >= min_duration]
        pool = long_enough or hits
        hit = pool[variant % len(pool)]
        chosen = self._pick_file(hit)
        if not chosen:
            return None

        path = cache_dir / _cache_name(self.name, hit["id"])
        if not path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            with httpx.stream("GET", chosen["url"], timeout=180, follow_redirects=True) as stream:
                stream.raise_for_status()
                with path.open("wb") as handle:
                    for block in stream.iter_bytes(65536):
                        handle.write(block)

        return VisualAsset(
            path=path,
            source=self.name,
            query=query,
            width=chosen.get("width") or 0,
            height=chosen.get("height") or 0,
            duration=float(hit.get("duration") or 0),
        )

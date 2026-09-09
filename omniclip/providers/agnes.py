"""Agnes AI video generation provider with 25-key pool and rate limiting."""

from __future__ import annotations

import base64
import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .base import QuotaExhausted, VideoGenerationProvider, VisualAsset, VisualError

PERMANENT_QUOTA_MARKERS = (
    "insufficient_user_quota", "用户额度不足", "insufficient quota",
    "account suspended", "invalid api key", "balance exhausted",
)


def is_permanent_quota(status_code: int, text: str) -> bool:
    if status_code in (401, 403):
        return True
    lowered = (text or "").lower()
    return any(marker in lowered for marker in PERMANENT_QUOTA_MARKERS)


def is_quota_message(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in PERMANENT_QUOTA_MARKERS)


def _cache_name(source: str, *parts, suffix: str = ".mp4") -> str:
    material = ":".join(str(p) for p in (source, *parts))
    digest = hashlib.sha1(material.encode()).hexdigest()[:16]
    return f"{source}_{digest}{suffix}"


@dataclass
class KeyState:
    key_id: str
    key_value: str
    status: str = "available"  # "available" | "busy" | "cooldown" | "failed"
    requests: int = 0
    last_used: float = 0.0
    cooldown_until: float = 0.0
    failures: int = 0


class AgnesKeyPool:
    """Manages pool of Agnes API keys with least-recently-used allocation and cooldowns."""

    def __init__(self, keys: list[str], cooldown: float = 62.0):
        self.keys_map: dict[str, KeyState] = {}
        for i, k in enumerate(keys):
            if k:
                kid = f"key_{i + 1}"
                self.keys_map[kid] = KeyState(key_id=kid, key_value=k)
        self.cooldown = cooldown
        self._lock = threading.Lock()
        self._retired: dict[str, str] = {}

    @property
    def total(self) -> int:
        return len(self.keys_map)

    def live_count(self) -> int:
        with self._lock:
            return sum(1 for kid, s in self.keys_map.items() if kid not in self._retired)

    def retire(self, key_val: str, reason: str) -> int:
        with self._lock:
            for kid, s in self.keys_map.items():
                if s.key_value == key_val:
                    s.status = "failed"
                    self._retired[kid] = reason
            return sum(1 for kid in self.keys_map if kid not in self._retired)

    def acquire(self, timeout: float = 900.0) -> str:
        effective_timeout = max(timeout, len(self.keys_map) * 80.0)
        deadline = time.monotonic() + effective_timeout
        while True:
            with self._lock:
                live = [s for kid, s in self.keys_map.items() if kid not in self._retired]
                if not live:
                    raise QuotaExhausted("All Agnes API generation keys are exhausted.")
                now = time.monotonic()
                # Pick least recently used key that is ready
                candidate = min(live, key=lambda s: s.cooldown_until)
                if candidate.cooldown_until <= now:
                    candidate.status = "busy"
                    candidate.last_used = now
                    candidate.cooldown_until = now + self.cooldown
                    candidate.requests += 1
                    return candidate.key_value
                wait = candidate.cooldown_until - now
            if time.monotonic() + wait > deadline:
                raise VisualError(f"Agnes: no key free within {effective_timeout:.0f}s")
            time.sleep(min(wait, 5.0))

    def peek_ready(self) -> str:
        with self._lock:
            live = [s for kid, s in self.keys_map.items() if kid not in self._retired]
            if not live:
                raise QuotaExhausted("All Agnes API generation keys are exhausted.")
            self._peek_idx = (getattr(self, "_peek_idx", 0) + 1) % len(live)
            return live[self._peek_idx].key_value


class GlobalLimiter:
    """One submission gate shared by all threads to respect machine-level ceiling."""

    def __init__(self, requests_per_minute: float):
        self._interval = 60.0 / max(requests_per_minute, 0.01)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next:
                    self._next = now + self._interval
                    return
                delay = self._next - now
            time.sleep(min(delay, 5.0))

    def throttle(self, delay_seconds: float = 65.0) -> None:
        """Global circuit breaker: pause all workers on an IP-level 429 throttle."""
        with self._lock:
            now = time.monotonic()
            self._next = max(self._next, now + delay_seconds)


class AgnesProvider(VideoGenerationProvider):
    """Primary AI-generated video and still clip provider via Agnes API."""

    name = "agnes"
    BASE_URL = "https://apihub.agnes-ai.com/v1"
    POLL_URL = "https://apihub.agnes-ai.com/agnesapi"
    DEFAULT_MODEL = "agnes-video-v2.0"
    FALLBACK_MODEL = "agnes-video-2.5-flash"
    IMAGE_MODEL = "agnes-image-2.1-flash"
    MAX_CLIP_SECONDS = 20
    FRAME_RATE = 24
    MAX_PROMPT_CHARS = 1500

    def __init__(
        self,
        api_key: str | list[str] = "",
        model: str = "agnes-video-v2.0",
        width: int = 768,
        height: int = 1152,
        rpm: float = 1.0,
        pool_rpm: float = 1.0,
        poll_interval: float = 15.0,
        timeout: float = 900.0,
    ):
        keys = [api_key] if isinstance(api_key, str) else list(api_key)
        self.pool = AgnesKeyPool(keys, cooldown=62.0)
        self.gate = GlobalLimiter(pool_rpm)
        self.model = model
        self.width = width
        self.height = height
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.references: dict[int, str] = {}

    def available(self) -> bool:
        if self.pool.live_count() == 0:
            return False
        try:
            key = self.pool.peek_ready()
            resp = httpx.get(f"{self.BASE_URL}/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def _frames_for(self, seconds: float) -> int:
        seconds = min(self.MAX_CLIP_SECONDS, max(2.0, seconds))
        raw = seconds * self.FRAME_RATE
        frames = max(9, int(round((raw - 1) / 8)) * 8 + 1)
        ceiling = self.MAX_CLIP_SECONDS * self.FRAME_RATE
        while frames > ceiling:
            frames -= 8
        return max(9, frames)

    def _fit_prompt(self, prompt: str) -> str:
        prompt = " ".join(prompt.split())
        return prompt[: self.MAX_PROMPT_CHARS]

    def make_still(
        self,
        prompt: str,
        cache_dir: Path,
        variant: int = 0,
        size: str = "1024x1024",
    ) -> tuple[str, Path] | None:
        """Generate anchor still image using agnes-image-2.1-flash."""
        if self.pool.live_count() == 0:
            return None
        fitted = self._fit_prompt(prompt)
        path = cache_dir / _cache_name("still", fitted, variant, suffix=".jpg")
        if path.exists():
            return "", path

        cache_dir.mkdir(parents=True, exist_ok=True)
        response = None
        for _ in range(self.pool.total):
            key = self.pool.peek_ready()
            try:
                response = httpx.post(
                    f"{self.BASE_URL}/images/generations",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": self.IMAGE_MODEL, "n": 1, "size": size, "prompt": fitted},
                    timeout=120,
                )
                if response.status_code < 400:
                    break
                if response.status_code == 429:
                    time.sleep(3.0)
                    continue
                if is_permanent_quota(response.status_code, response.text):
                    self.pool.retire(key, f"Still image quota {response.status_code}")
                    continue
            except Exception:
                continue

        if response is None or response.status_code >= 400:
            return None

        try:
            data = response.json().get("data") or []
            if not data:
                return None
            item = data[0]
            url = item.get("url") or ""
            b64 = item.get("b64_json") or ""
            if b64:
                path.write_bytes(base64.b64decode(b64))
            elif url:
                with httpx.stream("GET", url, timeout=60) as s:
                    s.raise_for_status()
                    with path.open("wb") as h:
                        for b in s.iter_bytes(65536):
                            h.write(b)
            return url, path
        except Exception:
            return None

    def fetch(
        self,
        query: str,
        min_duration: float,
        cache_dir: Path,
        variant: int = 0,
        **kwargs,
    ) -> VisualAsset | None:
        """Generate video clip from prompt or reference still."""
        fitted = self._fit_prompt(query)
        active_model = self.model
        path = cache_dir / _cache_name(self.name, active_model, fitted, variant, suffix=".mp4")
        if path.exists():
            return VisualAsset(path=path, source=self.name, query=query)

        cache_dir.mkdir(parents=True, exist_ok=True)
        reference_url = self.references.get(variant)
        still_path = kwargs.get("still_path")

        frames = self._frames_for(min_duration)

        if str(active_model).startswith("agnes-video-2.5"):
            is_portrait = self.height > self.width
            aspect = "9:16" if is_portrait else "16:9"
            valid_http_ref = bool(reference_url and (reference_url.startswith("http://") or reference_url.startswith("https://")))
            payload: dict[str, Any] = {
                "model": active_model,
                "prompt": f"{fitted}, sharp focus, pristine details, cinematic smooth motion",
                "mode": "reference" if valid_http_ref else "text",
                "size": "720P",
                "aspect_ratio": aspect,
                "seconds": str(int(min(12, max(4, round(min_duration))))),
            }
            if valid_http_ref:
                payload["images"] = [reference_url]
        else:
            w = max(64, (self.width // 64) * 64)
            h = max(64, (self.height // 64) * 64)
            payload = {
                "model": active_model,
                "prompt": fitted,
                "width": w,
                "height": h,
                "num_frames": frames,
            }
            if reference_url:
                payload["image_url"] = reference_url
            elif still_path and Path(still_path).exists():
                b64_data = base64.b64encode(Path(still_path).read_bytes()).decode()
                payload["image_url"] = f"data:image/jpeg;base64,{b64_data}"

        self.gate.wait()
        key = self.pool.acquire()

        try:
            resp = httpx.post(
                f"{self.BASE_URL}/videos",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=90,
            )
            if resp.status_code >= 400:
                if resp.status_code == 429:
                    self.gate.throttle(65.0)
                    raise VisualError("Agnes rate limited (429), will retry.")
                if is_permanent_quota(resp.status_code, resp.text):
                    self.pool.retire(key, f"Video quota {resp.status_code}")
                    raise QuotaExhausted("Agnes quota reached on key.")
                raise VisualError(f"Agnes video submission failed: {resp.status_code}")

            task_id = resp.json().get("video_id") or resp.json().get("id")
            if not task_id:
                raise VisualError("No video task ID returned")

            # Poll for completion
            download_url = self._poll_task(task_id, key, active_model)
            with httpx.stream("GET", download_url, timeout=180) as stream:
                stream.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in stream.iter_bytes(65536):
                        handle.write(chunk)

            return VisualAsset(
                path=path,
                source=self.name,
                query=query,
                width=self.width,
                height=self.height,
                duration=min_duration,
            )
        except Exception as exc:
            if isinstance(exc, (QuotaExhausted, VisualError)):
                raise
            raise VisualError(f"Agnes generation error: {exc}") from exc

    def _poll_task(self, task_id: str, key: str, model_name: str = "") -> str:
        deadline = time.monotonic() + self.timeout
        params = {"video_id": task_id}
        if model_name:
            params["model_name"] = model_name
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            try:
                resp = httpx.get(
                    self.POLL_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    params=params,
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    status = (data.get("status") or "").lower()
                    if status in ("completed", "succeeded", "done"):
                        url = data.get("url") or data.get("video_url")
                        if url:
                            return url
                    elif status in ("failed", "error"):
                        raise VisualError(f"Agnes generation task failed: {data.get('error')}")
            except Exception as e:
                if isinstance(e, VisualError):
                    raise
        raise VisualError(f"Agnes task timed out after {self.timeout}s")

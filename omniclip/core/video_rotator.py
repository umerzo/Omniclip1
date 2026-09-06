"""Visual sourcing.

Finds a clip for every scene. Providers are tried in order, so a failure or an
exhausted quota falls through to the next one instead of stopping the run.

  PexelsProvider  - stock footage, 25k requests/month, returns immediately
  AgnesProvider   - AI-generated clips from the Agnes cloud API
                    (apihub.agnes-ai.com), roughly 100s per clip

Two separate limits govern Agnes, and both must be respected:

  per key   a key cannot be reused immediately (KeyRing's cooldown)
  per pool  the service accepts roughly ONE video submission per minute in
            total from one machine, however many keys are presented
            (GlobalLimiter)

The pool limit is the binding one and was measured, not documented: six unused
keys fired together yielded two acceptances. Pacing only per key lets a burst
through, most of it is rejected, and scenes fail after exhausting retries.
"""

from __future__ import annotations

import base64
import hashlib
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..utils.media import run as ffmpeg_run


class VisualError(RuntimeError):
    """Raised when no provider could supply a clip."""


class QuotaExhausted(VisualError):
    """The account cannot generate right now, and waiting will not change it.

    Distinct from a rate limit because the remedy is different: a throttle
    clears in seconds, a spent allowance clears when the plan resets. Retrying
    the second on the timer built for the first is how a run spends an hour
    failing quietly.
    """


# A rejection naming the plan or the allowance is about entitlement, not pace.
QUOTA_MARKERS = (
    "free user", "upgrade", "token plan", "quota", "insufficient",
    "billing", "credit", "subscribe", "plan to unlock",
)


def is_quota_message(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in QUOTA_MARKERS)


@dataclass
class VisualAsset:
    path: Path
    source: str
    query: str
    width: int = 0
    height: int = 0
    duration: float = 0.0

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width


class RateLimiter:
    """Spaces calls evenly so a burst never trips the provider's limit."""

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


def _variant(scene) -> int:
    """Cache-busting id for a scene: its position, plus any repair attempts."""
    return int(getattr(scene, "variant", 0) or getattr(scene, "index", 0))


def _cache_name(source: str, *parts, suffix: str = ".mp4") -> str:
    """Cache filename for a clip.

    Every input that changes the resulting file has to be in the key. The prompt
    alone is not enough: the same prompt at a different length is a different
    clip, and two scenes that happen to share a description need two distinct
    clips or the video visibly repeats itself.
    """
    material = ":".join(str(p) for p in (source, *parts))
    digest = hashlib.sha1(material.encode()).hexdigest()[:16]
    return f"{source}_{digest}{suffix}"


class PexelsProvider:
    """Stock footage. Free key, 25,000 requests per month."""

    name = "pexels"
    SEARCH_URL = "https://api.pexels.com/videos/search"

    def __init__(self, api_key: str, portrait: bool = True, rpm: float = 60):
        self.api_key = api_key
        self.portrait = portrait
        self.limiter = RateLimiter(rpm)

    def _pick_file(self, video: dict) -> dict | None:
        """Prefer the smallest file that still meets the target resolution."""
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

    def fetch(self, query: str, min_duration: float, cache_dir: Path,
              variant: int = 0) -> VisualAsset | None:
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

        # Long enough to cover the scene without looping, if such a clip exists.
        long_enough = [v for v in videos if (v.get("duration") or 0) >= min_duration]
        pool = long_enough or videos
        # Scenes sharing a query walk down the result list instead of all
        # landing on the same clip.
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


class PixabayProvider:
    """Stock footage, second opinion to Pexels.

    Two stock libraries matter because a query that returns nothing usable in
    one often has something in the other, and the chain falls through without
    failing the scene.
    """

    name = "pixabay"
    SEARCH_URL = "https://pixabay.com/api/videos/"

    def __init__(self, api_key: str, portrait: bool = True, rpm: float = 80):
        self.api_key = api_key
        self.portrait = portrait
        self.limiter = RateLimiter(rpm)

    def _pick_file(self, hit: dict) -> dict | None:
        """Smallest rendition that still covers the output frame."""
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

    def fetch(self, query: str, min_duration: float, cache_dir: Path,
              variant: int = 0) -> VisualAsset | None:
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
            with httpx.stream("GET", chosen["url"], timeout=180,
                              follow_redirects=True) as stream:
                stream.raise_for_status()
                with path.open("wb") as handle:
                    for block in stream.iter_bytes(65536):
                        handle.write(block)

        return VisualAsset(
            path=path, source=self.name, query=query,
            width=chosen.get("width") or 0, height=chosen.get("height") or 0,
            duration=float(hit.get("duration") or 0),
        )


class GlobalLimiter:
    """One submission gate shared by every worker and every key.

    KeyRing spaces requests per key, which assumes the quota is per key. It is
    not: measured from one machine, Agnes accepts roughly one video submission
    per minute in total no matter how many keys are presented. With many
    workers the per-key cooldown therefore lets a burst through, most of it is
    rejected, and scenes fail after exhausting their retries. This gate is what
    actually matches the observed limit.
    """

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


class KeyRing:
    """Hands out Agnes keys subject to a per-key cooldown.

    Agnes limits video generation to one request per minute *per key* — the
    quota is not pooled. Round-robin alone therefore fails as soon as there are
    more workers than keys, because a key comes back around before its minute
    is up. Each key is instead held back until it is genuinely free, which makes
    throughput exactly (number of keys) requests per minute.

    Keys are separate allowances, not views onto one account -- agnes_keys()
    collects AGNES_API_KEY_2, _3, ... precisely because more keys mean more
    quota. So a key whose plan is spent is retired from the rotation and the
    others carry on; the ring is finished only once every key has gone. An
    earlier version treated the first spent key as the end of the run, which
    threw away every other allowance along with it.
    """

    def __init__(self, keys: list[str], cooldown: float = 62.0):
        self.keys = [k for k in keys if k]
        self.cooldown = cooldown
        self._lock = threading.Lock()
        self._free_at: dict[str, float] = {k: 0.0 for k in self.keys}
        # key -> why it was retired. Kept for the message shown when the last
        # one goes, so the run says what actually stopped it.
        self._retired: dict[str, str] = {}

    @property
    def total(self) -> int:
        """Every key handed in, spent or not."""
        return len(self.keys)

    def _live(self) -> list[str]:
        """Keys still worth trying. Callers must hold the lock."""
        return [k for k in self.keys if k not in self._retired]

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._live())

    def __len__(self) -> int:
        with self._lock:
            return len(self._live())

    def spent_message(self) -> str:
        reasons = "; ".join(dict.fromkeys(self._retired.values()))
        return (f"all {self.total} generation keys are out of capacity "
                f"({reasons}). Waiting will not clear it; the plans have to "
                f"reset or be upgraded.")

    def retire(self, key: str, reason: str) -> int:
        """Take a spent key out of the rotation; returns how many are left."""
        with self._lock:
            if key in self._free_at:
                self._retired.setdefault(key, reason)
            return len(self._live())

    def acquire(self, timeout: float = 900.0) -> str:
        """Block until a live key is off cooldown, then reserve it."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                live = self._live()
                if not live:
                    raise QuotaExhausted(self.spent_message())
                now = time.monotonic()
                key = min(live, key=lambda k: self._free_at[k])
                ready = self._free_at[key]
                if ready <= now:
                    self._free_at[key] = now + self.cooldown
                    return key
                wait = ready - now
            if time.monotonic() + wait > deadline:
                raise VisualError(
                    f"agnes: no key free within {timeout:.0f}s "
                    f"({len(live)} live keys, {self.cooldown:.0f}s cooldown each)"
                )
            time.sleep(min(wait, 5.0))

    def peek_ready(self) -> str:
        """A live key for a non-generating call, ignoring cooldown."""
        with self._lock:
            live = self._live()
            if not live:
                raise QuotaExhausted(self.spent_message())
            return live[0]


class AgnesProvider:
    """AI-generated clips from the Agnes cloud API.

    Submit to POST /v1/videos, then poll /agnesapi until the job reports
    completed and hands back a download URL. Generation takes roughly 90-120
    seconds per clip, so scenes are best fetched concurrently.
    """

    name = "agnes"
    BASE_URL = "https://apihub.agnes-ai.com/v1"
    POLL_URL = "https://apihub.agnes-ai.com/agnesapi"
    # The model that has always been available. A refused choice falls back to
    # this rather than failing the scene, because a model the account may not
    # use is a different problem from an account that cannot generate.
    FALLBACK_MODEL = "agnes-video-v2.0"
    MAX_CLIP_SECONDS = 20
    FRAME_RATE = 24
    # Verified by submitting increasing lengths: 1500 characters is accepted.
    # (The max_length=256 echoed in request_params refers to another field, not
    # the prompt — truncating to it silently threw away scene detail.)
    MAX_PROMPT_CHARS = 1500
    # The service applies a generic negative prompt of its own; this replaces it
    # with one aimed at what actually went wrong: the same character rendered
    # several times in a frame, and melted faces.
    NEGATIVE_PROMPT = (
        "duplicate person, cloned figures, the same character repeated, "
        "twins, multiple heads, extra limbs, extra arms, extra legs, "
        "deformed face, distorted face, melted features, disfigured, "
        "mutated hands, blurry, text, watermark, signature"
    )

    def __init__(
        self,
        api_key: str | list[str] = "",
        model: str = "agnes-video-v2.0",
        width: int = 768,
        height: int = 1152,
        rpm: float = 15,
        poll_interval: float = 15.0,
        timeout: float = 900.0,
        submit_attempts: int = 8,
        pool_rpm: float = 1.0,
    ):
        keys = [api_key] if isinstance(api_key, str) else list(api_key)
        # Two gates, because there are two limits. The ring stops one key being
        # reused too soon; the pool gate keeps total submissions inside what
        # the service actually accepts.
        self.keys = KeyRing(keys, cooldown=60.0 / max(rpm, 0.1) + 2.0)
        self.gate = GlobalLimiter(pool_rpm)
        self.model = model
        self.width = width
        self.height = height
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.submit_attempts = submit_attempts
        # Tripped only once the ring has no live keys left, so the remaining
        # workers stop immediately instead of each queueing behind a dead pool.
        # One key going is not this: that key is retired and the rest carry on.
        self._exhausted = threading.Event()
        self._exhausted_detail = ""
        self._model_lock = threading.Lock()
        # scene variant -> still URL, filled in by the pipeline so each clip is
        # animated from its own image instead of generated from text alone.
        self.references: dict[int, str] = {}

    def _acquire_key(self) -> str:
        """A live key, or the end of the run once every one is spent."""
        if self._exhausted.is_set():
            raise QuotaExhausted(self._exhausted_detail)
        try:
            return self.keys.acquire()
        except QuotaExhausted as exc:
            self._exhausted_detail = str(exc)
            self._exhausted.set()
            raise

    def _peek_key(self) -> str:
        """A live key without reserving it, for calls the cooldown does not
        govern (stills, the models list). Same end-of-run handling as
        _acquire_key.
        """
        if self._exhausted.is_set():
            raise QuotaExhausted(self._exhausted_detail)
        try:
            return self.keys.peek_ready()
        except QuotaExhausted as exc:
            self._exhausted_detail = str(exc)
            self._exhausted.set()
            raise

    def _retire_key(self, key: str, detail: str) -> None:
        """Drop a key whose plan is spent, and say how many are left.

        Only when the last one goes is the run itself out of capacity; until
        then this is a smaller event than it looks, and the scene is retried on
        the next key rather than lost.
        """
        left = self.keys.retire(key, detail)
        with self._model_lock:
            print(f"   key spent, {left}/{self.keys.total} left "
                  f"({detail[:70]})", flush=True)
        if not left:
            self._exhausted_detail = self.keys.spent_message()
            self._exhausted.set()

    def available(self) -> bool:
        if not self.keys:
            return False
        try:
            response = httpx.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.keys.peek_ready()}"},
                timeout=15,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _frames_for(self, seconds: float) -> int:
        """The API requires num_frames == 8n + 1, and never past the clip cap.

        Rounding to the nearest multiple can land just over the ceiling, so the
        count is floored once it reaches the cap.
        """
        seconds = min(self.MAX_CLIP_SECONDS, max(2.0, seconds))
        raw = seconds * self.FRAME_RATE
        frames = max(9, int(round((raw - 1) / 8)) * 8 + 1)
        ceiling = self.MAX_CLIP_SECONDS * self.FRAME_RATE
        while frames > ceiling:
            frames -= 8
        return max(9, frames)

    IMAGE_MODEL = "agnes-image-2.1-flash"

    def make_still(self, prompt: str, cache_dir: Path, variant: int = 0,
                   size: str = "1024x1024") -> tuple[str, Path] | None:
        """Generate one still for a scene and cache it locally.

        Stills are the anchor for the clip that follows: animating a fixed
        image keeps a character looking like the same person, which independent
        text-to-video calls never do. They are also seconds rather than minutes
        to make, so a bad one is cheap to notice and replace.
        """
        if not self.keys:
            return None
        path = cache_dir / _cache_name("still", self._fit_prompt(prompt), variant,
                                       suffix=".jpg")
        if path.exists():
            return "", path

        # No negative_prompt here: the image queue rejects the field outright
        # ("negative_prompt is not supported by text image queue"), unlike video
        # generation. Everything it would have excluded -- duplicate figures,
        # lettering -- has to be handled in the prompt, which Panel.image_prompt
        # does.
        #
        # Stills run before any clip does, so this is usually where a spent key
        # is discovered first. Retiring it here means the clip stage starts with
        # a ring that already knows which keys are dead, instead of rediscovering
        # each one a minute at a time.
        response = None
        for _ in range(self.keys.total):
            # peek, not acquire: the image queue is not what the per-key video
            # cooldown paces, and reserving here would starve the clip stage.
            key = self._peek_key()
            response = httpx.post(
                f"{self.BASE_URL}/images/generations",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": self.IMAGE_MODEL, "n": 1, "size": size,
                      "prompt": self._fit_prompt(prompt)},
                timeout=180,
            )
            if response.status_code < 400:
                break
            if is_quota_message(response.text):
                self._retire_key(
                    key, f"{response.status_code}: {response.text[:160]}")
                continue
            raise VisualError(
                f"agnes image: {response.status_code} {response.text[:160]}"
            )
        if response is None or response.status_code >= 400:
            raise QuotaExhausted(self._exhausted_detail or self.keys.spent_message())
        url = ((response.json().get("data") or [{}])[0]).get("url")
        if not url:
            return None

        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(httpx.get(url, timeout=180, follow_redirects=True).content)
        return url, path

    def _fit_prompt(self, prompt: str) -> str:
        prompt = " ".join(prompt.split())
        if len(prompt) <= self.MAX_PROMPT_CHARS:
            return prompt
        clipped = prompt[: self.MAX_PROMPT_CHARS]
        space = clipped.rfind(" ")
        if space > 0:
            clipped = clipped[:space]
        return clipped.rstrip(" ,;.")

    # Never chain more than this. Each link is another full generation, and a
    # scene wanting more than a minute is a storyboard problem, not a clip one.
    MAX_CHAIN = 3

    def _last_frame_uri(self, clip: Path) -> str | None:
        """The final frame of a clip, inline, to seed the next one.

        Inline because the frame exists only inside this run -- there is no URL
        to give the service, and it accepts a data URI here.
        """
        frame = clip.with_suffix(".last.jpg")
        try:
            ffmpeg_run(["-sseof", "-0.5", "-i", str(clip), "-frames:v", "1",
                        "-q:v", "3", str(frame)], label="last frame")
            data = frame.read_bytes()
        except Exception:
            return None
        finally:
            frame.unlink(missing_ok=True)
        return "data:image/jpeg;base64," + base64.b64encode(data).decode()

    def _fetch_chain(self, query: str, seconds: float, cache_dir: Path,
                     variant: int, reference: str | None) -> VisualAsset | None:
        """One long shot, generated as several continued clips and joined."""
        parts = min(self.MAX_CHAIN,
                    max(2, math.ceil(seconds / self.MAX_CLIP_SECONDS)))
        # Never ask a segment for more than one generation can hold: the frame
        # count is clamped anyway, so asking for more silently returns less.
        # Past the chain limit the scene is longer than we can fill and the
        # compositor pads it, which is the honest outcome rather than a chain
        # that claims a length it did not produce.
        each = min(self.MAX_CLIP_SECONDS, seconds / parts)
        covered = each * parts
        prompt = self._fit_prompt(query)
        joined = cache_dir / _cache_name(self.name, prompt, int(seconds * 100),
                                         variant, "chain")
        joined = joined.with_suffix(".mp4")
        if joined.exists():
            return VisualAsset(joined, self.name, query, self.width,
                               self.height, covered)

        segments: list[Path] = []
        carry = reference
        for index in range(parts):
            piece = self.fetch(query, each, cache_dir,
                               variant=variant * 97 + index + 1,
                               reference=carry, _segment=True)
            if piece is None:
                break
            segments.append(piece.path)
            # Continue from where the last one ended; without this each link
            # would restart the shot and the joins would read as cuts.
            carry = self._last_frame_uri(piece.path) or carry

        if not segments:
            return None
        if len(segments) == 1:
            return VisualAsset(segments[0], self.name, query, self.width,
                               self.height, each)

        listing = cache_dir / f"{joined.stem}.txt"
        listing.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in segments),
            encoding="utf-8")
        ffmpeg_run(["-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", str(joined)], label="join chained clip")
        listing.unlink(missing_ok=True)
        return VisualAsset(joined, self.name, query, self.width, self.height,
                           each * len(segments))

    def fetch(self, query: str, min_duration: float, cache_dir: Path,
              variant: int = 0, reference: str | None = None,
              _segment: bool = False) -> VisualAsset | None:
        if not self.keys:
            return None

        if reference is None:
            reference = self.references.get(variant)
        # A scene longer than one generation is continued rather than looped.
        if not _segment and min_duration > self.MAX_CLIP_SECONDS:
            return self._fetch_chain(query, min_duration, cache_dir, variant,
                                     reference)
        frames = self._frames_for(min_duration)
        prompt = self._fit_prompt(query)
        # Key on what was actually submitted, plus the variant, so trimmed
        # duplicates share a file and repeated scenes do not.
        path = cache_dir / _cache_name(self.name, prompt, frames, variant,
                                       "ref" if reference else "txt")
        if path.exists():
            return VisualAsset(path, self.name, query, self.width, self.height,
                               frames / self.FRAME_RATE)

        body = {
            "model": self.model,
            "prompt": prompt,
            "width": self.width,
            "height": self.height,
            "num_frames": frames,
            "frame_rate": self.FRAME_RATE,
            "negative_prompt": self.NEGATIVE_PROMPT,
        }
        if reference:
            # "ti2vid" is the mode name the service expects; anything else is
            # accepted and then silently ignored, which is how an earlier
            # attempt at this appeared to work while doing nothing.
            body["image"] = reference
            body["mode"] = "ti2vid"

        # 429 is our quota; 5xx is Agnes's own capacity ("no available server").
        # Both are temporary and both are worth waiting out on another key,
        # because the alternative is losing the scene entirely.
        # Every key is spent. Nothing here will change it.
        if self._exhausted.is_set():
            raise QuotaExhausted(self._exhausted_detail)

        headers = {}
        response = None
        last_detail = "no attempt made"
        delay = 15.0
        # A while loop rather than `for attempt in range(...)`, because finding
        # a key spent is not a failed attempt -- the submission never happened.
        # Charging it one would let a handful of dead keys use up the retries
        # the live keys still need. Retirement is permanent and the ring itself
        # raises once empty, so the extra passes are bounded by the key count.
        attempt = 0
        while attempt < self.submit_attempts:
            key = self._acquire_key()
            headers = {"Authorization": f"Bearer {key}"}
            self.gate.wait()
            if self._exhausted.is_set():
                raise QuotaExhausted(self._exhausted_detail)
            try:
                response = httpx.post(
                    f"{self.BASE_URL}/videos", headers=headers, json=body, timeout=90
                )
            except httpx.HTTPError as exc:
                response = None
                last_detail = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    break
                last_detail = f"{response.status_code}: {response.text[:160]}"

                # A refusal might be about this model rather than the key.
                # Try the model that is always allowed before deciding.
                refused = (is_quota_message(response.text)
                           or 400 <= response.status_code < 500)
                if refused and body.get("model") != self.FALLBACK_MODEL:
                    with self._model_lock:
                        print(f"   model refused, falling back to the standard "
                              f"model", flush=True)
                    body["model"] = self.FALLBACK_MODEL
                    attempt += 1
                    continue

                # Refused on the fallback too, so it is the key and not the
                # model. Keys are separate accounts, so this one is finished
                # while the rest are untouched: retire it and submit again on
                # the next. _acquire_key ends the run when none are left.
                if is_quota_message(response.text):
                    self._retire_key(key, last_detail)
                    continue

                if response.status_code == 429:
                    # Said out loud, because progress is only printed when a
                    # clip finishes and a throttled run otherwise looks busy.
                    print(f"   rate limited, waiting {delay:.0f}s "
                          f"(attempt {attempt + 1}/{self.submit_attempts})",
                          flush=True)
                elif response.status_code < 500:
                    raise VisualError(f"submit rejected, {last_detail}")

            attempt += 1
            if attempt < self.submit_attempts:
                time.sleep(delay)
                delay = min(delay * 1.8, 90.0)

        if response is None or response.status_code >= 400:
            raise VisualError(f"submit failed after retries, {last_detail}")

        submitted = response.json()
        video_id = submitted.get("video_id") or submitted.get("id")
        if not video_id:
            raise VisualError(f"agnes: no video_id in response: {response.text[:200]}")

        deadline = time.monotonic() + self.timeout
        url = None
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            status = httpx.get(
                self.POLL_URL, params={"video_id": video_id},
                headers=headers, timeout=30,
            ).json()
            state = str(status.get("status") or "").lower()
            if state in {"completed", "success", "succeeded"}:
                url = status.get("url") or status.get("video_url")
                break
            if state in {"failed", "error"}:
                raise VisualError(f"agnes: job failed: {str(status.get('error'))[:200]}")
        if not url:
            raise VisualError(f"agnes: timed out after {self.timeout:.0f}s")

        cache_dir.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=300, follow_redirects=True) as stream:
            stream.raise_for_status()
            with path.open("wb") as handle:
                for block in stream.iter_bytes(65536):
                    handle.write(block)

        return VisualAsset(
            path=path, source=self.name, query=query,
            width=self.width, height=self.height,
            duration=frames / self.FRAME_RATE,
        )


class VisualSourcer:
    """Runs a scene list through the provider chain and reports what it used."""

    def __init__(self, providers: list, cache_dir: str | Path):
        self.providers = [p for p in providers if p is not None]
        self.cache_dir = Path(cache_dir)

    def fetch_one(self, query: str, min_duration: float, variant: int = 0) -> VisualAsset:
        """A clip from the first provider that can supply one.

        Being out of quota is raised back as QuotaExhausted rather than
        flattened into a plain VisualError. The distinction is the whole
        difference between "this run is over" and "come back when the plans
        reset", and a caller that has to recover it by reading the message is a
        caller that will one day read it wrong.
        """
        errors = []
        exhausted = False
        for provider in self.providers:
            try:
                asset = provider.fetch(query, min_duration, self.cache_dir, variant)
                if asset:
                    return asset
                errors.append(f"{provider.name}: no result")
            except QuotaExhausted as exc:
                exhausted = True
                errors.append(f"{provider.name}: {exc}")
            except Exception as exc:
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")

        # The scene prompt runs to well over a thousand characters. Repeating
        # it here produced a single enormous log line, which is what a reader
        # then had to match against -- and a partly written one matched wrong.
        # The prompt is already in the storyboard; a handle on it is enough.
        short = query if len(query) <= 80 else query[:77].rstrip() + "..."
        detail = "; ".join(errors)
        if exhausted:
            raise QuotaExhausted(f"out of capacity for {short!r} ({detail})")
        raise VisualError(f"No clip for {short!r} ({detail})")

    def fetch_for_plan(
        self,
        plan,
        progress=None,
        fallback_query: str | None = None,
        max_workers: int = 4,
        attempts: int = 3,
        retry_delay: float = 20.0,
        on_asset=None,
    ) -> list[VisualAsset]:
        """Fetch a clip per scene, in parallel.

        AI generation takes a minute or two per clip, so scenes are fetched
        concurrently; the rate limiter still keeps submissions within quota.
        """
        done = 0
        lock = threading.Lock()
        attempts = max(1, attempts)

        def worker(scene):
            nonlocal done
            # Providers fail for transient reasons far more often than for the
            # prompt, so the same query is tried again before anything is
            # changed. Only then does it fall back to the style phrase, which
            # always describes something generatable.
            asset = None
            last = None
            for attempt in range(attempts):
                try:
                    asset = self.fetch_one(scene.visual_query, scene.duration, _variant(scene))
                    break
                except QuotaExhausted:
                    # Not a transient failure and not a prompt the wording can
                    # save: there is no capacity left to try with. Retrying
                    # here only spends a minute of sleep per scene on its way
                    # to the same answer, and the run resumes from the saved
                    # clips once the plans reset.
                    raise
                except VisualError as exc:
                    last = exc
                    if attempt < attempts - 1:
                        time.sleep(retry_delay * (attempt + 1))

            if asset is None:
                query = fallback_query or plan.style.visual_style
                try:
                    asset = self.fetch_one(query, scene.duration, _variant(scene))
                except VisualError as exc:
                    raise last or exc

            with lock:
                done += 1
                # Reported as soon as it lands, not after the whole batch:
                # generation is the expensive stage, and a crash partway
                # through must not discard the clips already paid for.
                if on_asset:
                    on_asset(scene, asset)
                if progress:
                    progress(done, len(plan.scenes))
            return asset

        workers = max(1, min(max_workers, len(plan.scenes)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(worker, plan.scenes))

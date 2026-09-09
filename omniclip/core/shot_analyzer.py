"""Shot analysis for videos that carry no narration.

Some videos say nothing at all — the story is entirely visual. There is no
transcript to rewrite, so instead the source is taken apart shot by shot: where
the cuts fall, how long each shot runs, and what is actually on screen. A vision
model reads one frame per shot, which keeps the rebuilt video anchored to what
the original showed rather than to anything invented about it.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..utils.media import ffmpeg_exe, probe_duration, run
from .scraper import PLAYER_CLIENTS, extract_video_id, ydl_options

_PTS = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
_SCORE = re.compile(r"lavfi\.scene_score=(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Cut:
    time: float
    score: float


class ShotError(RuntimeError):
    """Raised when a source video could not be analysed."""


def _retry_after(exc, default: float) -> float:
    """Honour the wait a provider asks for, when it names one."""
    text = str(exc)
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", text) or \
        re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", text)
    if match:
        return min(float(match.group(1)) + 1.0, 60.0)
    return default


def _parse_json(raw: str) -> dict:
    """Pull a JSON object out of a reply that may be fenced or prefaced."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace:
            raise ShotError(f"vision model did not return JSON: {raw[:160]}")
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError as exc:
            raise ShotError(f"vision model returned malformed JSON: {exc}") from exc


@dataclass
class Shot:
    index: int
    start: float
    duration: float
    description: str = ""
    donor_frame: str = ""

    @property
    def end(self) -> float:
        return self.start + self.duration


def _stream_url(info: dict, max_height: int = 480) -> str | None:
    """Pick a low-resolution video stream URL from yt-dlp's format list.

    Only frames are read from this file, so audio is unnecessary and the
    smallest usable video track is the right choice.
    """
    formats = [
        f for f in (info.get("formats") or [])
        if f.get("url") and f.get("vcodec") and f["vcodec"] != "none"
    ]
    if not formats:
        return info.get("url")

    def rank(f):
        height = f.get("height") or 9999
        return (height > max_height, height, f.get("ext") != "mp4")

    return sorted(formats, key=rank)[0]["url"]


def fetch_segment(url: str, dest: Path, seconds: float,
                  max_height: int = 480) -> Path:
    """Download only the first `seconds` of a source, using our own ffmpeg.

    yt-dlp can trim during download, but only via an ffmpeg it locates itself —
    and ours lives inside the virtualenv under a versioned filename it will not
    recognise. Asking yt-dlp for the stream URL and doing the cut here removes
    that dependency, and a three-hour 4K source stops being a gigabyte download
    for the sake of a few thumbnails.
    """
    import yt_dlp

    errors = []
    for client in PLAYER_CLIENTS:
        try:
            with yt_dlp.YoutubeDL(ydl_options(client, skip_download=True)) as ydl:
                info = ydl.extract_info(url, download=False)
            if info.get("_type") == "playlist" and info.get("entries"):
                info = info["entries"][0]
            stream = _stream_url(info, max_height)
            if not stream:
                errors.append(f"{client or 'default'}: no video stream")
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            run(["-t", f"{seconds:.3f}", "-i", stream, "-an",
                 "-c:v", "copy", str(dest)], label="fetch segment")
            return dest
        except Exception as exc:
            errors.append(f"{client or 'default'}: {str(exc).splitlines()[0]}")

    raise ShotError(f"Could not fetch a segment of {url}:\n  " + "\n  ".join(errors))


def download_video(url: str, out_dir: str | Path,
                   max_seconds: float | None = None) -> Path:
    """Fetch the source video itself — needed to look at its frames."""
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    suffix = f"_first{int(max_seconds)}" if max_seconds else ""

    if max_seconds:
        video_id = extract_video_id(url) or "source"
        dest = out_dir / f"{video_id}{suffix}.mp4"
        if dest.exists():
            return dest
        return fetch_segment(url, dest, max_seconds)

    for client in PLAYER_CLIENTS:
        # Only frames are read from this file, never pixels of the output, so
        # a small copy is downloaded. Pulling a 4K source to sample thumbnails
        # can mean gigabytes and many minutes for no benefit at all.
        options = ydl_options(
            client,
            # Progressive low-res files do not exist for every source (4K
            # uploads often have none), so separate streams are accepted and
            # merged, with a plain "best" as the last resort.
            format=(
                "bv*[height<=480]+ba/b[height<=480]/"
                "bv*[height<=720]+ba/b[height<=720]/b"
            ),
            outtmpl=str(out_dir / f"%(id)s{suffix}.%(ext)s"),
        )
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                if info.get("_type") == "playlist" and info.get("entries"):
                    info = info["entries"][0]
                path = Path(ydl.prepare_filename(info))
            if not path.exists():
                matches = list(out_dir.glob(f"{info.get('id', '*')}{suffix}.*"))
                if not matches:
                    continue
                path = matches[0]
            return path
        except Exception as exc:
            errors.append(f"{client or 'default'}: {str(exc).splitlines()[0]}")

    raise ShotError(f"Could not download video for {url}:\n  " + "\n  ".join(errors))


def find_cuts(video: str | Path, threshold: float = 0.08,
              cluster_window: float = 0.4) -> list[Cut]:
    """Scene changes with their strength, one entry per real cut.

    ffmpeg flags a single hard cut on two or three consecutive frames, so raw
    output triples the apparent shot count. Detections closer together than
    `cluster_window` are collapsed into the strongest of the group.
    """
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(video), "-filter:v",
         f"select='gt(scene,{threshold})',metadata=print", "-f", "null", "-"],
        capture_output=True, text=True,
        creationflags=creationflags,
    )
    stderr = proc.stderr or ""
    times = [float(t) for t in _PTS.findall(stderr)]
    scores = [float(s) for s in _SCORE.findall(stderr)]
    raw = sorted(zip(times, scores), key=lambda p: p[0])

    clustered: list[Cut] = []
    for time_s, score in raw:
        if clustered and time_s - clustered[-1].time <= cluster_window:
            if score > clustered[-1].score:
                clustered[-1] = Cut(clustered[-1].time, score)
        else:
            clustered.append(Cut(time_s, score))
    return clustered



def native_shot_seconds(video: str | Path, threshold: float = 0.08,
                        cluster_window: float = 0.4) -> float:
    """The typical length of one shot in the source, before any merging.

    The median rather than the mean, so a single long establishing take at the
    front does not misreport a fast-cut film as a slow one.
    """
    total = probe_duration(video)
    cuts = find_cuts(video, threshold, cluster_window)
    if not cuts:
        return 0.0
    edges = [0.0] + [c.time for c in cuts] + [total]
    lengths = sorted(b - a for a, b in zip(edges, edges[1:]) if b > a)
    if not lengths:
        return 0.0
    middle = len(lengths) // 2
    if len(lengths) % 2:
        return lengths[middle]
    return (lengths[middle - 1] + lengths[middle]) / 2


def detect_shots(
    video: str | Path,
    threshold: float = 0.08,
    min_seconds: float = 3.0,
    max_seconds: float = 7.0,
    cluster_window: float = 0.4,
    split_after: float = 0.0,
) -> list[Shot]:
    """Group a source's cuts into shots worth generating one clip each.

    Rather than a fixed minimum length, the decision adapts to how the video is
    actually edited. Below `min_seconds` a shot is always absorbed into its
    neighbour, above `max_seconds` it always stands alone, and in between the
    cut's own strength decides — measured against the median cut strength of
    this particular video, so a frantic edit and a languid one are each judged
    on their own terms rather than against a number picked in advance.
    """
    video = Path(video)
    total = probe_duration(video)
    cuts = [c for c in find_cuts(video, threshold, cluster_window) if 0 < c.time < total]

    if not cuts:
        # A source with no cuts at all still needs slicing, or it becomes one
        # clip looped for the whole runtime.
        return subdivide([Shot(index=0, start=0.0, duration=total)], split_after)

    ranked = sorted(c.score for c in cuts)
    strong_enough = ranked[len(ranked) // 2]  # this video's median cut strength

    shots: list[Shot] = []
    start = 0.0
    for cut in cuts:
        length = cut.time - start
        if length < min_seconds:
            continue                      # too brief to stand alone
        if length < max_seconds and cut.score < strong_enough:
            continue                      # a soft cut inside a workable shot
        shots.append(Shot(index=len(shots), start=start, duration=length))
        start = cut.time

    tail = total - start
    if tail <= 0:
        pass
    elif tail < min_seconds and shots:
        # Absorb a stub ending rather than generating a clip for a fraction.
        last = shots[-1]
        shots[-1] = Shot(last.index, last.start, total - last.start, last.description)
    else:
        shots.append(Shot(index=len(shots), start=start, duration=tail))

    return subdivide(shots or [Shot(index=0, start=0.0, duration=total)],
                     split_after)


def subdivide(shots: list[Shot], split_after: float = 0.0) -> list[Shot]:
    """Break very long shots into segments of their own.

    Slow content — meditations, ambient pieces, static talking heads — can run
    for minutes without a single cut. Faithfully mirroring that means one clip
    looped for the whole runtime, which is not a video anyone watches. Long
    stretches are divided into even segments so each gets its own visual, while
    genuinely cut content is left exactly as detected.
    """
    if split_after <= 0:
        return shots

    result: list[Shot] = []
    for shot in shots:
        if shot.duration <= split_after:
            result.append(Shot(len(result), shot.start, shot.duration, shot.description))
            continue
        pieces = max(2, round(shot.duration / split_after))
        length = shot.duration / pieces
        for step in range(pieces):
            result.append(
                Shot(len(result), shot.start + step * length, length, shot.description)
            )
    return result


def _frame_data_url(video: Path, at: float, width: int = 448) -> str:
    """Grab one frame and encode it for the vision model."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
         "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={width}:-1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True,
        creationflags=creationflags,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise ShotError(f"Could not read a frame at {at:.1f}s from {video.name}")
    return "data:image/jpeg;base64," + base64.b64encode(proc.stdout).decode()


def extract_donor_frames(video: Path, shots: list[Shot], output_dir: Path) -> None:
    """Extract pristine full-resolution keyframe for each shot from donor video."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        exe = ffmpeg_exe()
        for s in shots:
            out_path = output_dir / f"shot_{s.index:04d}.jpg"
            if not out_path.exists():
                mid = s.start + (s.duration / 2.0)
                cmd = [
                    str(exe), "-y", "-ss", f"{mid:.3f}", "-i", str(video),
                    "-frames:v", "1", "-q:v", "2", str(out_path)
                ]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
                except Exception:
                    pass
            if out_path.exists():
                s.donor_frame = str(out_path)
    except Exception as exc:
        print(f"Notice: extract_donor_frames: {exc}", flush=True)


# Agnes accepts ~1500-character prompts (measured). The style phrase is
# appended to every shot, so the two together must stay inside that.
STYLE_CHAR_LIMIT = 220
SHOT_CHAR_LIMIT = 900

_STYLE_PROMPT = f"""These frames are sampled from one short video.

Describe the shared visual style in at most {STYLE_CHAR_LIMIT} characters:
medium or art style, colour grade, lighting, and setting. This phrase is
appended to every scene so the rebuilt video stays consistent.

Reply with the phrase alone. No preamble, no quotes."""

CHARACTER_CHAR_LIMIT = 220

_ROSTER_PROMPT = f"""These frames are sampled from one video.

Identify every character who appears more than once. For each, write a canonical
appearance description of at most {CHARACTER_CHAR_LIMIT} characters covering
age, build, hair, facial hair, skin tone, and clothing with colours — enough
that an artist could redraw them identically without seeing the original.

Give each a short lowercase id, like "younger_brother" or "old_man".

Ignore crowds and one-off background figures. If nobody recurs, return an empty
list.

Return ONLY JSON:
{{{{"characters": [{{{{"id": "...", "appearance": "..."}}}}]}}}}"""


_SHOT_BATCH_PROMPT = f"""These are {{count}} still frames taken from one video,
in order. They are labelled {{labels}}.

For EACH frame, write a prompt for an AI video generator that recreates that
moment: the subject and its appearance, what it is doing, the setting, the
lighting, and the camera framing. Present tense, concrete, filmable.

Each prompt may run up to {SHOT_CHAR_LIMIT} characters. Detail helps, but every
word must describe something actually visible in that frame.

Describe only what you can see. Do not invent dialogue, on-screen text, plot, or
anything not in the frame. Do not write "image", "frame", or "photo".

{{roster}}

If a frame is not story content — a title card, channel logo, end credits,
subscribe screen, or a plain text/graphic slate — set "skip": true for it and
leave the prompt empty. Recreating a channel's intro graphics wastes a scene.

Return ONLY JSON of the form:
{{{{"shots": [{{{{"index": <label number>, "prompt": "...", "skip": false}}}}]}}}}
Include every frame, using its exact label number."""


_ROSTER_INSTRUCTION = """These characters recur through the video:
{roster}

When one of them appears, keep their look consistent — but do NOT open the
prompt with their description. Every prompt must START with what is happening
and how it is framed: the action, the setting, the camera. Put the character's
fixed appearance at the END, in brackets, copied word for word.

Like this:
  Low-angle shot, dust drifting in the light: he drives a blade into dry soil
  and stops, staring at the horizon. [young_farmer: <exact appearance wording>]

Prompts that all begin the same way produce shots that all look the same. The
opening is where the variety lives; the bracket is where the consistency lives.
Only name a character actually visible in that frame."""


class ShotDescriber:
    """Reads frames with a vision model and writes generation prompts.

    Frames are sent several per request. Providers with big context but tight
    request quotas (Gemini's free tier allows five calls a minute) would
    otherwise stall for minutes describing a single video one shot at a time.
    """

    def __init__(self, api_key: str, base_url: str, model: str,
                 max_tokens: int = 2500, batch_size: int = 3,
                 max_retries: int = 6, fallback: dict | None = None,
                 min_interval: float = 35.0, frame_width: int = 320):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.model = model
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.max_retries = max_retries
        # Images are token-heavy, so a batch of frames can consume most of a
        # per-minute token allowance on its own. Pacing requests up front is
        # far cheaper than discovering the limit through repeated rejections.
        self.min_interval = min_interval
        # Frame width drives token cost more than anything else here; a wider
        # frame buys little description quality and a lot of quota.
        self.frame_width = frame_width
        self._next_call = 0.0
        # A second provider used only when the first runs out of quota. Its
        # image limit can differ, so batches are re-split when it takes over.
        self.fallback = fallback
        self._fallback_client = None
        if fallback:
            self._fallback_client = OpenAI(
                base_url=fallback["base_url"], api_key=fallback["api_key"],
                max_retries=0,
            )

    def _pace(self) -> None:
        wait = self._next_call - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next_call = time.monotonic() + self.min_interval

    def _call(self, client, model, content, max_tokens) -> str:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        text = (response.choices[0].message.content or "").strip()
        # Some reasoning models emit a <think> block before the answer.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return " ".join(text.strip().strip('"').split())

    def _ask(self, prompt: str, images: list[str]) -> str:
        from openai import APIStatusError, RateLimitError

        content = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": url}} for url in images]

        delay = 8.0
        last = None
        for attempt in range(self.max_retries):
            try:
                self._pace()
                text = self._call(self.client, self.model, content, self.max_tokens)
                if text:
                    return text
                last = ShotError("vision model returned nothing")
            except (RateLimitError, APIStatusError) as exc:
                if getattr(exc, "status_code", None) not in (429, 413, 503):
                    raise
                last = exc
                if attempt < self.max_retries - 1:
                    time.sleep(_retry_after(exc, delay))
                    delay = min(delay * 2, 60.0)

        primary_error = last
        if self._fallback_client:
            try:
                text = self._call(
                    self._fallback_client, self.fallback["model"], content,
                    self.fallback.get("max_completion_tokens", self.max_tokens),
                )
                if text:
                    return text
            except Exception as exc:
                # Report the primary's failure first — the fallback's message
                # otherwise hides why the run actually stopped.
                raise ShotError(
                    f"vision failed on {self.model} after {self.max_retries} "
                    f"tries ({primary_error}); fallback {self.fallback['model']} "
                    f"also failed ({exc})"
                ) from exc

        raise ShotError(
            f"vision failed on {self.model} after {self.max_retries} tries: "
            f"{primary_error}"
        )

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        """Models overshoot a character limit; enforce it at a clause boundary.

        Cutting on the nearest word leaves fragments like "matching flat-topped"
        -- an adjective with its noun removed. The image model then invents the
        missing noun, which is one of the ways a character stops looking like
        itself. Ending on the last complete clause costs a few words and leaves
        a description that still reads as a description.
        """
        if len(text) <= limit:
            return text
        cut = text[:limit]
        stop = max(cut.rfind(", "), cut.rfind(". "), cut.rfind("; "))
        if stop > limit * 0.5:
            return cut[:stop].rstrip(" ,;.")
        space = cut.rfind(" ")
        return (cut[:space] if space > 0 else cut).rstrip(" ,;.")

    def _sample(self, shots: list[Shot], count: int) -> list[Shot]:
        """Evenly spaced shots, never more than the provider accepts at once."""
        count = max(1, min(count, self.batch_size))
        step = max(1, len(shots) // count)
        return (shots[::step][:count]) or shots[:1]

    def describe_style(self, video: Path, shots: list[Shot], samples: int = 3) -> str:
        picks = self._sample(shots, samples)
        images = [_frame_data_url(video, s.start + s.duration / 2, self.frame_width) for s in picks]
        style = self._ask(_STYLE_PROMPT, images) or "cinematic, consistent colour grade"
        return self._clip(style, STYLE_CHAR_LIMIT)

    def find_characters(self, video: Path, shots: list[Shot],
                        samples: int | None = None) -> list[dict]:
        """Build a cast list with fixed appearance wording for each character.

        Without this every shot is described independently, so the same person
        comes back as a different-looking stranger in each generated clip. There
        is usually more than one character, so all of them are pinned, not just
        a lead.
        """
        # Deliberately a single pass. Splitting the sample across several
        # requests made each pass name the same person differently — one man
        # came back as "shepherd", "young_man" and "curly_haired_youth" — and
        # three fixed descriptions for one face is worse than none. Fewer
        # frames, one naming authority.
        samples = samples or self.batch_size
        step = max(1, len(shots) // max(1, samples))
        picks = (shots[::step][:samples]) or shots[:1]
        cast: dict[str, str] = {}
        for start in range(0, len(picks), self.batch_size):
            group = picks[start:start + self.batch_size]
            images = [_frame_data_url(video, s.start + s.duration / 2, self.frame_width) for s in group]
            try:
                data = _parse_json(self._ask(_ROSTER_PROMPT, images))
            except (ShotError, Exception):
                continue
            for entry in data.get("characters") or []:
                identity = str(entry.get("id") or "").strip()
                appearance = self._clip(
                    str(entry.get("appearance") or "").strip(), CHARACTER_CHAR_LIMIT
                )
                if identity and appearance:
                    cast.setdefault(identity, appearance)

        return self._consolidate(
            [{"id": k, "appearance": v} for k, v in cast.items()]
        )

    def _consolidate(self, cast: list[dict]) -> list[dict]:
        """Merge duplicate identities invented by separate roster passes.

        Each pass names characters independently, so one person comes back as
        "shepherd", "young_man" and "curly_haired_youth". Left alone that is
        worse than no roster at all — the same face gets three different fixed
        descriptions. This is a text-only request, so it costs almost nothing.
        """
        if len(cast) < 2:
            return cast

        listing = "\n".join(f"- {c['id']}: {c['appearance']}" for c in cast)
        prompt = (
            "These character descriptions came from separate passes over one "
            "video, so the same person may appear more than once under "
            "different names.\n\nMerge entries that describe the same person. "
            "Keep the fullest wording for each. Distinct people must stay "
            "separate — do not collapse two different characters into one.\n\n"
            f"Return ONLY JSON:\n"
            '{"characters": [{"id": "...", "appearance": "..."}]}\n\n'
            f"ENTRIES:\n{listing}"
        )
        try:
            data = _parse_json(self._ask(prompt, []))
        except Exception:
            return cast

        merged = []
        for entry in data.get("characters") or []:
            identity = str(entry.get("id") or "").strip()
            appearance = self._clip(
                str(entry.get("appearance") or "").strip(), CHARACTER_CHAR_LIMIT
            )
            if identity and appearance:
                merged.append({"id": identity, "appearance": appearance})
        return merged or cast

    def describe_shots(self, video: Path, shots: list[Shot], progress=None,
                       cast: list[dict] | None = None) -> set[int]:
        """Fill in each shot's description, several frames per request.

        Returns the indices of shots the model judged to be non-content —
        title cards, logos, end screens — for the caller to drop.
        """
        skipped: set[int] = set()
        roster = ""
        if cast:
            listing = "\n".join(f"  - {c['id']}: {c['appearance']}" for c in cast)
            roster = _ROSTER_INSTRUCTION.format(roster=listing)

        done = 0
        for start in range(0, len(shots), self.batch_size):
            batch = shots[start:start + self.batch_size]
            images = [
                _frame_data_url(video, s.start + s.duration / 2, self.frame_width) for s in batch
            ]
            labels = ", ".join(str(s.index) for s in batch)
            raw = self._ask(
                _SHOT_BATCH_PROMPT.format(
                    count=len(batch), labels=labels, roster=roster
                ),
                images,
            )

            by_index: dict[int, str] = {}
            try:
                for entry in (_parse_json(raw).get("shots") or []):
                    try:
                        key = int(entry.get("index"))
                    except (TypeError, ValueError):
                        continue
                    if entry.get("skip"):
                        skipped.add(key)
                        continue
                    by_index[key] = str(entry.get("prompt") or "")
            except ShotError:
                by_index = {}

            ordered = [v for _, v in sorted(by_index.items())]
            for offset, shot in enumerate(batch):
                # Models sometimes renumber or skip indices, so fall back to
                # the label, then the position within the batch, then order.
                text = (
                    by_index.get(shot.index)
                    or by_index.get(offset)
                    or (ordered[offset] if offset < len(ordered) else "")
                )
                shot.description = self._clip(text.strip(), SHOT_CHAR_LIMIT)
                done += 1
                if progress:
                    progress(done, len(shots))

        # Anything still blank would silently degrade to a generic style
        # prompt, so it is asked for again on its own.
        for shot in shots:
            if shot.description or shot.index in skipped:
                continue
            try:
                raw = self._ask(
                    _SHOT_BATCH_PROMPT.format(
                        count=1, labels=str(shot.index), roster=roster
                    ),
                    [_frame_data_url(video, shot.start + shot.duration / 2,
                                     self.frame_width)],
                )
                entries = _parse_json(raw).get("shots") or []
                if entries:
                    shot.description = self._clip(
                        str(entries[0].get("prompt") or "").strip(), SHOT_CHAR_LIMIT
                    )
            except Exception:
                continue

        return skipped


def analyse(
    url: str,
    cache_dir: str | Path,
    api_key: str,
    base_url: str,
    model: str,
    threshold: float = 0.08,
    min_shot_seconds: float = 3.0,
    max_shot_seconds: float = 7.0,
    cluster_window: float = 0.4,
    split_after: float = 0.0,
    max_duration: float | None = None,
    start_seconds: float = 0.0,
    scene_cap: int = 0,
    max_clip_seconds: float = 19.0,
    batch_size: int = 2,
    max_tokens: int = 2000,
    min_interval: float = 30.0,
    frame_width: int = 320,
    fallback: dict | None = None,
    progress=None,
    refresh: bool = False,
) -> tuple[list[Shot], str, Path]:
    """Return the source's shots with descriptions, plus its shared style.

    The analysis is cached beside the downloaded video. A vision model phrases
    each description slightly differently every time it is asked, and those
    descriptions are the cache keys for generated clips — so re-analysing an
    unchanged video would silently invalidate every clip already generated for
    it. Reusing the stored analysis is what makes a retry resume instead of
    starting over.
    """
    video = download_video(url, cache_dir, max_duration)
    if start_seconds > 0:
        # Cut the head off once and analyse the remainder, so every timestamp
        # downstream is relative to the film rather than to the credits.
        trimmed = video.with_name(f"{video.stem}_from{int(start_seconds)}.mp4")
        if not trimmed.exists():
            run(["-ss", f"{start_seconds:.3f}", "-i", str(video),
                 "-c", "copy", str(trimmed)], label="skip opening")
        video = trimmed
    # The source's own pace decides the scene length, so the rebuild cuts as
    # often as the original does instead of averaging it into long takes.
    native = native_shot_seconds(video, threshold, cluster_window)
    if native > 0:
        from .storyboard import target_scene_seconds

        target = target_scene_seconds(probe_duration(video), scene_cap,
                                      native, max_clip_seconds)
        min_shot_seconds = max(1.5, min(min_shot_seconds, target * 0.8)) if scene_cap > 0 else max(1.5, native * 0.8 if native > 0 else 1.5)
        max_shot_seconds = min(max_clip_seconds, target * 1.6)
        split_after = min(max_clip_seconds, target * 1.8)

    store = video.with_suffix(".shots.json")

    # The cache is only valid for the settings that produced it. Scene length
    # targets change how cuts are merged, so reusing an analysis made under
    # different targets would silently return the wrong shot grid.
    signature = {
        "threshold": threshold, "min": min_shot_seconds,
        "max": max_shot_seconds, "cluster": cluster_window,
        "split_after": split_after, "model": model,
        "start": start_seconds, "native": round(native, 2),
    }
    if store.exists() and not refresh:
        cached = json.loads(store.read_text(encoding="utf-8"))
        if cached.get("signature") == signature:
            shots = [Shot(**entry) for entry in cached["shots"]]
            frames_dir = video.parent / "donor_frames"
            extract_donor_frames(video, shots, frames_dir)
            return shots, cached["style"], video

    shots = detect_shots(video, threshold, min_shot_seconds,
                         max_shot_seconds, cluster_window, split_after)
    describer = ShotDescriber(
        api_key, base_url, model, max_tokens=max_tokens, batch_size=batch_size,
        fallback=fallback, min_interval=min_interval, frame_width=frame_width,
    )
    style = describer.describe_style(video, shots)
    cast = describer.find_characters(video, shots)
    skipped = describer.describe_shots(video, shots, progress, cast)

    # Drop title cards and logos, then renumber so indices stay contiguous.
    # If the model wants to discard everything it has misjudged the material —
    # a still, textless backdrop is legitimate content for slow videos — so the
    # skip flags are ignored rather than leaving nothing to build.
    if skipped and len(skipped) < len(shots):
        shots = [s for s in shots if s.index not in skipped and s.description]
        for position, shot in enumerate(shots):
            shot.index = position
    if not shots:
        raise ShotError("Shot analysis produced nothing to rebuild")

    # Persist vision AI analysis immediately so network/API work is never lost
    def _save_cache() -> None:
        try:
            store.write_text(
                json.dumps(
                    {"style": style, "cast": cast, "signature": signature,
                     "shots": [asdict(s) for s in shots]},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as err:
            print(f"Notice: failed to cache shots.json: {err}", flush=True)

    _save_cache()

    # Extract full-resolution donor keyframes for every shot
    frames_dir = video.parent / "donor_frames"
    extract_donor_frames(video, shots, frames_dir)
    _save_cache()

    return shots, style, video

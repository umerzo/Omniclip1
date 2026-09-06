"""Pipeline orchestrator.

One call takes a link and returns a finished video. Every stage is a module in
core/; this file only wires them together and reports progress.

Two modes, chosen automatically:

  narrated - the source talks. Its transcript is rewritten into an original
             script, spoken with TTS, and captioned.
  mirror   - the source barely speaks. There is nothing to rewrite, so the
             video is rebuilt shot by shot from what its frames actually show,
             and stays silent because the original was silent.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .core.compositor import compose
from .core.jobstore import Job, SceneRecord, open_job
from .core.scraper import (
    IngestError,
    SourceVideo,
    TranscriptSegment,
    fetch_metadata,
    scrape,
)
from .core.script_brain import Scene, ScriptBrain, ScriptPlan, StyleProfile
from .core.language import name_of, normalise, pick_voice, word_level_possible
from .core.music import MusicError, build_bed, mix_under
from .core.shot_analyzer import Shot, analyse
from .core.storyboard import Storyboard, target_scene_seconds, write_storyboard
from .core.subtitles import build_from_lines, build_subtitles
from .core.transcriber import make_transcriber
from .core.tts_engine import narrate_plan
from .core.video_rotator import (
    AgnesProvider,
    PexelsProvider,
    PixabayProvider,
    VisualAsset,
    VisualSourcer,
)
from .utils.config import load_env, load_settings, resolve_path


@dataclass
class PipelineResult:
    video_path: Path
    plan: ScriptPlan
    source: SourceVideo
    mode: str = "narrated"
    assets: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    frame: dict = field(default_factory=dict)
    job: object = None

    @property
    def duration(self) -> float:
        return self.plan.duration_seconds


def agnes_keys() -> list[str]:
    """AGNES_API_KEY plus any numbered extras. More keys, more quota."""
    load_env()  # safe to call repeatedly; never overwrites a real env var
    keys = [os.environ.get("AGNES_API_KEY", "")]
    index = 2
    while (key := os.environ.get(f"AGNES_API_KEY_{index}")):
        keys.append(key)
        index += 1
    return [k for k in keys if k]


def _vision_fallback(cfg: dict) -> dict | None:
    """Second vision model, on the same endpoint as the first.

    A fallback on a different provider is not usable here: the retry reuses the
    batch that already failed, and providers disagree on how many images one
    request may carry. Staying on the same endpoint and only swapping the model
    keeps the batch valid — an auto-router works well in this slot because it
    picks whichever free model is currently up.
    """
    model = cfg["shots"].get("fallback_vision_model")
    if not model:
        return None
    return {
        "base_url": cfg["script"]["base_url"],
        "api_key": cfg["script"]["api_key"],
        "model": model,
        "max_completion_tokens": cfg["shots"].get("vision_max_tokens", 6000),
    }


def frame_spec(cfg: dict, aspect: str | None = None) -> dict:
    """Output and generation dimensions for the chosen aspect ratio."""
    aspect = aspect or cfg["video"]["aspect_ratio"]
    frames = cfg["video"]["frames"]
    if aspect not in frames:
        raise ValueError(
            f"Unknown aspect ratio {aspect!r}; known: {', '.join(sorted(frames))}"
        )
    return {"aspect": aspect, "fps": cfg["video"]["fps"], **frames[aspect]}


def build_providers(
    cfg: dict,
    prefer_ai: bool = True,
    allow_stock: bool = True,
    frame: dict | None = None,
    video_model: str | None = None,
) -> list:
    """Provider chain: AI generation first when reachable, stock as backup.

    With allow_stock off, every scene must come from AI generation — a scene
    that cannot be generated fails the run instead of quietly becoming stock.
    """
    frame = frame or frame_spec(cfg)
    portrait = frame["height"] > frame["width"]
    providers = []
    if prefer_ai:
        agnes = AgnesProvider(
            api_key=agnes_keys(),
            rpm=cfg["queue"]["requests_per_minute"],
            width=frame["gen_width"],
            height=frame["gen_height"],
            model=video_model or cfg["visuals"]["agnes_model"],
            pool_rpm=cfg["queue"]["pool_requests_per_minute"],
        )
        if not agnes.available():
            if not allow_stock:
                raise RuntimeError(
                    "Agnes is unreachable and stock fallback is disabled. "
                    "Check AGNES_API_KEY."
                )
        else:
            providers.append(agnes)
    if allow_stock:
        providers.append(
            PexelsProvider(cfg["visuals"]["pexels_api_key"], portrait=portrait)
        )
        providers.append(
            PixabayProvider(cfg["visuals"]["pixabay_api_key"], portrait=portrait)
        )
    if not providers:
        raise RuntimeError("No visual providers available")
    return providers




# A rate limit or a server error is a "later", not a "no". Anything else is a
# real failure and is raised immediately rather than waited on.
_TRANSIENT = ("429", "too many requests", "rate limit", "timed out", "timeout",
              "temporarily unavailable", "connection reset", "connection aborted",
              "500", "502", "503", "504")


def is_transient(error: Exception) -> bool:
    """Whether asking again later could plausibly succeed."""
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _TRANSIENT)


def with_backoff(work, attempts: int = 4, first_wait: float = 10.0,
                 progress=None):
    """Run `work`, waiting longer each time it fails for a transient reason.

    10s, 30s, then 90s. Chosen from what actually cleared a YouTube 429 by
    hand: a couple of minutes. Shorter waits spend attempts without ever
    reaching the point where the limit lifts.
    """
    wait = first_wait
    for attempt in range(1, attempts + 1):
        try:
            return work()
        except Exception as error:
            if attempt == attempts or not is_transient(error):
                raise
            if progress:
                progress(f"{type(error).__name__}: {error}"[:90],
                         attempt, attempts, wait)
            time.sleep(wait)
            wait *= 3
    raise RuntimeError("unreachable")


def _ingest(url: str, cfg: dict, translate: bool, progress=None) -> SourceVideo | None:
    """Transcript from captions, or from Whisper when there are none.

    A source that rate-limits us is retried rather than abandoned; only a real
    ingest failure falls through to the metadata-only path.
    """
    def announce(reason, attempt, attempts, wait):
        if progress:
            progress(0, 0)
        print(f"   ingest attempt {attempt}/{attempts} failed ({reason}); "
              f"waiting {wait:.0f}s", flush=True)

    def fetch():
        return scrape(
            url,
            languages=cfg["ingest"]["languages"],
            transcriber=make_transcriber(
                # Whisper lives on its own endpoint; the script model does not
                # host it, so these are deliberately separate from [script].
                api_key=cfg["ingest"]["whisper_api_key"],
                base_url=cfg["ingest"]["whisper_base_url"],
                cache_dir=resolve_path(cfg["ingest"]["cache_dir"]) / "audio",
                model=cfg["ingest"]["whisper_model"],
                translate=translate,
                progress=progress,
            ),
        )

    try:
        return with_backoff(fetch, progress=announce)
    except IngestError:
        return None
    except Exception as error:
        # Out of retries on something transient, or a network error that is not
        # an IngestError. Neither is worth ending the run over: the caller falls
        # back to metadata, and the transcript check downstream decides whether
        # what is left is enough to build from.
        print(f"   ingest failed after retries: {type(error).__name__}: {error}",
              flush=True)
        return None


def _source_from_job(job: Job) -> SourceVideo:
    """Rebuild enough of the source to continue without re-ingesting.

    Only the transcript, title and runtime are ever used after ingest, so those
    are what the job stores; the per-caption timings are not needed again.
    """
    saved = job.source
    transcript = saved.get("transcript") or ""
    duration = float(saved.get("duration_seconds") or 0.0)
    segments = (
        [TranscriptSegment(text=transcript, start=0.0, duration=duration)]
        if transcript else []
    )
    return SourceVideo(
        url=saved.get("url") or job.url,
        video_id=saved.get("video_id"),
        title=saved.get("title") or "Untitled",
        description=saved.get("description") or "",
        uploader=saved.get("uploader") or "",
        duration_seconds=duration,
        segments=segments,
        language=saved.get("language"),
    )


def _plan_from_job(job: Job) -> ScriptPlan:
    """Rebuild the plan from saved scene records."""
    style = job.style or {}
    plan = ScriptPlan(
        title=style.get("title") or job.source.get("title") or "Untitled",
        style=StyleProfile(
            tone=style.get("tone", ""),
            pacing_wpm=float(style.get("pacing_wpm") or 0.0),
            structure=style.get("structure", ""),
            visual_style=style.get("visual_style", ""),
        ),
    )
    for record in job.scenes:
        plan.scenes.append(
            Scene(index=record.index, narration=record.narration,
                  visual_query=record.visual_query,
                  start=record.start, duration=record.duration,
                  variant=record.variant)
        )
    return plan


def _store_plan(job: Job, plan: ScriptPlan) -> None:
    # Merge rather than replace: the cast and style came from shot analysis and
    # are not in the plan, so overwriting here silently removed character
    # locking from every prompt on the next resume.
    job.style = {
        **(job.style or {}),
        "title": plan.title,
        "tone": plan.style.tone,
        "pacing_wpm": plan.style.pacing_wpm,
        "structure": plan.style.structure,
        "visual_style": plan.style.visual_style,
    }
    for scene in plan.scenes:
        record = job.scene(scene.index)
        if record is None:
            record = SceneRecord(index=scene.index)
            job.scenes.append(record)
        record.narration = scene.narration
        record.visual_query = scene.visual_query
        record.start = scene.start
        record.duration = scene.duration
    job.save()



TRANSCRIPT_TAGS = re.compile(r"\[[^\]]*\]")


def transcript_health(transcript: str, duration: float) -> dict:
    """Judge whether a transcript is usable before anything is built on it.

    Whisper does not fail loudly on a language it cannot handle. It returns
    filler — bracketed event tags and repeated "Foreign speech" — which looks
    like a transcript and counts like one. Building a whole video on that is how
    a talkative cartoon came back as a silent slideshow, so the shape of the
    text is checked rather than just its length.
    """
    raw = transcript or ""
    tags = TRANSCRIPT_TAGS.findall(raw)
    spoken = TRANSCRIPT_TAGS.sub("", raw).split()
    words = len(spoken)
    lowered = [w.strip(".,!?").lower() for w in spoken]
    foreign = sum(1 for w in lowered if w in ("foreign", "fore"))
    unique = len(set(lowered))

    rate = words / duration if duration else 0.0
    broken = bool(
        words
        and (
            foreign / max(1, words) > 0.25
            or (words > 20 and unique / words < 0.25)
        )
    )
    return {
        "words": words, "tags": len(tags), "rate": round(rate, 3),
        "foreign_ratio": round(foreign / max(1, words), 3),
        "unique_ratio": round(unique / max(1, words), 3),
        "broken": broken,
        "has_music": any("music" in t.lower() for t in tags),
        "has_effects": any(
            k in t.lower() for t in tags
            for k in ("laugh", "scream", "gasp", "cry", "applause", "footsteps")
        ),
    }


def is_genuinely_silent(health: dict, duration: float) -> bool:
    """Whether a source simply has no speech, as opposed to a failed transcript.

    Both look like "almost no words". The difference is what surrounds them: a
    failed transcription is full of filler like "Foreign speech", while a mute
    cartoon comes back nearly empty, often with music and effect tags. Treating
    the second as a failure would refuse videos we can rebuild; treating the
    first as silence is what produced a ten minute slideshow.
    """
    if health["broken"]:
        return False
    return health["rate"] < 0.3 and (health["has_music"] or health["has_effects"]
                                     or health["words"] < 5)


class PipelinePaused(Exception):
    """Raised to stop a run cleanly at a requested checkpoint.

    Not a failure -- the caller asked to stop here, most often to look at the
    stills before paying for the clips they anchor. `job.json` is already
    saved and complete up to this stage, so calling run() again with the same
    output directory picks up exactly where this left off.
    """

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"paused after '{stage}' as requested")


class IngestQualityError(RuntimeError):
    """Raised when a transcript is too broken to build a video from."""




from .pipeline_run import plan_from_board, run  # noqa: E402,F401

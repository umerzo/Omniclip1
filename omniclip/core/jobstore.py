"""Durable job state.

A run is expensive — clip generation costs about a minute per scene — so losing
one to a crash at the last step is the worst failure this pipeline has. Every
stage records what it produced, which lets an interrupted run pick up where it
stopped and lets a single bad scene be redone without rebuilding the rest.

The file is also the contract a UI reads: stage progress, per-scene prompts and
durations, which provider supplied each clip, and what has been locked.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

JOB_FILE = "job.json"
SCHEMA_VERSION = 1

# Why a run ended, recorded as data. A reader used to have to infer this from
# the log, which meant control flow ran on substring matches against prose --
# and a half-flushed final line was enough to turn "waiting for capacity" into
# "failed". These are the only values `stop["reason"]` ever takes:
#
#   complete  the video was rendered
#   paused    stopped on purpose, waiting for a person (review, --pause-after)
#   quota     every generation key is spent; the work so far is kept
#   error     anything else, with the exception type in `detail`
#
# Anything deciding what happens next keys off this, never off the log.
STOP_REASONS = ("complete", "paused", "quota", "error")

# The pipeline's stages, in order, each with the label a UI should show and a
# rough share of total runtime for progress reporting.
#
# All three facts live together deliberately. They used to be spread across
# modules, and every time a stage was added or renamed the copies drifted --
# once crashing the interface on a stage that no longer existed. Anything that
# needs a label or a weight derives it from here.
STAGE_INFO = (
    ("ingest",     "Ingest",     2),
    ("shots",      "Shots",      8),
    ("storyboard", "Storyboard", 3),
    ("narrate",    "Voice",      5),
    ("captions",   "Captions",   1),
    ("stills",     "Stills",     8),
    ("review",     "Review",     3),
    ("visuals",    "Clips",     57),
    ("music",      "Music",      5),
    ("render",     "Render",     8),
)

STAGES = tuple(name for name, _, _ in STAGE_INFO)
STAGE_LABELS = {name: label for name, label, _ in STAGE_INFO}
STAGE_WEIGHTS = {name: weight for name, _, weight in STAGE_INFO}


@dataclass
class SceneRecord:
    index: int
    start: float = 0.0
    duration: float = 0.0
    narration: str = ""
    narration_en: str = ""
    visual_query: str = ""
    still_path: str = ""
    still_url: str = ""
    audio_path: str = ""
    asset_path: str = ""
    asset_source: str = ""
    locked: bool = False
    # Bumped each time the scene is repaired. It feeds the clip cache key, so
    # a repair asks for a genuinely new generation instead of handing back the
    # identical cached file the user just rejected.
    attempt: int = 0
    purpose: str = ""
    visual_type: str = ""
    generation_mode: str = "image_to_video"
    quality_score: int = 0

    @property
    def variant(self) -> int:
        return self.index + self.attempt * 1009

    @property
    def has_asset(self) -> bool:
        return bool(self.asset_path) and Path(self.asset_path).exists()

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_path) and Path(self.audio_path).exists()

    @property
    def has_still(self) -> bool:
        return bool(self.still_path) and Path(self.still_path).exists()


@dataclass
class Job:
    directory: Path
    url: str = ""
    options: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    style: dict = field(default_factory=dict)
    profile: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    content_dna: dict = field(default_factory=dict)
    creative_strategy: dict = field(default_factory=dict)
    style_bible: dict = field(default_factory=dict)
    qa_report: dict = field(default_factory=dict)
    scenes: list[SceneRecord] = field(default_factory=list)
    stop: dict = field(default_factory=dict)
    video_path: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    # ---- persistence ------------------------------------------------------

    @property
    def id(self) -> str:
        return self.directory.name

    @property
    def file(self) -> Path:
        return self.directory / JOB_FILE

    @classmethod
    def load(cls, directory: str | Path) -> "Job | None":
        directory = Path(directory)
        path = directory / JOB_FILE
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if raw.get("version") != SCHEMA_VERSION:
            return None

        job = cls(
            directory=directory,
            url=raw.get("url", ""),
            options=raw.get("options") or {},
            stages=raw.get("stages") or {},
            source=raw.get("source") or {},
            style=raw.get("style") or {},
            profile=raw.get("profile") or {},
            classification=raw.get("classification") or {},
            content_dna=raw.get("content_dna") or {},
            creative_strategy=raw.get("creative_strategy") or {},
            style_bible=raw.get("style_bible") or {},
            qa_report=raw.get("qa_report") or {},
            stop=raw.get("stop") or {},
            video_path=raw.get("video_path", ""),
            created=raw.get("created") or time.time(),
            updated=raw.get("updated") or time.time(),
        )
        # Safely construct SceneRecords
        known_keys = {
            "index", "start", "duration", "narration", "narration_en",
            "visual_query", "still_path", "still_url", "audio_path",
            "asset_path", "asset_source", "locked", "attempt",
            "purpose", "visual_type", "generation_mode", "quality_score",
        }
        job.scenes = [
            SceneRecord(**{k: v for k, v in s.items() if k in known_keys})
            for s in (raw.get("scenes") or [])
        ]
        return job

    def save(self) -> Path:
        self.updated = time.time()
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SCHEMA_VERSION,
            "url": self.url,
            "options": self.options,
            "stages": self.stages,
            "source": self.source,
            "style": self.style,
            "profile": self.profile,
            "classification": self.classification,
            "content_dna": self.content_dna,
            "creative_strategy": self.creative_strategy,
            "style_bible": self.style_bible,
            "qa_report": self.qa_report,
            "scenes": [asdict(s) for s in self.scenes],
            "stop": self.stop,
            "video_path": self.video_path,
            "created": self.created,
            "updated": self.updated,
        }
        temp = self.file.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.file)
        return self.file

    # ---- stage tracking ---------------------------------------------------

    def done(self, stage: str) -> bool:
        return bool((self.stages.get(stage) or {}).get("done"))

    def mark(self, stage: str, **detail) -> None:
        self.stages[stage] = {"done": True, "at": time.time(), **detail}
        self.save()

    def mark_stop(self, reason: str, detail: str = "", **extra) -> None:
        """Record why this run ended, for whoever decides what happens next.

        `reason` must come from STOP_REASONS; an unknown one is stored as
        "error" rather than silently becoming a value no reader handles.
        """
        if reason not in STOP_REASONS:
            extra["given_reason"] = reason
            reason = "error"
        self.stop = {"reason": reason, "detail": str(detail)[:400],
                     "at": time.time(), **extra}
        self.save()

    def clear_stop(self) -> None:
        """Forget the last run's ending, because a new one is starting."""
        if self.stop:
            self.stop = {}
            self.save()

    def invalidate(self, stage: str) -> None:
        """Clear a stage and everything after it — later work depended on it."""
        if stage not in STAGES:
            return
        for name in STAGES[STAGES.index(stage):]:
            self.stages.pop(name, None)

    def resume_from(self) -> str:
        """The first stage that still needs doing."""
        for stage in STAGES:
            if not self.done(stage):
                return stage
        return "complete"

    def matches(self, url: str, options: dict) -> bool:
        """Whether saved work belongs to the run being asked for.

        Changing the aspect ratio or the mode invalidates the clips already
        generated, so those count as a different job rather than a resume.
        """
        if self.url != url:
            return False
        significant = ("aspect", "mode", "allow_stock", "prefer_ai", "translate")
        return all(self.options.get(k) == options.get(k) for k in significant)

    # ---- scenes -----------------------------------------------------------

    def scene(self, index: int) -> SceneRecord | None:
        return next((s for s in self.scenes if s.index == index), None)

    def set_scenes(self, scenes: list[SceneRecord]) -> None:
        self.scenes = scenes

    def unresolved(self) -> list[SceneRecord]:
        """Scenes still missing a clip on disk."""
        return [s for s in self.scenes if not s.has_asset]

    def clear_assets(self, indices) -> list[int]:
        """Forget the clips for these scenes so they will be made again.

        Locked scenes are left alone; locking exists so that a scene you are
        happy with survives a repair pass aimed at its neighbours.
        """
        cleared = []
        for index in indices:
            record = self.scene(index)
            if record is None or record.locked:
                continue
            record.asset_path = ""
            record.asset_source = ""
            # The still is the clip's anchor, so a repair replaces both.
            record.still_path = ""
            record.still_url = ""
            record.attempt += 1
            cleared.append(index)
        if cleared:
            self.invalidate("stills")
            self.save()
        return cleared

    def summary(self) -> dict:
        """Compact state for a progress display."""
        return {
            "url": self.url,
            "title": self.source.get("title", ""),
            "stage": self.resume_from(),
            "stages": {s: self.done(s) for s in STAGES},
            "scenes": len(self.scenes),
            "scenes_with_clips": sum(1 for s in self.scenes if s.has_asset),
            "scenes_with_stills": sum(1 for s in self.scenes if s.has_still),
            "locked": sum(1 for s in self.scenes if s.locked),
            "video": self.video_path,
            "stop": self.stop,
            "updated": self.updated,
        }


def open_job(directory: str | Path, url: str, options: dict,
             fresh: bool = False) -> Job:
    """Load a matching job to continue, or start a new one."""
    directory = Path(directory)
    if not fresh:
        existing = Job.load(directory)
        if existing and existing.matches(url, options):
            # The previous ending belongs to the previous attempt. Leaving it
            # in place would let a finished run still read as quota-stopped.
            existing.clear_stop()
            return existing
    return Job(directory=directory, url=url, options=options)

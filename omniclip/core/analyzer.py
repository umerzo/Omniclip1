"""Source Analyzer.

Orchestrates complete multi-modal analysis of the input video:
metadata, audio signals, shot cuts, frame thumbnails, content classification,
and Content DNA extraction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .classifier import ContentClassification, ContentClassifier
from .dna import ContentDNA, extract_content_dna
from .router import get_router
from .scraper import SourceVideo, fetch_metadata, scrape
from .transcriber import make_transcriber
from ..utils.config import load_settings, resolve_path


@dataclass
class SourceAnalysisReport:
    url: str
    metadata: dict[str, Any] = field(default_factory=dict)
    classification: ContentClassification | None = None
    content_dna: ContentDNA | None = None
    shots_summary: str = ""
    transcript: str = ""
    duration_seconds: float = 0.0
    aspect_ratio: str = "16:9"

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "metadata": self.metadata,
            "classification": self.classification.to_dict() if self.classification else None,
            "content_dna": self.content_dna.to_dict() if self.content_dna else None,
            "shots_summary": self.shots_summary,
            "transcript": self.transcript,
            "duration_seconds": self.duration_seconds,
            "aspect_ratio": self.aspect_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceAnalysisReport:
        return cls(
            url=data.get("url", ""),
            metadata=data.get("metadata") or {},
            classification=ContentClassification.from_dict(data["classification"]) if data.get("classification") else None,
            content_dna=ContentDNA.from_dict(data["content_dna"]) if data.get("content_dna") else None,
            shots_summary=data.get("shots_summary", ""),
            transcript=data.get("transcript", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            aspect_ratio=data.get("aspect_ratio", "16:9"),
        )


class SourceAnalyzer:
    """Performs deep end-to-end analysis of source content."""

    def __init__(self, cfg: dict | None = None, router=None):
        self.cfg = cfg or load_settings()
        self.router = router or get_router()
        self.classifier = ContentClassifier(self.router)

    def analyze(
        self,
        url: str,
        user_target: str | None = None,
        shots_summary: str = "",
        progress=None,
    ) -> SourceAnalysisReport:
        """Run metadata extraction, transcript fetching, classification, and Content DNA."""
        if progress:
            progress("Fetching metadata", 1, 4)

        meta = fetch_metadata(url)
        duration = float(meta.get("duration") or 0.0)
        title = meta.get("title") or "Untitled"
        description = meta.get("description") or ""
        width = int(meta.get("width") or 1920)
        height = int(meta.get("height") or 1080)
        aspect = "9:16" if height > width else ("1:1" if width == height else "16:9")

        if progress:
            progress("Extracting transcript", 2, 4)

        transcript = ""
        try:
            transcriber = make_transcriber(
                api_key=self.cfg["ingest"]["whisper_api_key"],
                base_url=self.cfg["ingest"]["whisper_base_url"],
                cache_dir=resolve_path(self.cfg["ingest"]["cache_dir"]) / "audio",
                model=self.cfg["ingest"]["whisper_model"],
            )
            scraped = scrape(url, languages=self.cfg["ingest"]["languages"], transcriber=transcriber)
            if scraped and scraped.transcript:
                transcript = scraped.transcript
        except Exception as exc:
            print(f"Transcript extraction fallback: {exc}")

        if progress:
            progress("Classifying content format", 3, 4)

        classification = self.classifier.classify(
            title=title,
            description=description,
            duration=duration,
            aspect_ratio=aspect,
            transcript=transcript,
            shots_summary=shots_summary,
            user_target=user_target,
        )

        if progress:
            progress("Extracting Content DNA", 4, 4)

        dna = extract_content_dna(
            title=title,
            transcript=transcript,
            classification=classification,
            shots_summary=shots_summary,
            duration=duration,
            router=self.router,
        )

        return SourceAnalysisReport(
            url=url,
            metadata=meta,
            classification=classification,
            content_dna=dna,
            shots_summary=shots_summary,
            transcript=transcript,
            duration_seconds=duration,
            aspect_ratio=aspect,
        )

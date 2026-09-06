"""Content Classifier.

Analyzes source signals (metadata, transcript, shot cuts, audio, visual features)
to identify the video's creative format and genre taxonomy.
Prevents treating non-narrative videos (cars, tutorials, products, viral clips)
as fictional narrative stories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .router import get_router

PRIMARY_TYPES = (
    "cinematic_story",
    "short_story",
    "documentary",
    "explainer",
    "tutorial",
    "commentary",
    "talking_head",
    "podcast_clip",
    "product_showcase",
    "car_showcase",
    "automotive_cinematic",
    "travel",
    "food",
    "fitness",
    "sports",
    "gaming",
    "educational",
    "news",
    "motivational",
    "listicle",
    "meme",
    "comedy",
    "reaction",
    "compilation",
    "transformation",
    "before_after",
    "satisfying",
    "lifestyle",
    "fashion",
    "technology",
    "science",
    "history",
    "finance",
    "business",
    "advertisement",
    "music",
    "montage",
    "vlog",
    "other",
)


@dataclass
class ContentClassification:
    primary_type: str
    secondary_types: list[str] = field(default_factory=list)
    format: str = "short"  # "short" | "long_form"
    visual_dependency: str = "high"  # "low" | "medium" | "high"
    narration_dependency: str = "medium"
    dialogue_dependency: str = "low"
    story_dependency: str = "low"
    motion_dependency: str = "medium"
    recommended_generation_strategy: str = ""
    confidence: float = 0.9
    summary_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentClassification:
        return cls(
            primary_type=data.get("primary_type", "cinematic_story"),
            secondary_types=data.get("secondary_types") or [],
            format=data.get("format", "short"),
            visual_dependency=data.get("visual_dependency", "high"),
            narration_dependency=data.get("narration_dependency", "medium"),
            dialogue_dependency=data.get("dialogue_dependency", "low"),
            story_dependency=data.get("story_dependency", "low"),
            motion_dependency=data.get("motion_dependency", "medium"),
            recommended_generation_strategy=data.get("recommended_generation_strategy", ""),
            confidence=float(data.get("confidence", 0.9)),
            summary_rationale=data.get("summary_rationale", ""),
        )


class ContentClassifier:
    """Classifies content into structured format and creative DNA."""

    def __init__(self, router=None):
        self.router = router or get_router()

    def classify(
        self,
        title: str,
        description: str,
        duration: float,
        aspect_ratio: str,
        transcript: str,
        shots_summary: str = "",
        user_target: str | None = None,
    ) -> ContentClassification:
        """Analyze source signals using Gemini 3.8 Flash to classify creative format."""
        format_hint = "short" if (duration <= 65 or aspect_ratio == "9:16") else "long_form"
        if user_target:
            if "short" in user_target.lower() or "reel" in user_target.lower() or "tiktok" in user_target.lower():
                format_hint = "short"
            elif "long" in user_target.lower():
                format_hint = "long_form"

        system = (
            "You are an elite video content analyst and YouTube algorithm specialist. "
            "Your task is to classify a video's exact CREATIVE FORMAT and PRODUCTION GENRE. "
            "DO NOT assume every video is a fictional narrative story. "
            "A car video is an automotive showcase or review, not a character story. "
            "A cooking video is culinary demonstration. "
            "A viral Short is often a pattern interrupt, visual gag, or quick explanation. "
            "Return valid JSON adhering strictly to the requested schema."
        )

        prompt = f"""Analyze this source video and classify its content format and production requirements.

VIDEO SIGNALS:
- Title: {title}
- Description: {description[:1000]}
- Duration: {duration:.1f} seconds
- Aspect Ratio: {aspect_ratio}
- Format Hint: {format_hint}
- Shots / Visual Summary: {shots_summary[:1200] if shots_summary else 'No visual summary yet'}
- Transcript:
{transcript[:3000] if transcript else '[No spoken transcript - visual or musical video]'}

ALLOWED PRIMARY/SECONDARY TYPES:
{', '.join(PRIMARY_TYPES)}

Output a JSON object with:
{{
  "primary_type": "<one from allowed types>",
  "secondary_types": ["<1-3 secondary types>"],
  "format": "short" or "long_form",
  "visual_dependency": "low" | "medium" | "high",
  "narration_dependency": "low" | "medium" | "high",
  "dialogue_dependency": "low" | "medium" | "high",
  "story_dependency": "low" | "medium" | "high",
  "motion_dependency": "low" | "medium" | "high",
  "recommended_generation_strategy": "<concise statement of how an original video in this format should be structured>",
  "confidence": <float 0.0 to 1.0>,
  "summary_rationale": "<1-2 sentences explaining why this format applies>"
}}
"""

        try:
            res = self.router.run_prompt(
                prompt=prompt,
                system=system,
                high_reasoning=True,
                response_json=True,
            )
            if isinstance(res, dict):
                # Ensure primary_type is valid
                primary = res.get("primary_type", "").lower().strip()
                if primary not in PRIMARY_TYPES:
                    # Best match fallback
                    for pt in PRIMARY_TYPES:
                        if pt in primary or primary in pt:
                            res["primary_type"] = pt
                            break
                    else:
                        res["primary_type"] = "explainer" if format_hint == "long_form" else "short_story"
                return ContentClassification.from_dict(res)
        except Exception:
            pass

        # Deterministic Python heuristic fallback
        return self._heuristic_fallback(title, transcript, duration, aspect_ratio, format_hint)

    def _heuristic_fallback(
        self,
        title: str,
        transcript: str,
        duration: float,
        aspect: str,
        format_hint: str,
    ) -> ContentClassification:
        text = f"{title} {transcript}".lower()
        if any(w in text for w in ("car", "supercar", "exhaust", "bmw", "porsche", "ferrari", "lamborghini", "driving")):
            primary = "automotive_cinematic"
            strategy = "automotive_showcase"
        elif any(w in text for w in ("recipe", "cook", "kitchen", "bake", "taste", "chef", "delicious")):
            primary = "food"
            strategy = "culinary_demonstration"
        elif any(w in text for w in ("how to", "tutorial", "step by step", "guide", "learn")):
            primary = "tutorial"
            strategy = "step_by_step"
        elif any(w in text for w in ("unboxing", "review", "specs", "iphone", "gadget", "product")):
            primary = "product_showcase"
            strategy = "product_breakdown"
        elif any(w in text for w in ("history", "documentary", "war", "century", "empire", "explained")):
            primary = "documentary"
            strategy = "documentary_investigation"
        elif duration <= 60 or format_hint == "short":
            primary = "short_story"
            strategy = "rapid_viral_short"
        else:
            primary = "cinematic_story"
            strategy = "narrative_story"

        return ContentClassification(
            primary_type=primary,
            secondary_types=["montage"],
            format=format_hint,
            visual_dependency="high",
            narration_dependency="high" if len(transcript.split()) > 20 else "low",
            dialogue_dependency="low",
            story_dependency="high" if "story" in primary else "low",
            motion_dependency="high" if primary == "automotive_cinematic" else "medium",
            recommended_generation_strategy=strategy,
            confidence=0.75,
            summary_rationale="Classified via heuristic keyword matching fallback.",
        )

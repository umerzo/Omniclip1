"""Content DNA Extractor.

Extracts the structural creative blueprint of why a video works:
pacing, retention mechanics, emotional arc, camera grammar, audio design,
and what makes it engaging — without copying its specific scenes or characters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .classifier import ContentClassification
from .router import get_router


@dataclass
class ContentDNA:
    hook: dict[str, Any] = field(default_factory=dict)
    audience: dict[str, Any] = field(default_factory=dict)
    promise: dict[str, Any] = field(default_factory=dict)
    narrative_structure: dict[str, Any] = field(default_factory=dict)
    pacing: dict[str, Any] = field(default_factory=dict)
    retention_mechanisms: list[dict[str, Any]] = field(default_factory=list)
    emotional_arc: list[str] = field(default_factory=list)
    visual_language: dict[str, Any] = field(default_factory=dict)
    camera_language: dict[str, Any] = field(default_factory=dict)
    editing_language: dict[str, Any] = field(default_factory=dict)
    sound_language: dict[str, Any] = field(default_factory=dict)
    narration_style: dict[str, Any] = field(default_factory=dict)
    cta_strategy: dict[str, Any] = field(default_factory=dict)
    unique_elements: list[str] = field(default_factory=list)
    things_to_avoid: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentDNA:
        raw = data.get("content_dna", data)
        return cls(
            hook=raw.get("hook") or {},
            audience=raw.get("audience") or {},
            promise=raw.get("promise") or {},
            narrative_structure=raw.get("narrative_structure") or {},
            pacing=raw.get("pacing") or {},
            retention_mechanisms=raw.get("retention_mechanisms") or [],
            emotional_arc=raw.get("emotional_arc") or [],
            visual_language=raw.get("visual_language") or {},
            camera_language=raw.get("camera_language") or {},
            editing_language=raw.get("editing_language") or {},
            sound_language=raw.get("sound_language") or {},
            narration_style=raw.get("narration_style") or {},
            cta_strategy=raw.get("cta_strategy") or {},
            unique_elements=raw.get("unique_elements") or [],
            things_to_avoid=raw.get("things_to_avoid") or [],
        )


def extract_content_dna(
    title: str,
    transcript: str,
    classification: ContentClassification,
    shots_summary: str = "",
    duration: float = 0.0,
    router=None,
) -> ContentDNA:
    """Extract structural creative blueprint using Gemini 3.8 Flash."""
    router = router or get_router()

    system = (
        "You are an expert creative director, video strategist, and retention engineer. "
        "Analyze this source video to extract its 'Content DNA' — the underlying creative mechanics, "
        "pacing rhythm, visual grammar, and psychology that make it work. "
        "This is NOT a transcription or summary. It is an architectural creative blueprint "
        "used to create an ORIGINAL new video with similar engagement power."
    )

    prompt = f"""Extract the Content DNA for this video.

SOURCE OVERVIEW:
- Title: {title}
- Duration: {duration:.1f}s
- Classified Format: {classification.format} ({classification.primary_type})
- Visual Dependency: {classification.visual_dependency}
- Strategy: {classification.recommended_generation_strategy}
- Visual Summary: {shots_summary[:1200] if shots_summary else 'Derived from frames'}
- Transcript Excerpt:
{transcript[:3500] if transcript else '[No dialogue/narration]'}

Return a JSON object with this exact structure:
{{
  "content_dna": {{
    "hook": {{
      "mechanism": "Curiosity gap / Pattern interrupt / Bold visual statement",
      "timing_seconds": 2.5,
      "hook_formula": "Why the opening grabs attention immediately"
    }},
    "audience": {{
      "target": "Who is watching this",
      "desire": "What the viewer wants to see or learn"
    }},
    "promise": {{
      "core_value": "The implicit promise made to the viewer in the first 5 seconds"
    }},
    "narrative_structure": {{
      "framework": "e.g. 3-beat escalation / problem-solution / hero reveal / process transformation",
      "phases": ["Phase 1: ...", "Phase 2: ...", "Phase 3: ..."]
    }},
    "pacing": {{
      "cut_frequency_seconds": "average shot length",
      "rhythm": "fast / deliberate / dynamic / slow-build",
      "words_per_minute": 140
    }},
    "retention_mechanisms": [
      {{"technique": "Open loop / Micro-reveal / Audio-visual punch", "purpose": "Keep viewer watching"}}
    ],
    "emotional_arc": [
      "Curiosity", "Anticipation", "Surprise", "Satisfaction"
    ],
    "visual_language": {{
      "palette": "Color tone description",
      "lighting": "Natural / Studio / High-contrast / Cinematic moody",
      "texture": "Clean / Gritty / Atmospheric / Macro-detailed"
    }},
    "camera_language": {{
      "primary_angles": "Low tracking / Close-up macro / Wide dynamic",
      "movement": "Push-in / Parallax pan / Whip-pan / Steady glide"
    }},
    "editing_language": {{
      "transition_style": "Hard cuts / Match cuts / Whip / Seamless flow",
      "text_overlay_presence": true
    }},
    "sound_language": {{
      "narration_presence": true,
      "music_mood": "Driving synth / Ambient chill / Epic orchestral / Punchy beats",
      "sfx_importance": "high / medium / low"
    }},
    "narration_style": {{
      "tone": "Authoritative / Enthusiastic / Calm / Conversational / Dramatic",
      "voice_speed": "+0%"
    }},
    "cta_strategy": {{
      "placement": "end / subtle / none",
      "style": "Clean follow / loop"
    }},
    "unique_elements": [
      "Key creative elements that set this piece apart"
    ],
    "things_to_avoid": [
      "Things that would ruin this video type (e.g. generic AI walking people, boring static shots, robotic voice)"
    ]
  }}
}}
"""

    try:
        res = router.run_prompt(prompt, system=system, high_reasoning=True, response_json=True)
        if isinstance(res, dict):
            return ContentDNA.from_dict(res)
    except Exception:
        pass

    # Heuristic fallback
    return ContentDNA(
        hook={"mechanism": "Fast visual statement", "timing_seconds": 2.0, "hook_formula": "Showcase subject immediately"},
        audience={"target": "General online viewers", "desire": "Engaging, high quality content"},
        promise={"core_value": "Immediate visual engagement"},
        narrative_structure={"framework": "Introduction -> Escalation -> Payoff", "phases": ["Hook", "Core demonstration", "Resolution"]},
        pacing={"cut_frequency_seconds": "3.5s", "rhythm": "dynamic", "words_per_minute": 145},
        retention_mechanisms=[{"technique": "Visual progression", "purpose": "Maintain viewer focus"}],
        emotional_arc=["Curiosity", "Engagement", "Satisfaction"],
        visual_language={"palette": "Vibrant and clear", "lighting": "Natural cinematic", "texture": "Crisp"},
        camera_language={"primary_angles": "Dynamic three-quarter", "movement": "Smooth tracking"},
        editing_language={"transition_style": "Crisp cuts", "text_overlay_presence": True},
        sound_language={"narration_presence": bool(transcript), "music_mood": "Upbeat modern", "sfx_importance": "medium"},
        narration_style={"tone": "Clear, engaging and natural", "voice_speed": "+0%"},
        cta_strategy={"placement": "end", "style": "Brief outro"},
        unique_elements=["Fast hook", "High visual variety"],
        things_to_avoid=["Generic AI walking characters", "Repetitive angles", "Slow opening"],
    )

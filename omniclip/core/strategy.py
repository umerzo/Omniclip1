"""Creative Strategist & Originality Engine.

Explicitly separates WHAT TO PRESERVE (high-level format, pacing, retention mechanics,
audience psychology) from WHAT MUST CHANGE (characters, specific story, dialogue,
exact scenes, locations, visual assets).

Develops an ORIGINAL concept and production strategy rather than copying or merely
rewriting the source transcript.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .classifier import ContentClassification
from .dna import ContentDNA
from .router import get_router


@dataclass
class CreativeStrategy:
    new_concept: str
    target_audience: str
    core_promise: str
    hook_strategy: str
    story_strategy: str
    visual_strategy: str
    pacing_strategy: str
    sound_strategy: str
    ending_strategy: str
    originality_notes: list[str] = field(default_factory=list)
    what_to_preserve: list[str] = field(default_factory=list)
    what_must_change: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreativeStrategy:
        raw = data.get("creative_strategy", data)
        return cls(
            new_concept=raw.get("new_concept", ""),
            target_audience=raw.get("target_audience", ""),
            core_promise=raw.get("core_promise", ""),
            hook_strategy=raw.get("hook_strategy", ""),
            story_strategy=raw.get("story_strategy", ""),
            visual_strategy=raw.get("visual_strategy", ""),
            pacing_strategy=raw.get("pacing_strategy", ""),
            sound_strategy=raw.get("sound_strategy", ""),
            ending_strategy=raw.get("ending_strategy", ""),
            originality_notes=raw.get("originality_notes") or [],
            what_to_preserve=raw.get("what_to_preserve") or [],
            what_must_change=raw.get("what_must_change") or [],
        )


class CreativeStrategist:
    """Develops a creative strategy for a completely original piece of content."""

    def __init__(self, router=None):
        self.router = router or get_router()

    def develop_strategy(
        self,
        dna: ContentDNA,
        classification: ContentClassification,
        source_title: str,
        source_transcript: str = "",
        user_idea: str | None = None,
        user_instructions: str | None = None,
        target_platform: str | None = None,
        cast: list[dict] | None = None,
        shots: list | None = None,
        style_phrase: str = "",
    ) -> CreativeStrategy:
        """Formulate a creative blueprint for an original video or high-fidelity remake."""
        is_remake = not bool(user_idea and user_idea.strip())

        cast_desc = "\n".join(f"- {c.get('id', c.get('name'))}: {c.get('appearance')}" for c in (cast or []))
        shots_desc = "\n".join(f"- Shot {i}: {getattr(s, 'description', str(s))[:160]}" for i, s in enumerate(shots or []))

        if is_remake and (cast or shots):
            system = (
                "You are an executive creative director and animation supervisor. "
                "Your mission: Recreate and elevate the source video into a high-production remake. "
                "PRESERVE THE DONOR FILM'S CHARACTERS, SETTING, AND VISUAL GAG/STORYLINE. "
                "Do NOT replace the characters with unrelated animals or fantasy creatures. "
                "If the original features a boy and a goat, the remake MUST feature that boy and that goat! "
                "Elevate the cinematography, camera motion, and physical comedy timing while keeping the same core cast and story."
            )
        else:
            system = (
                "You are an award-winning creative director and digital media strategist. "
                "Your mission: Formulate a compelling, ORIGINAL creative strategy that uses the "
                "proven structural DNA of the source format while creating a 100% NEW concept. "
                "Ensure the concept is tailored to the content classification and platform."
            )

        prompt = f"""Formulate the Creative Strategy for this video production.

SOURCE CLASSIFICATION:
- Primary Type: {classification.primary_type}
- Format: {classification.format} ({target_platform or 'Standard'})
- Generation Strategy: {classification.recommended_generation_strategy}

DONOR WORLD & CHARACTERS (PRESERVE THESE WHEN REMAKING):
- Visual Style: {style_phrase or 'Preserve donor visual style'}
- Cast / Characters:
{cast_desc if cast_desc else 'No specific characters detected'}
- Visual Story Sequence:
{shots_desc if shots_desc else 'No specific shot breakdown'}

SOURCE CONTENT DNA:
- Hook Mechanism: {dna.hook.get('mechanism', 'Attention grabber')}
- Core Promise: {dna.promise.get('core_value', 'Value delivery')}
- Pacing & Rhythm: {dna.pacing.get('rhythm', 'Dynamic')}
- Narrative Framework: {dna.narrative_structure.get('framework', 'Escalation')}
- Things to Avoid: {', '.join(dna.things_to_avoid) if dna.things_to_avoid else 'AI slop, static shots'}

USER SPECIFIC INPUTS:
- User Idea / Topic: {user_idea or ('Rebuild and elevate the source video with the same characters and story' if is_remake else 'Original concept')}
- User Instructions: {user_instructions or 'Maximize visual production value and preserve character continuity'}
- Target Platform: {target_platform or classification.format}

TASK:
1. Define WHAT TO PRESERVE ({"The exact characters (" + ", ".join(c.get('id', c.get('name', '')) for c in (cast or [])) + "), setting, and comedic story beats" if is_remake and cast else "The structural pacing, curiosity hooks, visual rhythm"}).
2. Define WHAT TO ELEVATE / CHANGE (enhanced cinematography, dynamic camera movement, crisp textures).
3. Create the Creative Strategy JSON:
{{
  "creative_strategy": {{
    "new_concept": "Detailed description of the video concept (keeping the original characters and story if remaking)",
    "target_audience": "Specific audience target",
    "core_promise": "What the viewer gets from watching this video",
    "hook_strategy": "Specific hook to capture attention in the first 0-3 seconds",
    "story_strategy": "The narrative or thematic arc",
    "visual_strategy": "The cinematography, camera angles, textures, and lighting",
    "pacing_strategy": "Cut rhythm and timing mechanism",
    "sound_strategy": "Audio design and music mood",
    "ending_strategy": "Payoff or punchline",
    "originality_notes": [
      "Cinematographic and visual improvements over the donor footage"
    ],
    "what_to_preserve": [
      {"'Characters', 'Core gag', 'Visual style'" if is_remake and cast else "'Pacing beat', 'Hook formula'"}
    ],
    "what_must_change": [
      "Enhanced camera tracking, cleaner lighting, sharper textures"
    ]
  }}
}}
"""

        try:
            res = self.router.run_prompt(prompt, system=system, high_reasoning=True, response_json=True)
            if isinstance(res, dict):
                return CreativeStrategy.from_dict(res)
        except Exception as exc:
            print(f"CreativeStrategist error: {exc}")

        # Fallback strategy
        return CreativeStrategy(
            new_concept=f"Original {classification.primary_type.replace('_', ' ').title()} exploration inspired by {source_title}",
            target_audience="Engaged digital viewers",
            core_promise="Fast-paced, visually stunning insight",
            hook_strategy="Bold opening statement and immediate high-motion visual",
            story_strategy="3-phase escalation leading to a rewarding conclusion",
            visual_strategy="Dynamic three-quarter tracking with natural dramatic lighting",
            pacing_strategy="Quick cuts every 3-4s with punchy voiceover",
            sound_strategy="Clean narration backed by driving modern instrumental",
            ending_strategy="Memorable climax followed by clean CTA",
            originality_notes=["Completely new script and visual subject", "Zero reused dialogue"],
            what_to_preserve=["Cut frequency", "Retention hook style"],
            what_must_change=["Subject matter", "Script", "Visual assets"],
        )

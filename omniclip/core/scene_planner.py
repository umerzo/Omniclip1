"""Scene Planner & Production Architecture.

Transforms script beats into concrete production units.
Enforces:
1. Mandatory scene purpose (hook, establish, explain, demonstrate, escalate, etc.)
2. Style Bible with appropriate continuity (vehicles for car videos, ingredients for cooking,
   characters ONLY when relevant)
3. Intelligent selection of generation mode (image_to_video vs text_to_video)
4. Genre-specific visual priorities (body lines and wheels for cars, food textures for cooking,
   not random people walking around).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .classifier import ContentClassification
from .dna import ContentDNA
from .router import get_router
from .script_engine import GeneratedScript
from .strategy import CreativeStrategy

VALID_PURPOSES = (
    "hook",
    "establish",
    "explain",
    "demonstrate",
    "escalate",
    "emotional_beat",
    "evidence",
    "transition",
    "payoff",
    "cta",
    "visual_breathing_room",
)


@dataclass
class StyleBible:
    visual_style: str
    cinematography: str
    color_language: str
    lighting: str
    lens_language: str
    editing_style: str
    characters: list[dict[str, Any]] = field(default_factory=list)
    locations: list[dict[str, Any]] = field(default_factory=list)
    important_objects: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleBible:
        return cls(
            visual_style=data.get("visual_style", "Cinematic professional"),
            cinematography=data.get("cinematography", "Smooth tracking shots, deliberate composition"),
            color_language=data.get("color_language", "Natural balanced palette with rich contrast"),
            lighting=data.get("lighting", "Atmospheric natural lighting"),
            lens_language=data.get("lens_language", "35mm and 50mm shallow depth of field"),
            editing_style=data.get("editing_style", "Dynamic pacing with seamless visual flow"),
            characters=data.get("characters") or [],
            locations=data.get("locations") or [],
            important_objects=data.get("important_objects") or [],
        )


@dataclass
class ProductionScene:
    scene_id: int
    purpose: str
    duration: float
    narration: str
    narration_en: str = ""
    dialogue: str = ""
    visual_type: str = "cinematic_shot"
    visual_prompt: str = ""
    camera: str = ""
    motion: str = ""
    environment: str = ""
    lighting: str = ""
    sound_design: str = ""
    on_screen_text: str = ""
    transition: str = "cut"
    continuity_requirements: list[str] = field(default_factory=list)
    generation_mode: str = "image_to_video"  # "image_to_video" | "text_to_video"
    reference_assets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProductionScene:
        purpose = data.get("purpose", "demonstrate").lower()
        if purpose not in VALID_PURPOSES:
            purpose = "demonstrate"
        return cls(
            scene_id=int(data.get("scene_id", 0)),
            purpose=purpose,
            duration=float(data.get("duration", 4.0)),
            narration=data.get("narration", ""),
            narration_en=data.get("narration_en", ""),
            dialogue=data.get("dialogue", ""),
            visual_type=data.get("visual_type", "cinematic_shot"),
            visual_prompt=data.get("visual_prompt", ""),
            camera=data.get("camera", ""),
            motion=data.get("motion", ""),
            environment=data.get("environment", ""),
            lighting=data.get("lighting", ""),
            sound_design=data.get("sound_design", ""),
            on_screen_text=data.get("on_screen_text", ""),
            transition=data.get("transition", "cut"),
            continuity_requirements=data.get("continuity_requirements") or [],
            generation_mode=data.get("generation_mode", "image_to_video"),
            reference_assets=data.get("reference_assets") or [],
        )


@dataclass
class ProductionPlan:
    title: str
    style_bible: StyleBible
    scenes: list[ProductionScene] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.scenes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "style_bible": self.style_bible.to_dict(),
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProductionPlan:
        return cls(
            title=data.get("title", "Production Plan"),
            style_bible=StyleBible.from_dict(data.get("style_bible") or {}),
            scenes=[ProductionScene.from_dict(s) for s in data.get("scenes") or []],
        )


class ScenePlanner:
    """Plans individual production scenes with mandatory purpose and style bible."""

    def __init__(self, router=None):
        self.router = router or get_router()

    def plan(
        self,
        script: GeneratedScript,
        strategy: CreativeStrategy,
        classification: ContentClassification,
        dna: ContentDNA,
        cast: list[dict] | None = None,
        shots: list | None = None,
        style_phrase: str = "",
    ) -> ProductionPlan:
        """Create a complete production plan with style bible and scene specs."""
        genre = classification.primary_type
        is_character_centric = genre in ("cinematic_story", "short_story", "documentary", "talking_head", "vlog")

        cast_desc = "\n".join(f"- {c.get('id', c.get('name'))}: {c.get('appearance')}" for c in (cast or []))
        shots_desc = "\n".join(f"- Shot {i}: {getattr(s, 'description', str(s))[:160]}" for i, s in enumerate(shots or []))

        system = (
            "You are an executive video producer and cinematographer. "
            "Convert this script into a high-level Production Plan with a Style Bible and detailed Scene Specs. "
            "STRICT RULES:\n"
            "1. Every scene MUST have an explicit 'purpose' from: hook, establish, explain, demonstrate, "
            "escalate, emotional_beat, evidence, transition, payoff, cta, visual_breathing_room.\n"
            "2. If recurring characters from the donor are provided, PRESERVE THEM EXACTLY in the Style Bible and scene prompts. "
            "Do NOT invent unrelated new animals or characters. If the original has a boy and a goat, the remake MUST feature that boy and that goat!\n"
            "3. Choose 'generation_mode': use 'image_to_video' when subject consistency or composition matters; "
            "use 'text_to_video' for fast abstract B-roll or generic environmental scenes.\n"
            "4. Make visual prompts concrete and cinematic: specify exact framing, subject action, camera movement, and lighting."
        )

        beats_json = json.dumps([b.to_dict() for b in script.beats], indent=2)

        prompt = f"""Plan the production scenes and Style Bible for this video.

VIDEO SPECIFICATION:
- Title: {script.title}
- Format: {script.format}
- Primary Genre: {genre}
- Visual Strategy: {strategy.visual_strategy}
- Visual Style to Preserve: {style_phrase or 'Preserve donor visual style'}

CHARACTERS TO ENFORCE IN STYLE BIBLE & SCENES:
{cast_desc if cast_desc else 'No specific characters specified'}

REFERENCE SHOT SEQUENCE:
{shots_desc if shots_desc else 'No specific shot breakdown'}

SCRIPT BEATS:
{beats_json}

Output a JSON object with this exact schema:
{{
  "title": "{script.title}",
  "style_bible": {{
    "visual_style": "{style_phrase or 'High-level visual identity'}",
    "cinematography": "Camera rules and lens grammar",
    "color_language": "Color palette and grading notes",
    "lighting": "Lighting setup",
    "lens_language": "Focal lengths (e.g. 24mm wide, 85mm portrait)",
    "editing_style": "Editing and rhythm notes",
    "characters": [
      {{"name": "Character Name", "appearance": "Detailed physical description matching cast above"}}
    ],
    "locations": [
      {{"name": "Main Setting", "description": "Atmosphere and geography"}}
    ],
    "important_objects": [
      {{"name": "Important Object (e.g. Bicycle, Haystack)", "description": "Exact visual specs for continuity"}}
    ]
  }},
  "scenes": [
    {{
      "scene_id": 0,
      "purpose": "hook / establish / explain / demonstrate / escalate / emotional_beat / evidence / transition / payoff / cta / visual_breathing_room",
      "duration": 4.0,
      "narration": "Narration text",
      "dialogue": "",
      "visual_type": "cinematic_shot / b_roll / product_close_up / car_driving / environmental",
      "visual_prompt": "Production prompt describing the scene and featuring the specific characters",
      "camera": "Low tracking / static tripod / dynamic push-in / orbital",
      "motion": "Fast forward momentum / slow subtle pan",
      "environment": "Setting details",
      "lighting": "Golden hour side-lighting",
      "sound_design": "Sound design notes",
      "on_screen_text": "",
      "transition": "cut",
      "continuity_requirements": ["Character appearance continuity"],
      "generation_mode": "image_to_video"
    }}
  ]
}}
"""

        try:
            res = self.router.run_prompt(prompt, system=system, high_reasoning=True, response_json=True)
            if isinstance(res, dict) and res.get("scenes"):
                # If cast was provided, ensure characters in style_bible contains them
                if cast:
                    bible = res.setdefault("style_bible", {})
                    bible_chars = bible.setdefault("characters", [])
                    existing_names = {c.get("name", "").lower() for c in bible_chars}
                    for c in cast:
                        c_id = c.get("id", c.get("name", ""))
                        if c_id.lower() not in existing_names:
                            bible_chars.append({"name": c_id, "appearance": c.get("appearance", "")})
                plan = ProductionPlan.from_dict(res)
                beat_map = {b.index: b for b in script.beats}
                for sc in plan.scenes:
                    if sc.scene_id in beat_map:
                        orig_beat = beat_map[sc.scene_id]
                        if orig_beat.narration.strip():
                            sc.narration = orig_beat.narration
                            sc.duration = max(sc.duration, orig_beat.target_duration)
                return plan
        except Exception as exc:
            print(f"ScenePlanner error: {exc}")

        return self._fallback_plan(script, strategy, classification)

    def _fallback_plan(
        self,
        script: GeneratedScript,
        strategy: CreativeStrategy,
        classification: ContentClassification,
    ) -> ProductionPlan:
        scenes = []
        for b in script.beats:
            purpose = b.role if b.role in VALID_PURPOSES else "demonstrate"
            scenes.append(
                ProductionScene(
                    scene_id=b.index,
                    purpose=purpose,
                    duration=b.target_duration,
                    narration=b.narration,
                    visual_type="cinematic_shot" if "story" in classification.primary_type else "product_close_up",
                    visual_prompt=f"{b.visual_direction}. {strategy.visual_strategy}",
                    camera="Dynamic tracking shot",
                    motion="Fluid motion",
                    environment="Atmospheric setting",
                    lighting="Natural cinematic lighting",
                    sound_design="Subtle ambient background",
                    on_screen_text=b.on_screen_text,
                    generation_mode="image_to_video",
                    continuity_requirements=["Consistent style and lighting"],
                )
            )

        bible = StyleBible(
            visual_style=strategy.visual_strategy,
            cinematography="Smooth cinematic tracking",
            color_language="Clean contrast, cinematic palette",
            lighting="Natural daylight with soft shadows",
            lens_language="50mm prime look",
            editing_style="Fast, dynamic cuts",
            characters=[],
            locations=[{"name": "Primary Environment", "description": "Modern atmospheric space"}],
            important_objects=[{"name": "Hero Subject", "description": "Primary visual focus of the video"}],
        )

        return ProductionPlan(title=script.title, style_bible=bible, scenes=scenes)

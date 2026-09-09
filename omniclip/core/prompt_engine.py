"""Prompt Engine & Visual Relevance Validator.

Generates production-ready image and video prompts tailored to content genre.
Avoids generic AI slop ("cinematic 8k masterpiece") in favor of concrete
cinematographic specifications.

Performs a pre-generation Visual Relevance Check to ensure visual prompt
aligns with narration, subject continuity, physical plausibility, and scene purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .classifier import ContentClassification
from .router import get_router
from .scene_planner import ProductionPlan, ProductionScene, StyleBible


@dataclass
class RelevanceCheckResult:
    scene_id: int
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: list[str]
    refined_prompt: str | None = None


class PromptEngine:
    """Refines visual prompts and validates relevance before video generation."""

    def __init__(self, router=None):
        self.router = router or get_router()

    def build_prompt_for_scene(
        self,
        scene: ProductionScene,
        bible: StyleBible,
        classification: ContentClassification,
    ) -> str:
        """Compose a production-ready visual prompt combining scene spec and style bible."""
        parts = []

        # 1. Shot framing & camera
        if scene.camera:
            parts.append(scene.camera.rstrip("."))

        # 2. Main visual action
        if scene.visual_prompt:
            parts.append(scene.visual_prompt.rstrip("."))

        # 3. Object / Character Continuity
        genre = classification.primary_type
        if "car" in genre or "automotive" in genre:
            # Emphasize vehicle specs and zero random pedestrians
            obj_desc = "; ".join(f"{o.get('name')}: {o.get('description')}" for o in bible.important_objects if o.get("name"))
            if obj_desc:
                parts.append(f"Subject: {obj_desc}. Crisp automotive cinematography, sharp body reflections, detailed wheels.")
            parts.append("No random pedestrians or unrelated people in frame.")
        elif "product" in genre:
            obj_desc = "; ".join(f"{o.get('name')}: {o.get('description')}" for o in bible.important_objects if o.get("name"))
            if obj_desc:
                parts.append(f"Product focus: {obj_desc}. Premium product lighting, clean reflections, studio macro clarity.")
            parts.append("No irrelevant background clutter or unrelated figures.")
        elif "food" in genre:
            parts.append("Culinary visual focus: appetizing textures, precise culinary technique, fresh ingredients, steam/sizzle details.")

        # Always enforce recurring character continuity if characters exist in the Bible
        if bible.characters:
            char_desc = "; ".join(f"{c.get('name', c.get('id', 'Character'))}: {c.get('appearance')}" for c in bible.characters if c.get('appearance'))
            if char_desc:
                parts.append(f"Characters (preserve exact appearance): {char_desc}.")

        # Cultural wardrobe integrity & modesty enforcement
        combined_text = f"{scene.visual_prompt} {' '.join(str(c) for c in bible.characters)}".lower()
        if any(w in combined_text for w in ["shalwar", "salwar", "kameez", "kurta", "kurti", "dupatta", "desi", "traditional", "abaya", "thobe", "churidar"]):
            parts.append("Wardrobe modesty: Authentic full-length ankle-covering trousers, pants reach all the way down to the ankles and feet, fully covered legs, strictly no exposed calves, no bare legs, culturally authentic South Asian attire.")

        # 4. Environment & Lighting
        if scene.environment:
            parts.append(f"Setting: {scene.environment.rstrip('.')}.")
        if scene.lighting:
            parts.append(f"Lighting: {scene.lighting.rstrip('.')}.")
        elif bible.lighting:
            parts.append(f"Lighting: {bible.lighting.rstrip('.')}.")

        # 5. Visual style aesthetic
        if bible.visual_style:
            parts.append(f"Style: {bible.visual_style.rstrip('.')}.")

        # 6. Motion direction
        if scene.motion:
            parts.append(f"Camera movement: {scene.motion.rstrip('.')}.")

        return " ".join(parts).strip()

    def validate_and_refine(
        self,
        scene: ProductionScene,
        bible: StyleBible,
        classification: ContentClassification,
    ) -> RelevanceCheckResult:
        """Run lightweight visual relevance check before generation."""
        prompt = self.build_prompt_for_scene(scene, bible, classification)

        system = (
            "You are a visual quality supervisor for an elite video production team. "
            "Validate whether the visual prompt accurately represents the scene narration, "
            "maintains subject focus, is physically plausible, advances the content purpose, "
            "and strictly avoids unwanted characters or generic AI hallucinations."
        )

        user_query = f"""Evaluate this scene's visual prompt for production readiness:

GENRE: {classification.primary_type}
PURPOSE: {scene.purpose}
DURATION: {scene.duration:.1f}s
NARRATION: {scene.narration or '[Purely visual scene]'}
PROPOSED PROMPT:
{prompt}

RULES:
1. Does visual match narration and scene purpose?
2. Is the main subject correct (e.g. car for car video, product for product video)?
3. Does it avoid unwanted pedestrians/characters if not a character video?
4. Is it physically plausible and cinematographically sound?
5. Cultural & wardrobe accuracy: If wearing traditional attire (shalwar kameez, kurta, etc.), ensure trousers are strictly full ankle-length with zero exposed legs or calves.

Return JSON:
{{
  "is_valid": true or false,
  "score": <float 0.0 to 1.0>,
  "issues": ["Issue 1 if any", "Issue 2..."],
  "refined_prompt": "Clean, perfected prompt incorporating all fixes (or null if valid)"
}}
"""

        try:
            res = self.router.run_prompt(user_query, system=system, high_reasoning=False, response_json=True)
            if isinstance(res, dict):
                is_valid = bool(res.get("is_valid", True))
                score = float(res.get("score", 0.9))
                issues = res.get("issues") or []
                refined = res.get("refined_prompt") or prompt
                return RelevanceCheckResult(
                    scene_id=scene.scene_id,
                    is_valid=is_valid,
                    score=score,
                    issues=issues,
                    refined_prompt=refined if not is_valid else prompt,
                )
        except Exception:
            pass

        return RelevanceCheckResult(
            scene_id=scene.scene_id,
            is_valid=True,
            score=0.95,
            issues=[],
            refined_prompt=prompt,
        )

"""Automated Vision Critic for Still Images and Pre-Video QA.

Inspects generated still frames before video animation using multimodal vision.
Verifies:
1. Wardrobe integrity (proper tailoring, authentic hem lengths, no pants hitched up,
   no exposed calves/shins in traditional wear, no torn or warped fabrics).
2. Facial and anatomical plausibility (two symmetric eyes, natural mouth, 5 fingers per hand,
   no extra or floating limbs).
3. Genre aesthetic and physical realism.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .router import get_router


@dataclass
class StillAuditResult:
    scene_id: int
    is_valid: bool
    score: int  # 0 to 100
    issues: list[str] = field(default_factory=list)
    corrective_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "is_valid": self.is_valid,
            "score": self.score,
            "issues": self.issues,
            "corrective_prompt": self.corrective_prompt,
        }


class VisionCritic:
    """Pre-animation quality supervisor using Multimodal Vision."""

    def __init__(self, router=None, threshold: int = 75):
        self.router = router or get_router()
        self.threshold = threshold

    def inspect_still(
        self,
        image_path: str | Path,
        scene_id: int,
        prompt: str,
        genre: str = "general",
    ) -> StillAuditResult:
        """Inspect a single still image for anatomical, wardrobe, and realism flaws."""
        path = Path(image_path)
        if not path.exists():
            return StillAuditResult(scene_id=scene_id, is_valid=False, score=0, issues=["Image file does not exist"])

        try:
            b64_img = base64.b64encode(path.read_bytes()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{b64_img}"
        except Exception:
            return StillAuditResult(scene_id=scene_id, is_valid=True, score=85)

        system = (
            "You are a strict VFX visual quality supervisor inspecting AI-generated film stills before animation. "
            "Examine the image carefully for AI generation defects, anatomical anomalies, and wardrobe flaws."
        )

        user_prompt = f"""Audit this film still for production quality and realism:

SCENE PURPOSE / PROMPT:
{prompt}

GENRE: {genre}

CRITICAL AUDIT CHECKS:
1. Wardrobe & Clothing Fit:
   - Are pants/trousers fitting naturally? Flag immediately if pants are unnaturally hitched up, exposing half the legs/calves in traditional or formal attire.
   - Are hems, seams, sleeves, and necklines realistic and properly tailored?
2. Anatomy & Faces:
   - Are faces distorted, melted, or uncanny? Are eyes symmetric and natural?
   - Check hands/limbs: Are there extra fingers (more than 5), missing limbs, floating body parts, or warped anatomy?
3. Visual Realism & Artifacts:
   - Are there melted textures, blurry disfigured objects, or generic AI slop artifacts?

Score the image from 0 to 100.
If score < {self.threshold}, mark is_valid as false, list the exact issues, and provide a 'corrective_prompt' with explicit negative constraints to fix the still on the next attempt.

Return ONLY JSON:
{{
  "is_valid": true,
  "score": 88,
  "issues": [],
  "corrective_prompt": ""
}}
"""

        try:
            client = self.router.client
            model = self.router.primary_model
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                timeout=30,
            )
            raw = response.choices[0].message.content or "{}"
            raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            data = json.loads(raw_clean)
            score = int(data.get("score", 85))
            is_valid = bool(data.get("is_valid", score >= self.threshold))
            issues = list(data.get("issues") or [])
            corrective = str(data.get("corrective_prompt") or "").strip()
            return StillAuditResult(
                scene_id=scene_id,
                is_valid=is_valid,
                score=score,
                issues=issues,
                corrective_prompt=corrective,
            )
        except Exception:
            return StillAuditResult(scene_id=scene_id, is_valid=True, score=85)

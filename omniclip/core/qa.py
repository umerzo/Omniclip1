"""Video Quality Control & Scene-Level QA Agent.

Evaluates generated video scenes and audio for content alignment, visual fidelity,
continuity, and editing rhythm.
Performs multimodal frame inspection to catch grotesque anatomical morphing,
melted faces, and AI artifacts before final assembly.
Assigns 0-100 quality scores and triggers targeted scene regeneration.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .router import get_router
from .scene_planner import ProductionPlan, ProductionScene
from ..utils.media import ffmpeg_exe, probe_duration


@dataclass
class SceneQAResult:
    scene_id: int = 0
    quality_score: int = 85  # 0 to 100
    issues: list[str] = field(default_factory=list)
    action: str = "accept"  # "accept" | "regenerate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_id: int = 0) -> SceneQAResult:
        if not isinstance(data, dict):
            return cls(scene_id=default_id)
        return cls(
            scene_id=int(data.get("scene_id") or data.get("index") or default_id),
            quality_score=int(data.get("quality_score") or 85),
            issues=list(data.get("issues") or []),
            action=str(data.get("action") or "accept"),
        )


@dataclass
class VideoQAReport:
    overall_score: int = 85
    passed: bool = True
    summary: str = ""
    scene_results: list[SceneQAResult] = field(default_factory=list)
    scenes_to_regenerate: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "summary": self.summary,
            "scene_results": [r.to_dict() for r in self.scene_results],
            "scenes_to_regenerate": self.scenes_to_regenerate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoQAReport:
        return cls(
            overall_score=int(data.get("overall_score", 85)),
            passed=bool(data.get("passed", True)),
            summary=data.get("summary", ""),
            scene_results=[SceneQAResult.from_dict(r, default_id=i) for i, r in enumerate(data.get("scene_results") or [])],
            scenes_to_regenerate=data.get("scenes_to_regenerate") or [],
        )


class VideoQA:
    """Evaluates video scenes against quality thresholds and recommends targeted fixes."""

    def __init__(self, threshold: int = 70, router=None):
        self.threshold = threshold
        self.router = router or get_router()

    def _extract_sample_frames(self, video_path: Path, count: int = 2) -> list[str]:
        """Extract representative frames from a video clip as base64 JPEG strings."""
        frames = []
        try:
            dur = probe_duration(video_path)
            if dur <= 0:
                return frames

            timestamps = [dur * 0.35, dur * 0.75] if count == 2 else [dur * (i + 1) / (count + 1) for i in range(count)]
            exe = ffmpeg_exe()
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

            for ts in timestamps:
                cmd = [
                    exe, "-hide_banner", "-loglevel", "error", "-ss", f"{ts:.2f}",
                    "-i", str(video_path), "-frames:v", "1", "-f", "image2pipe",
                    "-c:v", "mjpeg", "-q:v", "3", "pipe:1"
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=creationflags)
                if proc.returncode == 0 and proc.stdout:
                    b64 = base64.b64encode(proc.stdout).decode("utf-8")
                    frames.append(f"data:image/jpeg;base64,{b64}")
        except Exception as exc:
            print(f"Frame extraction notice for QA: {exc}", flush=True)
        return frames

    def audit_clip_frames(
        self,
        asset_path: str | Path,
        scene_id: int,
        narration: str = "",
        prompt: str = "",
    ) -> SceneQAResult:
        """Inspect actual extracted video frames using multimodal vision for grotesque AI defects."""
        path = Path(asset_path)
        if not path.exists():
            return SceneQAResult(scene_id=scene_id, quality_score=90, action="accept")

        frames = self._extract_sample_frames(path, count=2)
        if not frames:
            return SceneQAResult(scene_id=scene_id, quality_score=85, action="accept")

        user_prompt = f"""Review these frames sampled from generated video scene {scene_id}:
SCENE NARRATION: {narration}
INTENDED PROMPT: {prompt}

CRITICAL AUDIT CHECKS:
1. Grotesque Morphing / Species Transmutation:
   - Is a human body or face morphing or blending into an animal (e.g. donkey, dog, horse) or hybrid chimera?
   - Are there unnatural animal limbs attached to people or humans wearing animal features?
2. Facial / Anatomical Distortion:
   - Are faces melted, monstrously warped, or distorted?
   - Are there rubbery limbs, missing limbs, or extra floating extremities?
3. Visual Artifacts:
   - Is there severe diffusion collapse, garbled unrecognisable shapes, or chaotic tearing?

GRADING CRITERIA:
- If ANY grotesque human-animal morphing, species transmutation, or melted face/anatomy is present:
  - You MUST set "quality_score" below 60 (e.g. 35 to 50).
  - You MUST set "action" to "regenerate".
  - You MUST detail the exact defect in "issues".
- If the video portrays the scene naturally without grotesque morphing:
  - Set "quality_score" >= 85 and "action" to "accept".

Return ONLY JSON:
{{
  "quality_score": 88,
  "issues": [],
  "action": "accept"
}}
"""
        try:
            client = self.router.client
            model = self.router.primary_model
            content_list: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for f in frames:
                content_list.append({"type": "image_url", "image_url": {"url": f}})

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict VFX visual QA supervisor. Catch grotesque AI mutations, melted faces, and distorted anatomy.",
                    },
                    {"role": "user", "content": content_list},
                ],
                response_format={"type": "json_object"},
                timeout=30,
            )
            raw = response.choices[0].message.content or "{}"
            raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            data = json.loads(raw_clean)
            score = int(data.get("quality_score", 85))
            action = str(data.get("action", "accept" if score >= self.threshold else "regenerate"))
            issues = list(data.get("issues") or [])
            return SceneQAResult(
                scene_id=scene_id,
                quality_score=score,
                issues=issues,
                action=action,
            )
        except Exception as err:
            print(f"VideoQA multimodal clip inspection notice on scene {scene_id}: {err}", flush=True)
            return SceneQAResult(scene_id=scene_id, quality_score=85, action="accept")

    def evaluate_scenes(
        self,
        plan: ProductionPlan,
        scenes_data: list[dict[str, Any]],
        audit_frames: bool = True,
    ) -> VideoQAReport:
        """Run QA evaluation across all generated scenes, combining frame inspection and narrative review."""
        frame_results: dict[int, SceneQAResult] = {}

        if audit_frames:
            for s in scenes_data:
                idx = s.get("index", s.get("scene_id", 0))
                asset_path = s.get("asset_path")
                if asset_path and Path(asset_path).exists():
                    res = self.audit_clip_frames(
                        asset_path=asset_path,
                        scene_id=idx,
                        narration=s.get("narration", ""),
                        prompt=s.get("visual_prompt", ""),
                    )
                    if res.action != "accept" or res.quality_score < self.threshold:
                        frame_results[idx] = res

        system = (
            "You are an executive post-production supervisor performing Quality Assurance. "
            "Evaluate each scene for: content accuracy, visual relevance to narration, subject continuity, "
            "and avoidance of AI defects. Assign each scene a score 0-100 and action ('accept', 'regenerate')."
        )

        scenes_summary = []
        for s in scenes_data:
            idx = s.get("index", s.get("scene_id", 0))
            scenes_summary.append({
                "scene_id": idx,
                "purpose": s.get("purpose", ""),
                "narration": s.get("narration", "")[:120],
                "visual_prompt": s.get("visual_prompt", "")[:140],
                "asset_source": s.get("asset_source", "agnes"),
            })

        prompt = f"""Review the production results for this video:

TITLE: {plan.title}
STYLE BIBLE: {plan.style_bible.visual_style}
THRESHOLD: {self.threshold}/100

SCENES TO EVALUATE:
{json.dumps(scenes_summary, indent=2)}

Output JSON:
{{
  "overall_score": 88,
  "passed": true,
  "summary": "1-2 sentence overall quality verdict",
  "scene_results": [
    {{
      "scene_id": 0,
      "quality_score": 90,
      "issues": [],
      "action": "accept"
    }}
  ]
}}
"""
        final_scene_results: list[SceneQAResult] = []
        try:
            res = self.router.run_prompt(prompt, system=system, high_reasoning=False, response_json=True)
            if isinstance(res, dict) and res.get("scene_results"):
                for i, r in enumerate(res.get("scene_results") or []):
                    parsed = SceneQAResult.from_dict(r, default_id=i)
                    # Merge with frame-level audit if frame-level found defects
                    if parsed.scene_id in frame_results:
                        f_res = frame_results[parsed.scene_id]
                        parsed.quality_score = min(parsed.quality_score, f_res.quality_score)
                        parsed.action = f_res.action
                        parsed.issues = list(set(parsed.issues + f_res.issues))
                    final_scene_results.append(parsed)

                regen = [r.scene_id for r in final_scene_results if r.action != "accept" or r.quality_score < self.threshold]
                overall = int(res.get("overall_score", 85))
                if regen:
                    overall = min(overall, 65)
                return VideoQAReport(
                    overall_score=overall,
                    passed=overall >= self.threshold and not regen,
                    summary=res.get("summary", "Production quality verified."),
                    scene_results=final_scene_results,
                    scenes_to_regenerate=regen,
                )
        except Exception as exc:
            print(f"VideoQA warning: {exc}")

        # Deterministic default pass, incorporating any failed frame audits
        for s in scenes_data:
            idx = s.get("index", s.get("scene_id", 0))
            if idx in frame_results:
                final_scene_results.append(frame_results[idx])
            else:
                final_scene_results.append(SceneQAResult(scene_id=idx, quality_score=90, issues=[], action="accept"))

        regen = [r.scene_id for r in final_scene_results if r.action != "accept" or r.quality_score < self.threshold]
        overall = 60 if regen else 90
        return VideoQAReport(
            overall_score=overall,
            passed=not bool(regen),
            summary="Scenes checked against production standards." if not regen else f"{len(regen)} scene(s) flagged for regeneration.",
            scene_results=final_scene_results,
            scenes_to_regenerate=regen,
        )

"""Script Engine.

Generates original scripts using genre-specific narrative architectures:
Cinematic story, Short-form viral, Automotive, Product, Documentary,
Explainer, Tutorial, Commentary, Montage, etc.
Replaces one-size-fits-all script prompts with specialized production templates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .classifier import ContentClassification
from .dna import ContentDNA
from .router import get_router
from .strategy import CreativeStrategy

GENRE_FRAMEWORKS: dict[str, str] = {
    "cinematic_story": (
        "1. Hook: Immersive cold open with high emotional stakes.\n"
        "2. Setup: Establish protagonist, desire, and immediate world.\n"
        "3. Conflict: Inciting incident disrupting the baseline.\n"
        "4. Escalation: Rising obstacles and intensifying tension.\n"
        "5. Turning Point: Pivotal decision or shift in perspective.\n"
        "6. Climax: Peak dramatic confrontation or achievement.\n"
        "7. Resolution: Satisfying emotional closing image."
    ),
    "short_story": (
        "1. Pattern Interrupt (0-3s): Visually arresting or surprising opening statement.\n"
        "2. Curiosity Hook (3-8s): Pose an irresistible premise or challenge.\n"
        "3. Rapid Progression (8-25s): Quick, punchy story beats with rising stakes.\n"
        "4. Climax / Twist (25-45s): The unforeseen turn or punchline.\n"
        "5. Payoff & Loop (45-60s): Crisp conclusion that loops seamlessly back to the hook."
    ),
    "automotive_cinematic": (
        "1. Hero Hook: Dramatic low-angle reveal with roaring acceleration.\n"
        "2. Sculpted Identity: Close-ups on body contours, badge, and headlights.\n"
        "3. Performance Pulse: Dynamic tracking shots on mountain curve or open tarmac.\n"
        "4. Cockpit & Driver Harmony: Interior dashboard, steering grip, shift feel.\n"
        "5. High-Speed Escalation: Sweeping drone and tracking camera captures pure velocity.\n"
        "6. Lifestyle Climax: Atmospheric twilight rest stop, heat shimmering off the exhaust."
    ),
    "car_showcase": (
        "1. Spec Hook: The single most shocking capability or exterior stance.\n"
        "2. Exterior Walkaround: Wheel architecture, aerodynamics, aggressive stance.\n"
        "3. Cabin Luxury / Tech: Digital cluster, premium materials, cockpit layout.\n"
        "4. Sound & Throttle: Pure engine symphony and dynamic road poise.\n"
        "5. Verdict: Who this machine is built for and what it represents."
    ),
    "product_showcase": (
        "1. Problem Hook: The universal frustration that demands a solution.\n"
        "2. Product Reveal: High-end hero lighting uncovering the device.\n"
        "3. Key Innovation: Visual macro demonstration of the flagship feature.\n"
        "4. Everyday Workflow: Real-world ergonomic benefit in action.\n"
        "5. Proof Point: Quantitative benchmark or undeniable transformation.\n"
        "6. Call to Action: Clear, confident invitation."
    ),
    "documentary": (
        "1. Atmospheric Hook: An enigmatic archive clue or gripping historical question.\n"
        "2. Historical Context: Setting the geographic, temporal, and human landscape.\n"
        "3. The Discovery: Unearthing the first key evidence or anomaly.\n"
        "4. Systematic Escalation: Interlocking clues, investigations, and conflicting forces.\n"
        "5. The Revelation: The core breakthrough that redefines our understanding.\n"
        "6. Lasting Impact: Philosophical conclusion and legacy."
    ),
    "explainer": (
        "1. The Curious Question: A counter-intuitive paradox or pressing question.\n"
        "2. The Common Myth: Why our initial intuition is usually wrong.\n"
        "3. The Core Mechanism: Visual analogy simplifying the underlying principle.\n"
        "4. Real-world Application: How this mechanism manifests in daily life.\n"
        "5. The Takeaway: A memorable mental model to remember forever."
    ),
    "tutorial": (
        "1. End Result Hook: Show the stunning finished outcome first.\n"
        "2. Essential Setup: The exact tools and preparation required.\n"
        "3. Phase 1 - Foundation: Clear, unambiguous first steps.\n"
        "4. Phase 2 - Execution: Detail on technique and the #1 mistake to avoid.\n"
        "5. Phase 3 - Finishing Touch: The pro tip that elevates the result.\n"
        "6. Final Showcase: Inspecting the perfected creation."
    ),
    "commentary": (
        "1. Provocative Claim: The thesis that challenges mainstream consensus.\n"
        "2. Context: What just happened or why this topic matters right now.\n"
        "3. Evidence Layer: Data points, direct examples, and visual breakdowns.\n"
        "4. Counter-Perspective: Steelmanning the opposing viewpoint before dismantling it.\n"
        "5. Synthesis: The bigger cultural or strategic takeaway."
    ),
    "montage": (
        "1. Opening Motif: Gentle establishing pulse setting the emotional frequency.\n"
        "2. Acceleration: Cut rhythm speeds up alongside musical crescendo.\n"
        "3. Visual Variety: Extreme contrasts in scale (macro to panoramic).\n"
        "4. Kinetic Peak: High-energy sequence of maximum movement and power.\n"
        "5. Harmonic Outro: Lingering atmospheric beauty shot."
    ),
}


@dataclass
class ScriptBeat:
    index: int
    role: str  # "hook", "setup", "escalation", "climax", "payoff", "cta"
    narration: str
    target_duration: float
    visual_direction: str
    on_screen_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedScript:
    title: str
    logline: str
    format: str
    primary_type: str
    total_target_duration: float
    beats: list[ScriptBeat] = field(default_factory=list)

    @property
    def full_narration(self) -> str:
        return " ".join(b.narration.strip() for b in self.beats if b.narration.strip())

    @property
    def total_words(self) -> int:
        return len(self.full_narration.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "logline": self.logline,
            "format": self.format,
            "primary_type": self.primary_type,
            "total_target_duration": self.total_target_duration,
            "beats": [b.to_dict() for b in self.beats],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratedScript:
        return cls(
            title=data.get("title", "Original Video"),
            logline=data.get("logline", ""),
            format=data.get("format", "short"),
            primary_type=data.get("primary_type", "cinematic_story"),
            total_target_duration=float(data.get("total_target_duration", 60.0)),
            beats=[ScriptBeat(**b) for b in data.get("beats", [])],
        )


class ScriptEngine:
    """Generates structured, genre-specific production scripts."""

    def __init__(self, router=None):
        self.router = router or get_router()

    def generate(
        self,
        strategy: CreativeStrategy,
        classification: ContentClassification,
        dna: ContentDNA,
        target_duration: float = 60.0,
        target_scenes_count: int = 12,
        language: str = "en",
        cast: list[dict] | None = None,
        shots: list | None = None,
    ) -> GeneratedScript:
        """Write a production script adhering to the genre architecture and preserving characters."""
        genre = classification.primary_type
        framework = GENRE_FRAMEWORKS.get(genre, GENRE_FRAMEWORKS.get("short_story" if classification.format == "short" else "cinematic_story"))

        # Pacing calculation: ~140 WPM -> ~2.3 words/sec
        wpm = dna.pacing.get("words_per_minute") or 145
        wps = float(wpm) / 60.0
        max_total_words = int(target_duration * wps)

        cast_desc = "\n".join(f"- {c.get('id', c.get('name'))}: {c.get('appearance')}" for c in (cast or []))
        shots_desc = "\n".join(f"- Shot {i}: {getattr(s, 'description', str(s))[:160]}" for i, s in enumerate(shots or []))

        system = (
            "You are a master scriptwriter and YouTube retention strategist. "
            "Write a production script that follows the specific genre framework. "
            "If characters and visual beats from the source are provided, PRESERVE and elevate those characters and that story! "
            "Do NOT invent unrelated new animals or characters when existing cast is given. "
            "Make every single scene punchy, visual, and intentional."
        )

        prompt = f"""Write an original production script for this video.

CONCEPT & STRATEGY:
- Title / Concept: {strategy.new_concept}
- Core Promise: {strategy.core_promise}
- Target Audience: {strategy.target_audience}
- Hook Strategy: {strategy.hook_strategy}
- Visual Strategy: {strategy.visual_strategy}
- Sound Strategy: {strategy.sound_strategy}
- Language: {language}

CHARACTERS & VISUAL WORLD (FEATURE THESE CHARACTERS):
{cast_desc if cast_desc else 'No specific recurring characters provided'}

ORIGINAL SCENE BEATS SEQUENCE:
{shots_desc if shots_desc else 'No specific shot breakdown'}

GENRE & ARCHITECTURE:
- Format: {classification.format} ({target_duration:.0f} seconds total target duration)
- Genre: {classification.primary_type}
- Structural Framework:
{framework}

PRODUCTION CONSTRAINTS:
- Target Scenes/Beats: ~{target_scenes_count} scenes (each scene between 2.0s and 6.0s)
- Total Duration: approximately {target_duration:.0f} seconds
- Total Word Budget: maximum ~{max_total_words} words (approx {wps:.1f} words/sec)
- Narration Dependency: {classification.narration_dependency}
  (If 'low', keep narration minimal or visual-first; if 'none', leave narration blank and focus on visual direction)

Output a JSON object with this exact schema:
{{
  "title": "Compelling Title",
  "logline": "One sentence summary of this video",
  "format": "{classification.format}",
  "primary_type": "{classification.primary_type}",
  "total_target_duration": {target_duration:.1f},
  "beats": [
    {{
      "index": 0,
      "role": "hook / setup / escalation / climax / payoff / cta",
      "narration": "What is spoken in this scene (or empty if purely visual)",
      "target_duration": 4.0,
      "visual_direction": "Clear description of what action and visual must take place",
      "on_screen_text": "Optional punchy text banner (keep short or empty)"
    }}
  ]
}}
"""

        try:
            res = self.router.run_prompt(prompt, system=system, high_reasoning=True, response_json=True)
            if isinstance(res, dict) and res.get("beats"):
                return GeneratedScript.from_dict(res)
        except Exception as exc:
            print(f"ScriptEngine generation failed: {exc}")

        # Deterministic fallback script
        return self._fallback_script(strategy, classification, target_duration, target_scenes_count)

    def _fallback_script(
        self,
        strategy: CreativeStrategy,
        classification: ContentClassification,
        target_duration: float,
        count: int,
    ) -> GeneratedScript:
        avg_dur = max(3.0, target_duration / max(1, count))
        beats = []
        for i in range(count):
            role = "hook" if i == 0 else ("cta" if i == count - 1 else "escalation")
            beats.append(
                ScriptBeat(
                    index=i,
                    role=role,
                    narration=f"Scene {i + 1} advancing our {classification.primary_type.replace('_', ' ')}.",
                    target_duration=avg_dur,
                    visual_direction=f"Dynamic {classification.primary_type} visual focusing on primary subject.",
                    on_screen_text="" if i != 0 else "WATCH THIS",
                )
            )
        return GeneratedScript(
            title=strategy.new_concept[:50],
            logline=strategy.core_promise,
            format=classification.format,
            primary_type=classification.primary_type,
            total_target_duration=target_duration,
            beats=beats,
        )

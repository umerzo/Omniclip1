"""Script Engine.

Generates original, broadcast-quality scripts using genre-specific narrative architectures:
Cinematic story, Short-form viral, Automotive, Product, Documentary,
Explainer, Science & Tech, Tutorial, Commentary, Montage, etc.

Features:
1. Adaptive Master Personas (Feynman for Science, Insider for Explainers, Storyteller for Drama).
2. Narrative Grounding: Ingests full donor transcripts to start from line one, never from halfway.
3. Natural Momentum: Writes continuous, flowing human speech with banned AI clichés.
4. Long-form Chapter Sequencing: Hierarchical chapter breakdown for 3–15+ minute productions.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .classifier import ContentClassification
from .dna import ContentDNA
from .router import get_router
from .strategy import CreativeStrategy

MASTER_PERSONAS: dict[str, dict[str, str]] = {
    "science_tech": {
        "title": "The Feynman Communicator",
        "voice": (
            "Infectiously curious, engaging, conversational, and plain-spoken. "
            "Explains deep physical, technological, or mathematical mechanisms using everyday physical "
            "analogies (cups of water, coins, magnets). Absolutely zero dry academic jargon or stiff prose."
        ),
        "hook_archetype": "The Counter-Intuitive Paradox: A question or fact that challenges standard intuition.",
        "momentum": "Builds from simple tangible intuition to the profound mechanism, with natural spoken connectors ('Here is the crazy part...', 'Think of it like this...').",
    },
    "explainer": {
        "title": "The Investigative Insider",
        "voice": (
            "Sharp, articulate, and compelling (Vox / Johnny Harris style). "
            "Connects hidden dots, reveals geopolitical or economic realities, and explains cause-and-effect with brisk momentum."
        ),
        "hook_archetype": "The Hidden Connection: A small concrete object or event that quietly controls a massive global reality.",
        "momentum": "Fast-paced logical escalation with clear milestones ('First, the setup...', 'That is when everything shifted...', 'And the real reason is...').",
    },
    "cinematic_story": {
        "title": "The Cinematic Storyteller",
        "voice": (
            "Warm, human, empathetic, and grounded. "
            "Establishes the character, their immediate world, and their true desire or dilemma before plunging into conflict. "
            "Natural spoken dialogue with relatable human cadence."
        ),
        "hook_archetype": "The Grounded Dilemma / Atmospheric Setup: Establish the world and the stakes clearly from line one. Never start mid-action from halfway without context.",
        "momentum": "Emotional cause-and-effect where each action raises personal stakes toward a satisfying payoff.",
    },
    "short_story": {
        "title": "The High-Engagement Storyteller",
        "voice": (
            "Punchy, relatable, and authentic. "
            "Fast setup that clearly introduces who, where, and what is at stake within the first sentence, "
            "then moves with escalating momentum to a clever twist or punchline."
        ),
        "hook_archetype": "The Irresistible Premise: Clear, immediate situation that sets up an urgent desire or humorous challenge.",
        "momentum": "Rapid story beats where every sentence propels the character forward.",
    },
    "documentary": {
        "title": "The Atmospheric Observer",
        "voice": (
            "Vivid, measured, and evocative (David Attenborough / Ken Burns standard). "
            "Gives weight and sensory texture to the environment and historical forces. Uses rhythm and pause for dramatic impact."
        ),
        "hook_archetype": "The Enigmatic Scale: A monumental question or enduring physical mystery.",
        "momentum": "Deep observational progression that lets the imagery and human drama breathe with gravitas.",
    },
    "product_showcase": {
        "title": "The Sensory Purist",
        "voice": (
            "Confident, tactile, and sensory-driven. "
            "Focuses on physical materials, mechanical tolerances, sound, and immediate ergonomic transformation. Minimalist and punchy."
        ),
        "hook_archetype": "The Core Transformation: The ultimate friction solved in one undeniable visual demonstration.",
        "momentum": "Pivots from problem to mechanical beauty to real-world capability.",
    },
    "automotive_cinematic": {
        "title": "The Kinetic Enthusiast",
        "voice": (
            "Passionate, visceral, and pulse-driven. "
            "Centers on the roar of the engine, chassis balance, steering feedback, and the emotion of speed and freedom."
        ),
        "hook_archetype": "The Acoustic / Velocity Hook: Pure acceleration and aggressive stance.",
        "momentum": "Builds alongside throttle input from sculpted design to high-speed road harmony.",
    },
    "tutorial": {
        "title": "The Master Mentor",
        "voice": (
            "Encouraging, clear, and reassuring. "
            "Focuses on sensory cues ('listen for that crackle', 'until golden-brown') and common pitfalls to avoid. Zero fluff."
        ),
        "hook_archetype": "The Flawless Outcome First: Showcase the perfected end result, followed by the simple secret to achieving it.",
        "momentum": "Step-by-step foundation to pro execution.",
    },
    "commentary": {
        "title": "The Thought Leader",
        "voice": (
            "Bold, insightful, and provocative. "
            "Challenges conventional consensus with sharp logic, direct examples, and intellectual honesty."
        ),
        "hook_archetype": "The Bold Thesis: The provocative claim that mainstream thinking gets wrong.",
        "momentum": "Evidence-backed dismantle of myths leading to a synthesis.",
    },
}

BANNED_AI_WORDS = (
    "delve", "tapestry", "in a world where", "testament to", "crucial role",
    "let's explore", "think you know", "think again", "it's not just a",
    "nestled in", "beacon of", "symphony of", "top-secret mission", "kids are hyped",
    "game-changer", "dive in", "unravel the", "fast-forward to",
)


@dataclass
class ScriptBeat:
    index: int = 0
    role: str = "escalation"  # "hook", "setup", "escalation", "climax", "payoff", "cta"
    narration: str = ""
    target_duration: float = 3.0
    visual_direction: str = ""
    on_screen_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_index: int = 0) -> ScriptBeat:
        if not isinstance(data, dict):
            return cls(index=default_index)

        idx = data.get("index", default_index)
        try:
            idx = int(idx)
        except Exception:
            idx = default_index

        role = str(data.get("role") or "escalation")

        narration = ""
        for key in ["narration", "speech", "dialogue", "voiceover", "script", "text", "lines", "dialog"]:
            if data.get(key) is not None:
                narration = str(data[key]).strip()
                break

        visual_direction = ""
        for key in [
            "visual_direction", "visual_prompt", "visual", "action",
            "scene_description", "description", "scene", "shot"
        ]:
            if data.get(key) is not None:
                visual_direction = str(data[key]).strip()
                break

        dur = data.get("target_duration") or data.get("duration") or 3.0
        try:
            dur = float(dur)
        except Exception:
            dur = 3.0

        on_screen_text = str(data.get("on_screen_text") or data.get("overlay_text") or "").strip()

        return cls(
            index=idx,
            role=role,
            narration=narration,
            target_duration=dur,
            visual_direction=visual_direction,
            on_screen_text=on_screen_text,
        )


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
        raw_beats = data.get("beats") or data.get("scenes") or data.get("shots") or []
        beats = [ScriptBeat.from_dict(b, default_index=i) for i, b in enumerate(raw_beats)]
        return cls(
            title=str(data.get("title") or "Original Video"),
            logline=str(data.get("logline") or ""),
            format=str(data.get("format") or "short"),
            primary_type=str(data.get("primary_type") or "cinematic_story"),
            total_target_duration=float(data.get("total_target_duration") or 60.0),
            beats=beats,
        )


class ScriptEngine:
    """Intelligent, genre-adaptive production scriptwriter."""

    def __init__(self, router=None):
        self.router = router or get_router()

    def _pick_persona(self, genre: str) -> dict[str, str]:
        g = (genre or "cinematic_story").lower()
        if g in ("science", "technology", "educational"):
            return MASTER_PERSONAS["science_tech"]
        if g in ("explainer", "finance", "business", "news"):
            return MASTER_PERSONAS["explainer"]
        if g in ("short_story", "comedy"):
            return MASTER_PERSONAS["short_story"]
        if g in ("cinematic_story", "drama"):
            return MASTER_PERSONAS["cinematic_story"]
        if g in ("documentary", "history", "travel"):
            return MASTER_PERSONAS["documentary"]
        if g in ("product_showcase", "advertisement", "satisfying"):
            return MASTER_PERSONAS["product_showcase"]
        if g in ("automotive_cinematic", "car_showcase"):
            return MASTER_PERSONAS["automotive_cinematic"]
        if g in ("tutorial", "food", "fitness", "lifestyle", "before_after", "transformation"):
            return MASTER_PERSONAS["tutorial"]
        if g in ("commentary", "podcast_clip", "talking_head", "motivational"):
            return MASTER_PERSONAS["commentary"]
        return MASTER_PERSONAS["cinematic_story"]

    def _clean_text(self, text: str) -> str:
        """Strip banned AI cliches from dialogue and narration."""
        res = text
        for banned in BANNED_AI_WORDS:
            pattern = re.compile(rf"\b{re.escape(banned)}\b", re.IGNORECASE)
            res = pattern.sub("", res)
        return " ".join(res.split()).strip()

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
        source_transcript: str = "",
    ) -> GeneratedScript:
        """Write a complete production script with authentic persona and grounded momentum."""
        target_duration = max(15.0, float(target_duration))
        persona = self._pick_persona(classification.primary_type)

        # For long videos (> 180s / 3 min), use hierarchical multi-act chapter sequencing
        if target_duration > 180.0:
            return self._generate_longform(
                strategy=strategy,
                classification=classification,
                dna=dna,
                target_duration=target_duration,
                target_scenes_count=target_scenes_count,
                language=language,
                cast=cast,
                shots=shots,
                source_transcript=source_transcript,
                persona=persona,
            )

        return self._generate_standard(
            strategy=strategy,
            classification=classification,
            dna=dna,
            target_duration=target_duration,
            target_scenes_count=target_scenes_count,
            language=language,
            cast=cast,
            shots=shots,
            source_transcript=source_transcript,
            persona=persona,
        )

    def _generate_standard(
        self,
        strategy: CreativeStrategy,
        classification: ContentClassification,
        dna: ContentDNA,
        target_duration: float,
        target_scenes_count: int,
        language: str,
        cast: list[dict] | None,
        shots: list | None,
        source_transcript: str,
        persona: dict[str, str],
    ) -> GeneratedScript:
        genre = classification.primary_type
        wpm = dna.pacing.get("words_per_minute") or 145
        wps = float(wpm) / 60.0
        max_total_words = int(target_duration * wps)

        cast_desc = "\n".join(f"- {c.get('id', c.get('name'))}: {c.get('appearance')}" for c in (cast or []))
        shots_desc = "\n".join(f"- Shot {i} ({getattr(s, 'duration', 3.0):.1f}s): {getattr(s, 'description', str(s))[:140]}" for i, s in enumerate(shots or []))

        system = (
            f"You are {persona['title']}. {persona['voice']}\n"
            "MANDATORY WRITING PRINCIPLES:\n"
            "1. GROUNDED OPENING (NEVER START FROM HALFWAY): Introduce the core character, setting, curious question, or premise in sentence one. "
            "Do NOT plunge blindly into the middle of action without context.\n"
            "2. MOMENTUM & EASY WORDS: Use everyday, vivid spoken words with short, punchy active phrasing. Ban all corporate AI buzzwords (delve, tapestry, crucial, think you know X).\n"
            "3. CONTINUOUS FLOW: Write a flowing, broadcast-quality spoken narrative across all visual beats.\n"
            "4. SOURCE FIDELITY: If characters and story context from the source are provided, PRESERVE and ELEVATE them faithfully."
        )

        prompt = f"""Write an original, broadcast-quality script for this {genre} production.

CONCEPT & STRATEGY:
- Title / Concept: {strategy.new_concept}
- Core Promise: {strategy.core_promise}
- Target Audience: {strategy.target_audience}
- Hook Strategy: {strategy.hook_strategy} ({persona['hook_archetype']})
- Language: {language}

ORIGINAL SOURCE TRANSCRIPT & DIALOGUE (THE TRUE PREMISE):
{source_transcript[:2500] if source_transcript else 'No donor transcript available (derive fresh premise from visual shots below)'}

CHARACTERS & VISUAL WORLD (PRESERVE EXACT CHARACTERS):
{cast_desc if cast_desc else 'No specific recurring characters detected'}

REFERENCE CAMERA SHOT SEQUENCE (TOTAL DURATION: {target_duration:.1f}s):
{shots_desc if shots_desc else f'Plan ~{target_scenes_count} natural visual cuts totaling {target_duration:.0f} seconds'}

PRODUCTION CONSTRAINTS:
- Format: {classification.format}
- Total Video Runtime: EXACTLY ~{target_duration:.0f} seconds total
- Target Beats/Cuts: ~{target_scenes_count} visual beats
- Word Budget: ~{max_total_words} words total across the whole video
- Narration Dependency: {classification.narration_dependency}

Output a JSON object with this exact schema:
{{
  "title": "Engaging Title",
  "logline": "Compelling one-sentence summary",
  "format": "{classification.format}",
  "primary_type": "{classification.primary_type}",
  "total_target_duration": {target_duration:.1f},
  "beats": [
    {{
      "index": 0,
      "role": "hook / setup / escalation / climax / payoff / cta",
      "narration": "Natural spoken dialogue or narration in {language} for this visual beat (or \"\" if purely visual/cooking)",
      "target_duration": 3.5,
      "visual_direction": "Cinematic visual description and camera movement matching what is being spoken",
      "on_screen_text": ""
    }}
  ]
}}

CRITICAL REQUIREMENTS:
1. Write the narration in fluent, natural {language} (e.g. Urdu script for Urdu, Hindi script for Hindi, plain spoken English for English).
2. The total sum of 'target_duration' across all beats MUST match approximately {target_duration:.1f} seconds!
3. NO robotic placeholders like 'Scene 1 advancing our story'.
"""

        try:
            res = self.router.run_prompt(prompt, system=system, high_reasoning=True, response_json=True)
            if isinstance(res, dict):
                script = GeneratedScript.from_dict(res)
                if script.beats:
                    for b in script.beats:
                        b.narration = self._clean_text(b.narration)
                    return script
        except Exception as exc:
            print(f"ScriptEngine standard generation notice: {exc}")

        return self._fallback_script(
            strategy, classification, target_duration, target_scenes_count, shots=shots
        )

    def _generate_longform(
        self,
        strategy: CreativeStrategy,
        classification: ContentClassification,
        dna: ContentDNA,
        target_duration: float,
        target_scenes_count: int,
        language: str,
        cast: list[dict] | None,
        shots: list | None,
        source_transcript: str,
        persona: dict[str, str],
    ) -> GeneratedScript:
        """Hierarchical multi-act sequencing for long productions (3–15+ minutes)."""
        target_chapter_len = 120.0
        num_chapters = max(2, int(round(target_duration / target_chapter_len)))
        chapter_duration = target_duration / num_chapters

        print(f"   long-form script generation: {num_chapters} narrative chapters ({chapter_duration:.1f}s each, {target_duration:.0f}s total)")

        all_beats: list[ScriptBeat] = []
        prev_context = "Beginning of production."
        beat_cursor = 0

        total_shots = len(shots) if shots else target_scenes_count
        shots_per_chapter = max(3, total_shots // num_chapters)

        for ch_idx in range(num_chapters):
            ch_start = ch_idx * chapter_duration
            ch_end = min(target_duration, (ch_idx + 1) * chapter_duration)
            ch_dur = ch_end - ch_start

            shot_slice = None
            if shots:
                s_start = ch_idx * shots_per_chapter
                s_end = (ch_idx + 1) * shots_per_chapter if ch_idx < num_chapters - 1 else len(shots)
                shot_slice = shots[s_start:s_end]

            ch_shots_desc = "\n".join(
                f"- Shot {i} ({getattr(s, 'duration', 4.0):.1f}s): {getattr(s, 'description', str(s))[:120]}"
                for i, s in enumerate(shot_slice or [])
            )

            act_name = {
                0: "Act 1: The Setup & Inciting Dilemma",
                1: "Act 2: The Core Mechanism / Escalating Stakes",
                2: "Act 3: The Deeper Crisis / Turning Point",
                3: "Act 4: The Climax / Peak Encounter",
                4: "Act 5: The Payoff & Resolution",
            }.get(ch_idx, f"Chapter {ch_idx + 1}: Narrative Progression")

            system = (
                f"You are {persona['title']}. {persona['voice']}\n"
                f"You are scripting {act_name} of a full-length ({target_duration/60:.1f} minute) {classification.primary_type}.\n"
                "Maintain complete narrative continuity from previous chapters. Zero filler. Active, engaging momentum."
            )

            prompt = f"""Script {act_name} (Duration: {ch_dur:.0f} seconds).

OVERALL PREMISE: {strategy.new_concept}
PREVIOUS CHAPTER CONTEXT: {prev_context}
LANGUAGE: {language}

SHOT SEQUENCE FOR THIS CHAPTER:
{ch_shots_desc if ch_shots_desc else f'Plan ~{max(3, len(shot_slice or []))} beats for {ch_dur:.0f}s'}

Return ONLY JSON:
{{
  "chapter_summary": "Brief summary of what occurred in this chapter",
  "beats": [
    {{
      "index": 0,
      "role": "establish / explain / demonstrate / escalate / emotional_beat",
      "narration": "Spoken line in {language} for this beat",
      "target_duration": 4.0,
      "visual_direction": "Visual description"
    }}
  ]
}}
"""
            try:
                ch_res = self.router.run_prompt(prompt, system=system, high_reasoning=True, response_json=True)
                if isinstance(ch_res, dict) and ch_res.get("beats"):
                    prev_context = str(ch_res.get("chapter_summary") or prev_context)
                    for b_data in ch_res["beats"]:
                        beat = ScriptBeat.from_dict(b_data, default_index=beat_cursor)
                        beat.index = beat_cursor
                        beat.narration = self._clean_text(beat.narration)
                        all_beats.append(beat)
                        beat_cursor += 1
            except Exception as exc:
                print(f"   Chapter {ch_idx + 1} script generation notice: {exc}")

        if not all_beats:
            return self._fallback_script(strategy, classification, target_duration, target_scenes_count, shots=shots)

        return GeneratedScript(
            title=strategy.new_concept[:60],
            logline=strategy.core_promise,
            format=classification.format,
            primary_type=classification.primary_type,
            total_target_duration=sum(b.target_duration for b in all_beats),
            beats=all_beats,
        )

    def _fallback_script(
        self,
        strategy: CreativeStrategy,
        classification: ContentClassification,
        target_duration: float,
        count: int,
        shots: list | None = None,
    ) -> GeneratedScript:
        avg_dur = max(2.5, target_duration / max(1, count))
        beats = []
        for i in range(count):
            role = "hook" if i == 0 else ("cta" if i == count - 1 else "escalation")
            shot_desc = ""
            if shots and i < len(shots):
                shot_desc = getattr(shots[i], "description", str(shots[i]))
            beats.append(
                ScriptBeat(
                    index=i,
                    role=role,
                    narration="",
                    target_duration=avg_dur,
                    visual_direction=shot_desc or f"Cinematic {classification.primary_type.replace('_', ' ')} action.",
                    on_screen_text="",
                )
            )
        return GeneratedScript(
            title=strategy.new_concept[:50] if strategy and strategy.new_concept else "Original Video",
            logline=strategy.core_promise if strategy and strategy.core_promise else "",
            format=classification.format,
            primary_type=classification.primary_type,
            total_target_duration=target_duration,
            beats=beats,
        )

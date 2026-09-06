"""Comprehensive verification for the AI YouTube Content Engine components.

Run from the project root:
    python tests/test_components.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omniclip.core.classifier import ContentClassification, ContentClassifier
from omniclip.core.dna import ContentDNA
from omniclip.core.jobstore import Job, SceneRecord
from omniclip.core.prompt_engine import PromptEngine
from omniclip.core.qa import VideoQAReport, VideoQA
from omniclip.core.retention import RetentionOptimizer
from omniclip.core.scene_planner import ProductionScene, ScenePlanner, StyleBible
from omniclip.core.script_engine import ScriptEngine
from omniclip.core.strategy import CreativeStrategy


class TestContentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ContentClassifier()

    def test_heuristic_automotive(self):
        cls = self.classifier._heuristic_fallback(
            title="Lamborghini Revuelto V12 Pure Exhaust Sound",
            transcript="Listen to this naturally aspirated monster roar down the highway.",
            duration=45.0,
            aspect="9:16",
            format_hint="short",
        )
        self.assertEqual(cls.primary_type, "automotive_cinematic")
        self.assertEqual(cls.format, "short")
        self.assertEqual(cls.motion_dependency, "high")

    def test_heuristic_tutorial(self):
        cls = self.classifier._heuristic_fallback(
            title="How to Build an AI Agent in 5 Minutes (Step by Step)",
            transcript="In this tutorial I will guide you step by step to build your first agent.",
            duration=350.0,
            aspect="16:9",
            format_hint="long_form",
        )
        self.assertEqual(cls.primary_type, "tutorial")
        self.assertEqual(cls.format, "long_form")

    def test_heuristic_cooking(self):
        cls = self.classifier._heuristic_fallback(
            title="Crispy Garlic Butter Steak Bites Recipe",
            transcript="Melt two tablespoons of butter in a hot skillet and sear until crispy and delicious.",
            duration=55.0,
            aspect="9:16",
            format_hint="short",
        )
        self.assertEqual(cls.primary_type, "food")


class TestContentDNAAndStrategy(unittest.TestCase):
    def test_dna_roundtrip(self):
        dna = ContentDNA(
            hook={"type": "visual_pattern_interrupt", "description": "Screeching tires"},
            audience={"demographic": "Supercar enthusiasts", "intent": "Entertainment"},
            promise={"core_value": "Experience the last naturally aspirated V12"},
            pacing={"profile": "rapid", "cut_frequency": "fast"},
            emotional_arc=["excitement", "tension", "exhilaration", "satisfaction"],
            retention_mechanisms=[{"name": "auditory spike", "timing": "0s"}],
            things_to_avoid=["talking head intro", "slow drone intro", "text heavy disclaimers"],
        )
        d = dna.to_dict()
        rebuilt = ContentDNA.from_dict(d)
        self.assertEqual(rebuilt.hook["type"], "visual_pattern_interrupt")
        self.assertEqual(rebuilt.pacing["profile"], "rapid")
        self.assertEqual(len(rebuilt.things_to_avoid), 3)

    def test_strategy_roundtrip(self):
        strat = CreativeStrategy(
            new_concept="Cyberpunk Street Racing",
            target_audience="Supercar enthusiasts",
            core_promise="Witness an electric prototype challenge the underground",
            hook_strategy="High-speed launch in rainy neon alley",
            story_strategy="Rivalry between heritage and electric power",
            visual_strategy="Anamorphic neon reflections on wet asphalt",
            pacing_strategy="Rapid cuts syncing with acceleration",
            sound_strategy="Turbine whine building into bass drop",
            ending_strategy="Disappearing into city skyline",
            originality_notes=["Completely new vehicle and environment"],
            what_to_preserve=["Rapid tire-smoke hook", "Auditory crescendo before drop"],
            what_must_change=["Replace gas car with neon prototype", "Night city aesthetic instead of desert"],
        )
        d = strat.to_dict()
        rebuilt = CreativeStrategy.from_dict(d)
        self.assertEqual(rebuilt.new_concept, strat.new_concept)
        self.assertEqual(len(rebuilt.what_to_preserve), 2)
        self.assertEqual(len(rebuilt.what_must_change), 2)


class TestScenePlannerAndStyleBible(unittest.TestCase):
    def test_style_bible(self):
        bible = StyleBible(
            visual_style="Cinematic anamorphic, moody twilight rain, reflections",
            cinematography="Low-angle tracking and fast push-ins",
            color_language="Matte dark gray, neon acid green, amber wet asphalt",
            lighting="Volumetric headlights, golden hour rim lighting",
            lens_language="35mm anamorphic with subtle horizontal flare",
            editing_style="Fast rhythmic cuts on beats",
            important_objects=[{"name": "Concept Car", "color": "Matte black with acid green"}],
        )
        d = bible.to_dict()
        rebuilt = StyleBible.from_dict(d)
        self.assertEqual(rebuilt.visual_style, bible.visual_style)
        self.assertEqual(len(rebuilt.important_objects), 1)

    def test_production_scene(self):
        scene = ProductionScene(
            scene_id=0,
            purpose="hook",
            duration=3.5,
            narration="Most engines whisper. This one declares war.",
            visual_prompt="Extreme close-up of a titanium exhaust tip glowing red, spitting flames in slow motion.",
            camera="Fast push-in with subtle handheld shake",
            lighting="Volumetric backlight cutting through tire smoke",
            generation_mode="text_to_video",
        )
        self.assertEqual(scene.purpose, "hook")
        self.assertEqual(scene.generation_mode, "text_to_video")
        d = scene.to_dict()
        rebuilt = ProductionScene.from_dict(d)
        self.assertEqual(rebuilt.scene_id, 0)
        self.assertEqual(rebuilt.purpose, "hook")


class TestPromptEngineAndQA(unittest.TestCase):
    def test_prompt_building(self):
        engine = PromptEngine()
        scene = ProductionScene(
            scene_id=1,
            purpose="demonstrate",
            duration=4.0,
            narration="0 to 60 in under three seconds.",
            visual_prompt="Wide track shot of matte black Porsche accelerating away leaving rubber on asphalt.",
            camera="Low-angle tracking shot moving with the vehicle",
            lighting="Dramatic twilight, wet asphalt reflections",
        )
        bible = StyleBible(
            visual_style="Cinematic 35mm, rainy neon track",
            cinematography="Smooth low tracking",
            color_language="Deep charcoal and neon amber",
            lighting="Wet ground reflections and neon halos",
            lens_language="35mm prime",
            editing_style="Punched-in rhythm",
            important_objects=[{"name": "Porsche 911 GT3 RS", "description": "Matte black with acid green accents"}],
        )
        classification = ContentClassification(
            primary_type="automotive_cinematic",
            format="short",
            motion_dependency="high",
        )
        prompt = engine.build_prompt_for_scene(scene=scene, bible=bible, classification=classification)
        self.assertIn("Porsche", prompt)
        self.assertIn("Low-angle tracking", prompt)

    def test_qa_evaluation(self):
        qa = VideoQA()
        from omniclip.core.scene_planner import ProductionPlan
        plan = ProductionPlan(
            title="Supercar Showcase",
            scenes=[],
            style_bible=StyleBible(
                visual_style="Cinematic",
                cinematography="Dynamic",
                color_language="Natural",
                lighting="Daylight",
                lens_language="35mm",
                editing_style="Rhythmic",
            ),
        )
        scenes_data = [
            {
                "scene_id": 0,
                "purpose": "hook",
                "duration": 3.5,
                "narration": "Listen to this roar.",
                "visual_prompt": "Extreme close-up of titanium exhaust glowing red spitting flames in slow motion with lens flare.",
            },
            {
                "scene_id": 1,
                "purpose": "escalate",
                "duration": 4.0,
                "narration": "Speed is unmatched.",
                "visual_prompt": "Low-angle tracking shot of matte black supercar accelerating down wet track leaving smoke and tire marks.",
            },
        ]
        # Live QA evaluation via Gemini 3.8 Flash
        rep = qa.evaluate_scenes(plan=plan, scenes_data=scenes_data)
        self.assertIsInstance(rep.overall_score, int)
        self.assertGreaterEqual(rep.overall_score, 0)
        self.assertLessEqual(rep.overall_score, 100)
        self.assertIsInstance(rep.scene_results, list)
        self.assertEqual(len(rep.scene_results), 2)


class TestJobstoreExtendedFields(unittest.TestCase):
    def test_job_serialization(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job(directory=Path(tmpdir), url="https://youtube.com/watch?v=123")
            job.classification = {"primary_type": "automotive_cinematic", "format": "short"}
            job.content_dna = {"hook": {"type": "instant audio drop"}}
            job.creative_strategy = {"new_concept": "Original Concept"}
            job.style_bible = {"visual_style": "Moody neon"}
            job.qa_report = {"overall_score": 92, "passed": True}

            record = SceneRecord(
                index=0,
                start=0.0,
                duration=3.5,
                narration="Testing narration",
                visual_query="Testing visual prompt",
                purpose="hook",
                generation_mode="text_to_video",
            )
            job.scenes.append(record)
            job.save()

            rebuilt = Job.load(tmpdir)
            self.assertIsNotNone(rebuilt)
            self.assertEqual(rebuilt.classification["primary_type"], "automotive_cinematic")
            self.assertEqual(rebuilt.scenes[0].purpose, "hook")
            self.assertEqual(rebuilt.scenes[0].generation_mode, "text_to_video")
            self.assertEqual(rebuilt.qa_report["overall_score"], 92)


if __name__ == "__main__":
    unittest.main()

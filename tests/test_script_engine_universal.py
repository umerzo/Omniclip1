"""Test suite for Universal ScriptEngine across all genres and durations."""

from omniclip.core.classifier import ContentClassification
from omniclip.core.dna import ContentDNA
from omniclip.core.script_engine import ScriptEngine, BANNED_AI_WORDS
from omniclip.core.strategy import CreativeStrategy


class MockRouter:
    """Mock router that simulates high-quality responses for different genres."""

    def run_prompt(self, prompt: str, system: str = "", high_reasoning: bool = False, response_json: bool = True):
        # Long-form chapter response
        if "Act 1:" in prompt or "Chapter 1:" in prompt:
            return {
                "chapter_summary": "Introduction to the quantum mystery and the double-slit setup.",
                "beats": [
                    {
                        "index": 0,
                        "role": "hook",
                        "narration": "If you shoot a beam of light through two tiny slits, something impossible happens on the back wall.",
                        "target_duration": 4.5,
                        "visual_direction": "Slow push-in on a dark laboratory table with a laser emitting a clean beam toward a copper barrier with two micro-slits.",
                    },
                    {
                        "index": 1,
                        "role": "explain",
                        "narration": "You would expect two clean lines. Instead, you get a rippling pattern of dozens of stripes.",
                        "target_duration": 4.0,
                        "visual_direction": "Macro camera showing an intricate wave interference pattern glowing gently on the detector screen.",
                    },
                ],
            }
        if "Act 2:" in prompt or "Chapter 2:" in prompt:
            return {
                "chapter_summary": "Testing individual photons and discovering wave-particle duality.",
                "beats": [
                    {
                        "index": 2,
                        "role": "escalate",
                        "narration": "So physicists slowed the laser down to fire one single photon at a time. And the crazy part? The pattern still appeared.",
                        "target_duration": 5.0,
                        "visual_direction": "High-speed camera tracking a single pulsing point of light drifting toward the double slit in slow motion.",
                    },
                ],
            }

        # Science test
        if "science" in system.lower() or "feynman" in system.lower():
            return {
                "title": "Why Hot Water Can Freeze Faster Than Cold Water",
                "logline": "A simple kitchen paradox that baffled Aristotle and modern physicists alike.",
                "format": "short",
                "primary_type": "science",
                "total_target_duration": 30.0,
                "beats": [
                    {
                        "index": 0,
                        "role": "hook",
                        "narration": "If you put boiling water and cold water into the freezer right now, the boiling cup can actually freeze first.",
                        "target_duration": 4.0,
                        "visual_direction": "Side-by-side macro shot of two glass beakers placed on a frost-covered freezer shelf, one steaming hot and one cold.",
                    },
                    {
                        "index": 1,
                        "role": "explain",
                        "narration": "Aristotle noticed this over two thousand years ago, and scientists still debate why.",
                        "target_duration": 3.5,
                        "visual_direction": "Close-up of ancient parchment illustration transitioning into a modern physics laboratory setup.",
                    },
                    {
                        "index": 2,
                        "role": "demonstrate",
                        "narration": "Rapid evaporation shrinks the hot water mass faster, letting it shed heat at an accelerated rate.",
                        "target_duration": 4.0,
                        "visual_direction": "High-speed macro capture of steam wisps rising rapidly from the water meniscus as ice needles begin forming.",
                    },
                ],
            }

        # Explainer test
        if "investigative" in system.lower() or "explainer" in system.lower():
            return {
                "title": "The $2 Chip Controlling World Trade",
                "logline": "How a microscopic piece of silicon became the single biggest bottleneck in modern manufacturing.",
                "format": "short",
                "primary_type": "explainer",
                "total_target_duration": 30.0,
                "beats": [
                    {
                        "index": 0,
                        "role": "hook",
                        "narration": "Inside your smartphone, your car, and your kitchen toaster sits a microscopic chip that costs less than a cup of coffee.",
                        "target_duration": 4.5,
                        "visual_direction": "Extreme macro push-in on a circuit board highlighting a tiny black silicon chip with gold contact pins.",
                    },
                    {
                        "index": 1,
                        "role": "evidence",
                        "narration": "Yet if a single factory in Taiwan halts production for forty-eight hours, global car manufacturing grinds to a dead stop.",
                        "target_duration": 4.5,
                        "visual_direction": "Wide tracking shot over thousands of finished vehicles parked in an empty shipping port waiting for components.",
                    },
                ],
            }

        # Default cinematic story
        return {
            "title": "A Midnight Kitchen Adventure",
            "logline": "Two sisters escape the summer heat to make a late-night feast.",
            "format": "short",
            "primary_type": "cinematic_story",
            "total_target_duration": 30.0,
            "beats": [
                {
                    "index": 0,
                    "role": "setup",
                    "narration": "On the hottest night of July, Ayesha and her sister slipped into the kitchen without making a sound.",
                    "target_duration": 4.0,
                    "visual_direction": "Soft warm amber lighting in a rustic kitchen as two sisters tiptoe across the tiled floor smiling conspiratorially.",
                },
                {
                    "index": 1,
                    "role": "escalate",
                    "narration": "Their plan was simple: whip up fresh mango chutney before anyone woke up.",
                    "target_duration": 3.5,
                    "visual_direction": "Medium shot of Ayesha lifting a wicker basket of ripe green mangoes onto the wooden cutting table.",
                },
            ],
        }


import unittest


class TestUniversalScriptEngine(unittest.TestCase):

    def test_persona_selection(self):
        engine = ScriptEngine(router=MockRouter())
        self.assertIn("Feynman", engine._pick_persona("science")["title"])
        self.assertIn("Feynman", engine._pick_persona("technology")["title"])
        self.assertIn("Investigative", engine._pick_persona("explainer")["title"])
        self.assertIn("Storyteller", engine._pick_persona("cinematic_story")["title"])
        self.assertIn("Observer", engine._pick_persona("documentary")["title"])
        self.assertIn("Mentor", engine._pick_persona("tutorial")["title"])


    def test_anti_slop_cleaning(self):
        engine = ScriptEngine(router=MockRouter())
        dirty = "Let's delve into this rich tapestry which is a testament to human innovation. Think you know cars? Think again!"
        cleaned = engine._clean_text(dirty)
        for banned in BANNED_AI_WORDS:
            self.assertNotIn(banned.lower(), cleaned.lower())

    def test_science_script_generation(self):
        engine = ScriptEngine(router=MockRouter())
        strat = CreativeStrategy(
            new_concept="Why Hot Water Freezes Faster",
            target_audience="Science enthusiasts",
            core_promise="Mind-bending kitchen physics",
            hook_strategy="Counter-intuitive paradox",
            story_strategy="Feynman breakdown",
            visual_strategy="Macro high-speed photography",
            pacing_strategy="Dynamic",
            sound_strategy="Clean narration",
            ending_strategy="Memorable takeaway",
        )
        classification = ContentClassification(
            primary_type="science",
            format="short",
            visual_dependency="high",
            narration_dependency="high",
        )
        dna = ContentDNA(pacing={"words_per_minute": 145})
        script = engine.generate(
            strategy=strat,
            classification=classification,
            dna=dna,
            target_duration=30.0,
            source_transcript="Did you know hot water can freeze faster than cold water? It's called the Mpemba effect.",
        )
        self.assertGreaterEqual(len(script.beats), 2)
        self.assertIn("boiling", script.full_narration.lower())
        for b in script.beats:
            self.assertNotEqual(b.visual_direction, "")
            self.assertNotIn("Scene 1 advancing our story", b.narration)

    def test_story_script_grounding_no_halfway(self):
        engine = ScriptEngine(router=MockRouter())
        strat = CreativeStrategy(
            new_concept="Midnight Kitchen Adventure",
            target_audience="General",
            core_promise="Relatable late night cooking",
            hook_strategy="Warm atmospheric opening",
            story_strategy="Three act escalation",
            visual_strategy="Warm golden cinema",
            pacing_strategy="Dynamic",
            sound_strategy="Acoustic score",
            ending_strategy="Satisfying smile",
        )
        classification = ContentClassification(
            primary_type="cinematic_story",
            format="short",
            visual_dependency="high",
            narration_dependency="high",
        )
        dna = ContentDNA(pacing={"words_per_minute": 145})
        script = engine.generate(
            strategy=strat,
            classification=classification,
            dna=dna,
            target_duration=30.0,
            source_transcript="Two sisters sneak into the kitchen to make midnight snacks.",
        )
        first_beat = script.beats[0]
        self.assertTrue("ayesha" in first_beat.narration.lower() or "kitchen" in first_beat.narration.lower())

    def test_longform_chapter_sequencing(self):
        engine = ScriptEngine(router=MockRouter())
        strat = CreativeStrategy(
            new_concept="The Quantum Double Slit Experiment",
            target_audience="Curious minds",
            core_promise="Deep dive into quantum weirdness",
            hook_strategy="Paradox reveal",
            story_strategy="5 chapter historical and modern investigation",
            visual_strategy="Laboratory macro and wave simulations",
            pacing_strategy="Measured and atmospheric",
            sound_strategy="Subtle ambient synth",
            ending_strategy="Philosophical revelation",
        )
        classification = ContentClassification(
            primary_type="science",
            format="long_form",
            visual_dependency="high",
            narration_dependency="high",
        )
        dna = ContentDNA(pacing={"words_per_minute": 145})
        script = engine.generate(
            strategy=strat,
            classification=classification,
            dna=dna,
            target_duration=240.0,
            source_transcript="The double slit experiment is the central mystery of quantum physics.",
        )
        self.assertGreaterEqual(len(script.beats), 3)
        indices = [b.index for b in script.beats]
        self.assertEqual(indices, list(range(len(script.beats))))


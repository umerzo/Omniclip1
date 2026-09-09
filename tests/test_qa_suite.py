"""Comprehensive QA Verification Suite for OmniClip.

Tests critical paths across:
- Shot analyzer (ffmpeg execution, cuts detection, frame extraction, subprocess flags)
- AI router & JSON repair
- Queue store (atomic write, read, transitions, retries)
- Config resolution (frozen vs non-frozen environments)
- Subtitles & publishing kits
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omniclip.core import shot_analyzer
from omniclip.core.shot_analyzer import Cut, Shot, find_cuts, subdivide
from omniclip.core.router import AIModelRouter, repair_json
from omniclip.core.publisher import format_youtube_timestamp, _clean_title
from omniclip.core.subtitles import Cue, WordTiming, group_words
from omniclip.frontend import queue_store as q
from omniclip.utils.config import load_settings, resolve_path, PROJECT_ROOT


class TestShotAnalyzerQA(unittest.TestCase):
    def test_os_module_available(self):
        """Verify os is imported and usable in shot_analyzer without NameError."""
        self.assertTrue(hasattr(shot_analyzer, "os"))
        self.assertIsNotNone(shot_analyzer.os.name)

    def test_subdivide_shots(self):
        """Test shot subdivision for short vs long duration scenes."""
        shots = [
            Shot(index=0, start=0.0, duration=2.5, description="Short intro"),
            Shot(index=1, start=2.5, duration=14.0, description="Long continuous scene"),
        ]
        split = subdivide(shots, split_after=5.0)
        self.assertGreater(len(split), len(shots))
        total_dur_orig = sum(s.duration for s in shots)
        total_dur_split = sum(s.duration for s in split)
        self.assertAlmostEqual(total_dur_orig, total_dur_split, places=2)

    @patch("subprocess.run")
    def test_find_cuts_subprocess_creationflags(self, mock_run):
        """Verify find_cuts invokes subprocess with proper creationflags and parses correctly."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = (
            "[Parsed_metadata_1 @ 000002] pts_time:1.200\n"
            "[Parsed_metadata_1 @ 000002] lavfi.scene_score=0.45\n"
            "[Parsed_metadata_1 @ 000002] pts_time:3.800\n"
            "[Parsed_metadata_1 @ 000002] lavfi.scene_score=0.72\n"
        )
        mock_run.return_value = mock_proc

        cuts = find_cuts("dummy_video.mp4", threshold=0.1)
        self.assertEqual(len(cuts), 2)
        self.assertEqual(cuts[0].time, 1.2)
        self.assertEqual(cuts[1].time, 3.8)

        # Ensure creationflags argument was passed to subprocess.run
        _, kwargs = mock_run.call_args
        self.assertIn("creationflags", kwargs)


class TestRouterAndJSONQA(unittest.TestCase):
    def test_repair_json_clean(self):
        raw = '{"name": "OmniClip", "version": 2.0}'
        res = repair_json(raw)
        self.assertEqual(res["name"], "OmniClip")

    def test_repair_json_markdown_wrapped(self):
        raw = '```json\n{"scenes": [1, 2, 3], "status": "ok"}\n```'
        res = repair_json(raw)
        self.assertEqual(res["scenes"], [1, 2, 3])

    def test_repair_json_trailing_comma(self):
        raw = '{"items": ["a", "b",],}'
        res = repair_json(raw)
        self.assertEqual(res["items"], ["a", "b"])

    def test_router_initialization(self):
        router = AIModelRouter()
        self.assertTrue(len(router.PRIMARY_MODEL) > 0)
        self.assertTrue(len(router.FALLBACK_MODEL) > 0)


class TestQueueStoreQA(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.orig_queue_path = q.queue_path
        test_file = Path(self.temp_dir.name) / "queue.json"
        q.queue_path = lambda: test_file

    def tearDown(self):
        q.queue_path = self.orig_queue_path
        self.temp_dir.cleanup()

    def test_queue_lifecycle(self):
        # Empty queue initially
        self.assertEqual(len(q.entries()), 0)

        # Add an entry
        entry = q.add("https://youtube.com/shorts/test1234", "test_video", {"aspect": "9:16"})
        self.assertIsNotNone(entry)
        self.assertEqual(entry["state"], q.WAITING)
        self.assertEqual(entry["name"], "test_video")

        # Query waiting entries
        next_job = q.next_waiting()
        self.assertIsNotNone(next_job)
        self.assertEqual(next_job["id"], entry["id"])

        # Transition to running
        q.update(entry["id"], state=q.RUNNING, started=12345.0)
        running_jobs = q.entries(q.RUNNING)
        self.assertEqual(len(running_jobs), 1)

        # Transition to failed with error message
        q.update(entry["id"], state=q.FAILED, message="Test error message")
        failed_jobs = q.entries(q.FAILED)
        self.assertEqual(len(failed_jobs), 1)
        self.assertEqual(failed_jobs[0]["message"], "Test error message")

        # Retry failed job -> back to waiting
        q.update(entry["id"], state=q.WAITING, message="", finished=None)
        retried = next(e for e in q.entries() if e["id"] == entry["id"])
        self.assertEqual(retried["state"], q.WAITING)
        self.assertEqual(retried["message"], "")


class TestPublisherAndSubtitlesQA(unittest.TestCase):
    def test_timestamp_formatting(self):
        self.assertEqual(format_youtube_timestamp(0.0), "00:00")
        self.assertEqual(format_youtube_timestamp(65.0), "01:05")
        self.assertEqual(format_youtube_timestamp(3665.0), "01:01:05")

    def test_title_cleaning(self):
        self.assertEqual(_clean_title('**"Secret Recipe"**'), "Secret Recipe")
        self.assertEqual(_clean_title('*Crispy Chicken*'), "Crispy Chicken")

    def test_group_words(self):
        words = [
            WordTiming(text="Hello", start=0.0, duration=0.4),
            WordTiming(text="world", start=0.5, duration=0.4),
            WordTiming(text="again", start=1.0, duration=0.4),
        ]
        cues = group_words(words, max_words=2)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "Hello world")
        self.assertEqual(cues[1].text, "again")


class TestConfigAndPathsQA(unittest.TestCase):
    def test_load_settings(self):
        settings = load_settings()
        self.assertIn("ingest", settings)
        self.assertIn("shots", settings)
        self.assertIn("output", settings)

    def test_resolve_path(self):
        rel = resolve_path("output")
        self.assertTrue(rel.is_absolute())
        self.assertTrue(str(rel).endswith("output"))

    def test_agnes_keys_integration(self):
        from omniclip.pipeline import agnes_keys
        keys = agnes_keys()
        self.assertEqual(len(keys), 41)
        self.assertTrue(all(k.startswith("sk-") for k in keys))
        self.assertEqual(keys[25], "sk-BdxS94wN03veF2HQXsueNSz0h7S5LFpg6ze4hbFs9kKs1w0m")
    def test_agnes_default_model(self):
        settings = load_settings()
        self.assertEqual(settings.get("visuals", {}).get("agnes_model"), "agnes-video-v2.0")


class TestLanguageProberQA(unittest.TestCase):
    def test_detect_language_english_with_auto_caps(self):
        from omniclip.core.prober import detect_language_from_text_and_meta
        # YouTube returns 150+ automatic_captions including 'ur'
        mock_info = {
            "language": "en-US",
            "automatic_captions": {"ur": [{"ext": "vtt"}], "en": [{"ext": "vtt"}]},
        }
        lang = detect_language_from_text_and_meta("SAMSON: The Rise and Fall of God's Strongest Warrior", mock_info)
        self.assertEqual(lang, "en")

    def test_detect_language_urdu_script(self):
        from omniclip.core.prober import detect_language_from_text_and_meta
        mock_info = {}
        lang = detect_language_from_text_and_meta("ملا نصر الدین اور گدھا ایک سبق آموز کہانی ہے", mock_info)
        self.assertEqual(lang, "ur")

    def test_detect_language_hindi_script(self):
        from omniclip.core.prober import detect_language_from_text_and_meta
        mock_info = {}
        lang = detect_language_from_text_and_meta("पंचतंत्र की कहानियां एक शिक्षाप्रद कहानी", mock_info)
        self.assertEqual(lang, "hi")


class TestVideoQAUpgrades(unittest.TestCase):
    def test_qa_serialization(self):
        from omniclip.core.qa import SceneQAResult, VideoQAReport
        r = SceneQAResult(scene_id=1, quality_score=45, issues=["Melted face"], action="regenerate")
        d = r.to_dict()
        self.assertEqual(d["action"], "regenerate")
        self.assertEqual(d["quality_score"], 45)
        
        rep = VideoQAReport(overall_score=50, passed=False, summary="Defect detected", scene_results=[r], scenes_to_regenerate=[1])
        rep_dict = rep.to_dict()
        self.assertFalse(rep_dict["passed"])
        self.assertEqual(len(rep_dict["scenes_to_regenerate"]), 1)


if __name__ == "__main__":
    unittest.main()

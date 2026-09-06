"""Automated YouTube Publishing Kit Generator.

Generates high-CTR video titles, accurate timestamped YouTube chapters,
SEO-optimized video descriptions, tags, and hashtags from scene timelines
and story DNA.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .jobstore import Job, SceneRecord
from .router import AIModelRouter, get_router


def format_youtube_timestamp(seconds: float) -> str:
    """Format seconds into YouTube chapter timestamp (MM:SS or HH:MM:SS)."""
    total_secs = max(0, int(round(seconds)))
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _clean_title(text: str) -> str:
    """Clean markdown quotes, asterisks, and extra whitespace from titles."""
    text = re.sub(r"^[\"']|[\"']$", "", text.strip())
    text = text.replace("**", "").replace("*", "").strip()
    return text


def _extract_core_topic(job: Job) -> tuple[str, str]:
    """Derive clean topic and context from job source and metadata."""
    title = (job.source.get("title") or "").strip()
    logline = (job.style.get("logline") or job.stages.get("storyboard", {}).get("logline") or "").strip()
    dna = job.content_dna or {}
    hook_text = (dna.get("hook", {}).get("mechanism") or "").strip()
    promise_text = (dna.get("promise", {}).get("core_value") or "").strip()

    subject = title or logline or (job.options or {}).get("idea") or job.directory.name
    context_parts = []
    if title:
        context_parts.append(f"Title: {title}")
    if logline:
        context_parts.append(f"Logline: {logline}")
    if hook_text:
        context_parts.append(f"Hook Mechanism: {hook_text}")
    if promise_text:
        context_parts.append(f"Core Promise: {promise_text}")

    return subject, "\n".join(context_parts)


def _build_fallback_chapters(scenes: list[SceneRecord], total_duration: float) -> list[dict]:
    """Construct valid YouTube chapters (min 3 chapters, starting at 00:00, >= 10s gap)."""
    if not scenes:
        return [
            {"timestamp": "00:00", "seconds": 0.0, "title": "Introduction"},
            {"timestamp": format_youtube_timestamp(max(10.0, total_duration * 0.4)), "seconds": total_duration * 0.4, "title": "Key Insights"},
            {"timestamp": format_youtube_timestamp(max(20.0, total_duration * 0.8)), "seconds": total_duration * 0.8, "title": "Summary & Takeaways"},
        ]

    chapters = []
    first_title = "The Hook & Introduction"
    if scenes[0].purpose:
        first_title = scenes[0].purpose.replace("_", " ").title()
    elif scenes[0].narration:
        first_title = scenes[0].narration.split(".")[0][:40].strip() or "Introduction"

    chapters.append({
        "timestamp": "00:00",
        "seconds": 0.0,
        "title": first_title,
        "scene_index": 0,
    })

    last_time = 0.0
    min_interval = 12.0 if total_duration <= 90 else 25.0

    for s in scenes[1:]:
        if (s.start - last_time) >= min_interval and (total_duration - s.start) >= 5.0:
            title = ""
            if s.purpose and s.purpose.lower() not in ("scene", "generic"):
                title = s.purpose.replace("_", " ").title()
            elif s.narration:
                first_sentence = s.narration.split(".")[0].strip()
                words = first_sentence.split()
                title = " ".join(words[:5]).capitalize() if words else f"Part {s.index + 1}"
            else:
                title = f"Scene {s.index + 1}"

            chapters.append({
                "timestamp": format_youtube_timestamp(s.start),
                "seconds": s.start,
                "title": title,
                "scene_index": s.index,
            })
            last_time = s.start

    if len(chapters) < 3 and total_duration >= 20.0:
        mid_time = total_duration * 0.5
        end_time = total_duration * 0.85
        chapters = [
            {"timestamp": "00:00", "seconds": 0.0, "title": "Introduction & Setup", "scene_index": 0},
            {"timestamp": format_youtube_timestamp(mid_time), "seconds": mid_time, "title": "The Turning Point", "scene_index": len(scenes) // 2},
            {"timestamp": format_youtube_timestamp(end_time), "seconds": end_time, "title": "The Final Revelation", "scene_index": len(scenes) - 1},
        ]

    return chapters


def _fallback_publishing_kit(job: Job) -> dict:
    """Instant deterministic fallback publishing kit if AI services are offline."""
    subject, _ = _extract_core_topic(job)
    scenes = job.scenes or []
    total_duration = sum(s.duration for s in scenes) or float(job.source.get("duration_seconds") or 60.0)

    clean_sub = re.sub(r"[|:–—\-].*$", "", subject).strip() or "This Secret"

    titles = [
        {
            "style": "Curiosity Hook",
            "title": f"The Hidden Truth About {clean_sub[:45]} Nobody Tells You",
            "ctr_explanation": "Creates an immediate curiosity gap that compels viewers to click to find out what is hidden.",
            "char_count": len(f"The Hidden Truth About {clean_sub[:45]} Nobody Tells You"),
        },
        {
            "style": "Direct Value & Search",
            "title": f"Why {clean_sub[:50]} (And How It Changes Everything)",
            "ctr_explanation": "Optimized for high search volume and provides clear, immediate value.",
            "char_count": len(f"Why {clean_sub[:50]} (And How It Changes Everything)"),
        },
        {
            "style": "Contrarian / High Urgency",
            "title": f"Stop Doing This If You Want {clean_sub[:45]}",
            "ctr_explanation": "Challenges conventional wisdom and triggers loss-aversion psychology.",
            "char_count": len(f"Stop Doing This If You Want {clean_sub[:45]}"),
        },
    ]

    chapters = _build_fallback_chapters(scenes, total_duration)
    chapter_text = "\n".join(f"{c['timestamp']} {c['title']}" for c in chapters)

    tags = [
        clean_sub.lower(),
        "motivation",
        "storytelling",
        "lessons",
        "success mindset",
        "productivity",
        "viral story",
        "life advice",
    ]

    description = (
        f"{subject}\n\n"
        f"In this video, discover the deeper story and the powerful lesson that changes how you look at persistence and success.\n\n"
        f"📌 TIMESTAMPS:\n{chapter_text}\n\n"
        f"🔔 Subscribe for more deep-dive cinematic stories and visual breakdowns.\n\n"
        f"#Shorts #Story #Mindset #Success #Inspiration"
    )

    return {
        "titles": titles,
        "chapters": chapters,
        "chapter_text": chapter_text,
        "description": description,
        "tags": tags,
        "hashtags": ["#Shorts", "#Story", "#Mindset", "#Success", "#Inspiration"],
        "generated_by": "deterministic_engine",
    }


def generate_publishing_kit(job: Job, use_ai: bool = True, force: bool = False) -> dict:
    """Generate or retrieve cached YouTube publishing kit for a job."""
    kit_file = job.directory / "publishing_kit.json"

    if not force:
        if "publish" in job.stages and isinstance(job.stages["publish"], dict) and job.stages["publish"].get("titles"):
            return job.stages["publish"]
        if kit_file.exists():
            try:
                cached = json.loads(kit_file.read_text(encoding="utf-8"))
                if cached.get("titles"):
                    job.stages["publish"] = cached
                    return cached
            except Exception:
                pass

    if not use_ai:
        kit = _fallback_publishing_kit(job)
        job.stages["publish"] = kit
        job.save()
        kit_file.write_text(json.dumps(kit, indent=2), encoding="utf-8")
        return kit

    subject, context = _extract_core_topic(job)
    scenes = job.scenes or []
    total_duration = sum(s.duration for s in scenes) or float(job.source.get("duration_seconds") or 60.0)

    scene_bullets = []
    for s in scenes[:25]:
        time_str = format_youtube_timestamp(s.start)
        narr = (s.narration or s.narration_en or "").strip()
        purpose = f"[{s.purpose}]" if s.purpose else ""
        scene_bullets.append(f"- {time_str} {purpose}: {narr[:100]}")
    timeline_str = "\n".join(scene_bullets) or "No detailed scene breakdown available."

    prompt = f"""You are an elite YouTube Growth Strategist & Retention Producer.
Generate a high-converting YouTube Publishing Kit for this video production.

### VIDEO CONTEXT:
{context}
Total Runtime: {total_duration:.1f} seconds ({format_youtube_timestamp(total_duration)})

### SCENE TIMELINE:
{timeline_str}

### REQUIREMENTS:
1. TITLES (Generate exactly 3 titles with high CTR):
   - Option 1 (Curiosity Hook): Irresistible curiosity gap, emotional hook (under 65 chars).
   - Option 2 (Direct Value & Search): Keyword-rich, clear value proposition (under 70 chars).
   - Option 3 (Contrarian / Intrigue): Bold perspective, urgency or contrarian angle (under 65 chars).
   Include for each: 'style', 'title', 'ctr_explanation', and 'char_count'.

2. CHAPTERS (YouTube timestamp chapters):
   - YouTube strictly requires the first chapter to start at "00:00".
   - Each chapter timestamp must be "MM:SS" (or "HH:MM:SS" if > 1 hr) followed by a short, punchy chapter title.
   - Must have at least 3 chapters in ascending order, each spaced at least 10-20 seconds apart according to natural story beats.
   - Align timestamps closely with the scene timeline provided.

3. DESCRIPTION:
   - Compelling 2-sentence hook overview.
   - Bulleted list of key takeaways.
   - Accurate timestamps block (YouTube will auto-index this).
   - Call to Action (Like, Subscribe, Comment).
   - 3 to 5 targeted hashtags (e.g. #Shorts, #Topic).

4. TAGS:
   - 8-12 comma-separated keyword tags for YouTube Studio.

### OUTPUT JSON SCHEMA:
{{
  "titles": [
    {{"style": "Curiosity Hook", "title": "...", "ctr_explanation": "...", "char_count": 52}},
    {{"style": "Direct Value & Search", "title": "...", "ctr_explanation": "...", "char_count": 58}},
    {{"style": "Contrarian / Intrigue", "title": "...", "ctr_explanation": "...", "char_count": 49}}
  ],
  "chapters": [
    {{"timestamp": "00:00", "seconds": 0.0, "title": "..."}},
    {{"timestamp": "00:25", "seconds": 25.0, "title": "..."}},
    {{"timestamp": "01:15", "seconds": 75.0, "title": "..."}}
  ],
  "description": "Full ready-to-copy description...",
  "tags": ["tag1", "tag2", "tag3"],
  "hashtags": ["#tag1", "#tag2"]
}}
"""

    system = "You are a world-class YouTube producer who creates viral, high-retention titles, SEO descriptions, and timestamped chapters."

    try:
        router = get_router()
        result = router.run_prompt(
            prompt=prompt,
            system=system,
            temperature=0.7,
            response_json=True,
            high_reasoning=True,
        )

        if not isinstance(result, dict) or "titles" not in result or "chapters" not in result:
            raise ValueError("Malformed AI publisher response")

        titles = []
        for t in result.get("titles", []):
            clean_t = _clean_title(t.get("title", ""))
            titles.append({
                "style": t.get("style", "Alternative"),
                "title": clean_t,
                "ctr_explanation": t.get("ctr_explanation", ""),
                "char_count": len(clean_t),
            })

        raw_chapters = result.get("chapters", [])
        formatted_chapters = []
        if not raw_chapters or raw_chapters[0].get("timestamp") != "00:00":
            formatted_chapters.append({"timestamp": "00:00", "seconds": 0.0, "title": "Introduction"})

        for c in raw_chapters:
            ts = c.get("timestamp", "00:00")
            title = _clean_title(c.get("title", "Chapter"))
            formatted_chapters.append({
                "timestamp": ts,
                "seconds": float(c.get("seconds", 0.0)),
                "title": title,
            })

        seen_ts = set()
        clean_chapters = []
        for c in formatted_chapters:
            if c["timestamp"] not in seen_ts:
                seen_ts.add(c["timestamp"])
                clean_chapters.append(c)

        if len(clean_chapters) < 3:
            clean_chapters = _build_fallback_chapters(scenes, total_duration)

        chapter_text = "\n".join(f"{c['timestamp']} {c['title']}" for c in clean_chapters)

        description = result.get("description", "")
        if chapter_text not in description:
            description += f"\n\n📌 TIMESTAMPS:\n{chapter_text}"

        kit = {
            "titles": titles,
            "chapters": clean_chapters,
            "chapter_text": chapter_text,
            "description": description.strip(),
            "tags": result.get("tags", []),
            "hashtags": result.get("hashtags", []),
            "generated_by": "gemini_router",
        }

    except Exception as exc:
        print(f"[publisher] AI generation failed ({exc}), using deterministic fallback", flush=True)
        kit = _fallback_publishing_kit(job)

    job.stages["publish"] = kit
    try:
        job.save()
        kit_file.write_text(json.dumps(kit, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as save_err:
        print(f"[publisher] Failed to persist kit: {save_err}", flush=True)

    return kit

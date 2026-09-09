"""Automatic source probe and language configuration.

Quickly inspects a video URL to detect:
1. Spoken audio language (Urdu, Hindi, English, Spanish, Arabic, Punjabi, etc.)
2. Recommended subtitle language (English, same-as-source, etc.)
3. Target platform and aspect ratio (Vertical 9:16 Shorts vs 16:9)
4. Video genre / kind (story, cooking, explainer, vlog, etc.)
"""

from __future__ import annotations

import re
from typing import Any

from .profile import KINDS
from .scraper import fetch_metadata

AUDIO_LANGUAGES = [
    ("auto", "Auto (Detect from Source)"),
    ("en", "English"),
    ("ur", "Urdu"),
    ("hi", "Hindi"),
    ("pa", "Punjabi"),
    ("es", "Spanish"),
    ("ar", "Arabic"),
    ("fr", "French"),
    ("de", "German"),
    ("pt", "Portuguese"),
    ("bn", "Bengali"),
    ("ru", "Russian"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("tr", "Turkish"),
    ("id", "Indonesian"),
    ("it", "Italian"),
]

SUBTITLE_LANGUAGES = [
    ("en", "English"),
    ("same", "Same as Audio"),
    ("ur", "Urdu"),
    ("hi", "Hindi"),
    ("pa", "Punjabi"),
    ("es", "Spanish"),
    ("ar", "Arabic"),
    ("fr", "French"),
    ("de", "German"),
    ("pt", "Portuguese"),
    ("bn", "Bengali"),
    ("ru", "Russian"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("none", "None / No Captions"),
]

LANG_NAME_MAP = {code: name for code, name in AUDIO_LANGUAGES}
LANG_NAME_MAP.update({code: name for code, name in SUBTITLE_LANGUAGES})


def detect_language_from_text_and_meta(text: str, info: dict) -> str:
    """Determine the primary spoken language code using direct metadata, unicode scripts, and text."""
    # 1. Direct language metadata from platform/creator if provided (e.g. 'en', 'en-US', 'hi')
    direct_lang = str(info.get("language") or info.get("audio_language") or "").strip()
    if direct_lang:
        code = direct_lang.split("-")[0].lower()
        if code in LANG_NAME_MAP:
            return code

    # 2. Unicode script in title and description (highest fidelity for non-Latin languages)
    # Arabic / Urdu script
    if re.search(r"[\u0600-\u06FF]", text):
        urdu_markers = [
            "ہے", "ہیں", "تھا", "تھی", "تھے", "نے", "سے", "کو", "کا", "کی", "کے",
            "اور", "میں", "پر", "والا", "والی", "والے", "کر", "کیا", "لیمو", "اچار",
            "ٹافیاں", "کھانا", "بہت", "نقصان", "برباد", "بیٹی", "بابا", "بھائی",
        ]
        if any(w in text for w in urdu_markers):
            return "ur"
        arabic_markers = ["في", "من", "على", "هذا", "هذه", "الله", "ان", "الى", "ما", "لا"]
        if any(w in text for w in arabic_markers):
            return "ar"
        return "ur"

    # Devanagari script (Hindi / Marathi)
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    # Gurmukhi script (Punjabi)
    if re.search(r"[\u0A00-\u0A7F]", text):
        return "pa"

    # Bengali script
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"

    # Chinese / Japanese / Korean
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "zh"
    if re.search(r"[\u3040-\u30FF]", text):
        return "ja"
    if re.search(r"[\uAC00-\uD7AF]", text):
        return "ko"

    # 3. Check author-uploaded manual subtitles (NOT machine auto_caps, which includes 150+ auto-translated tracks)
    subs = list((info.get("subtitles") or {}).keys())
    for lang_code in ["ur", "hi", "pa", "es", "ar", "fr", "de", "pt", "bn", "ru", "tr"]:
        if any(s.lower().startswith(lang_code) for s in subs):
            return lang_code

    # 4. Common Spanish words
    es_markers = [" y ", " en ", " de ", " la ", " el ", " los ", " las ", " por ", " como ", " con "]
    text_lower = f" {text.lower()} "
    if sum(1 for m in es_markers if m in text_lower) >= 3:
        return "es"

    # Default to English
    return "en"


def detect_kind_from_text(text: str) -> str:
    """Classify video type into one of KINDS."""
    t = text.lower()
    if any(w in t for w in ["recipe", "cook", "cooking", "dish", "candy", "candies", "kitchen", "bake", "food", "اچار", "ٹافیاں", "کھانا", "طريقة", "طبخ"]):
        return "cooking" if "cooking" in KINDS else "explainer"
    if any(w in t for w in ["story", "moral", "kahani", "قصة", "کہانی", "life lesson", "father", "daughter", "accident", "emotional"]):
        return "story" if "story" in KINDS else "vlog"
    if any(w in t for w in ["car", "automotive", "supercar", "porsche", "bmw", "drift", "engine", "exhaust"]):
        return "commercial" if "commercial" in KINDS else "explainer"
    if any(w in t for w in ["unboxing", "review", "test", "gadget", "iphone", "camera", "spec"]):
        return "product" if "product" in KINDS else "explainer"
    if any(w in t for w in ["how to", "tutorial", "learn", "diy", "hack", "step by step", "tips"]):
        return "explainer"
    if any(w in t for w in ["news", "update", "report", "breaking", "headline"]):
        return "news" if "news" in KINDS else "explainer"
    return "vlog" if "vlog" in KINDS else "explainer"


def probe_source(url: str) -> dict[str, Any]:
    """Inspect a video URL and return tailored configuration recommendations."""
    info = fetch_metadata(url)
    title = str(info.get("title") or "").strip()
    desc = str(info.get("description") or "").strip()
    duration = float(info.get("duration") or 0.0)

    # 1. Format / Aspect Ratio
    is_short = (
        ("shorts" in url.lower())
        or (0 < duration <= 180 and (info.get("height") or 0) > (info.get("width") or 0))
    )
    aspect = "9:16" if is_short else "16:9"
    platform = "YouTube Shorts (9:16)" if is_short else "YouTube Long-form (16:9)"

    # 2. Audio Language
    combined_text = f"{title} {desc[:800]}"
    audio_lang = detect_language_from_text_and_meta(combined_text, info)
    audio_lang_name = LANG_NAME_MAP.get(audio_lang, audio_lang.upper())

    # 3. Subtitle Language
    sub_lang = "en"
    sub_lang_name = LANG_NAME_MAP.get(sub_lang, "English")

    # 4. Kind of video
    kind = detect_kind_from_text(combined_text)

    summary = (
        f"Detected {audio_lang_name} voice | "
        f"{sub_lang_name} captions recommended | "
        f"{platform} format | "
        f"Category: {kind.title()}"
    )

    return {
        "title": title,
        "duration": duration,
        "aspect": aspect,
        "platform": platform,
        "audio_lang": audio_lang,
        "audio_lang_name": audio_lang_name,
        "sub_lang": sub_lang,
        "sub_lang_name": sub_lang_name,
        "kind": kind,
        "summary": summary,
    }

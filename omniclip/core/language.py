"""Language handling: which voice speaks, and which language the captions use.

A rebuilt video keeps the donor's language by default — a Pashto cartoon should
come back in Pashto — while the captions are written in English so they stay
readable to a wider audience.

That split has a consequence worth stating: word-level caption timing comes
from the speech itself, so it only exists when the captions and the voice are
the same language. When they differ, captions are timed to the scene instead of
to individual words, because an English word has no position inside an Urdu
sentence.
"""

from __future__ import annotations

import asyncio
import functools

NAMES = {
    "en": "English", "ur": "Urdu", "hi": "Hindi", "ps": "Pashto",
    "ar": "Arabic", "fa": "Persian", "bn": "Bengali", "tr": "Turkish",
    "id": "Indonesian", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "it": "Italian", "nl": "Dutch", "pl": "Polish",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "mr": "Marathi",
    "gu": "Gujarati", "pa": "Punjabi", "sw": "Swahili", "th": "Thai",
    "vi": "Vietnamese", "uk": "Ukrainian", "ms": "Malay", "he": "Hebrew",
}

# Voices that sound right for narration, where a language has several.
PREFERRED = {
    "en": "en-US-AndrewNeural",
    "ur": "ur-PK-AsadNeural",
    "hi": "hi-IN-MadhurNeural",
    "ps": "ps-AF-GulNawazNeural",
    "ar": "ar-EG-ShakirNeural",
    "fa": "fa-IR-FaridNeural",
    "bn": "bn-IN-BashkarNeural",
    "tr": "tr-TR-AhmetNeural",
    "id": "id-ID-ArdiNeural",
    "es": "es-ES-AlvaroNeural",
}


def normalise(code: str | None) -> str:
    """'en-US', 'English', 'ur_PK' -> 'en' / 'ur'."""
    if not code:
        return "en"
    text = str(code).strip().lower().replace("_", "-")
    if text in NAMES:
        return text
    head = text.split("-")[0]
    if head in NAMES:
        return head
    for key, name in NAMES.items():
        if name.lower() == text:
            return key
    return head[:2] or "en"


def name_of(code: str) -> str:
    return NAMES.get(normalise(code), normalise(code))


@functools.lru_cache(maxsize=1)
def _voices() -> list[dict]:
    import edge_tts

    try:
        return asyncio.run(edge_tts.list_voices())
    except Exception:
        return []


def voices_for(code: str) -> list[str]:
    code = normalise(code)
    matched = sorted(
        v["ShortName"] for v in _voices()
        if v.get("Locale", "").lower().startswith(f"{code}-")
    )
    if not matched and code == "pa":
        # Punjabi fallback to South Asian neural models
        matched = sorted(
            v["ShortName"] for v in _voices()
            if v.get("Locale", "").lower().startswith(("ur-", "hi-"))
        )
    return matched


def pick_voice(code: str, fallback: str = "en-US-AndrewNeural") -> str:
    """A voice in the requested language, or English if there is none."""
    code = normalise(code)
    available = voices_for(code)
    if not available:
        return fallback
    preferred = PREFERRED.get(code)
    if preferred and preferred in available:
        return preferred
    return available[0]



def word_level_possible(speech_code: str, caption_code: str) -> bool:
    """Word timings only line up when the captions are the spoken language."""
    return normalise(speech_code) == normalise(caption_code)

"""Names the interface is allowed to say.

The engine knows its providers by name. The interface must not: every vendor is
shown as a capability instead, and any text on its way to the screen is filtered
through the same map. This is one small module rather than constants scattered
through the pages, because a single page forgetting to filter is the whole point
of failure.
"""

from __future__ import annotations

import re

ENGINE_LABEL = "OmniClip Engine"

PROVIDER_LABEL = {
    "agnes": "AI generated",
    "pexels": "Stock library",
    "pixabay": "Stock library",
    "": "pending",
}

MODEL_LABEL = {
    "agnes-video-v2.0": "Standard (Default - 1080p)",
    "agnes-video-2.5-flash": "Flash (Fast)",
}
MODEL_VALUE = {label: name for name, label in MODEL_LABEL.items()}

PROVIDER_CLASS = {"agnes": "gen", "pexels": "stock", "pixabay": "stock"}

# Anything that would spell out a vendor, in any casing, is rewritten on sight.
# Longest first, so a specific model name is replaced before the bare vendor.
_MASK = [
    ("agnes-video-2.5-flash", "Engine Fast"),
    ("agnes-video-v2.0", "Engine Standard"),
    ("agnes-image-2.1-flash", "Engine Image"),
    ("apihub.agnes-ai.com", ENGINE_LABEL),
    ("agnes-ai.space", ENGINE_LABEL),
    ("agnes-ai", ENGINE_LABEL),
    ("AGNES_API_KEY", "ENGINE_KEY"),
    ("agnes", "engine"),
    ("Agnes", "Engine"),
    ("AGNES", "ENGINE"),
    ("pixabay", "stock"),
    ("Pixabay", "Stock"),
    ("pexels", "stock"),
    ("Pexels", "Stock"),
    ("openrouter", "language service"),
    ("OpenRouter", "Language service"),
    ("minimax", "language model"),
    ("lyria", "music model"),
    ("edge-tts", "voice engine"),
    ("whisper", "speech recognition"),
]


def mask(text: str) -> str:
    """Strip provider names out of anything shown to a person."""
    if not text:
        return text
    for needle, replacement in _MASK:
        text = text.replace(needle, replacement)
    return text


def provider_label(name: str) -> str:
    return PROVIDER_LABEL.get((name or "").lower(), "AI generated")


def provider_class(name: str) -> str:
    """CSS class for a badge. Never the provider id, which would leak in markup."""
    return PROVIDER_CLASS.get((name or "").lower(), "gen")


def model_label(name: str) -> str:
    return MODEL_LABEL.get(name, "Standard (Default - 1080p)")


def clean_error(text: str) -> str:
    """A failure message fit to show: masked, trimmed, one line."""
    text = mask((text or "").strip())
    text = re.sub(r"\s+", " ", text)
    return text[:200]

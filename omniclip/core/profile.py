"""What kind of video this is, decided once and read by every stage after.

Until now each stage guessed independently. Music mood was keyword-matched
against the logline, so a meditation video got calm music only if the word
"calm" happened to appear in it. Voice was chosen by language alone, so a sleep
video and a news read were spoken by the same person at the same speed. Nothing
knew whether it was building a bedtime piece or a bulletin.

One classification, made early from what we already collect, replaces all of
that guessing. Everything downstream reads the profile instead of inventing its
own answer, which is the same reason the stage table lives in one place.

The defaults here are a starting point, not a verdict: a caller may override any
field, because auto-detection is sometimes wrong and being wrong should cost a
dropdown rather than a video.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# Every kind the pipeline knows how to build, and how each one should sound and
# look. Written as data rather than branches so a new kind is a row, and so the
# interface can show and edit exactly what will be used.
#
#   realism   photoreal | illustrated | donor  (donor = keep the source's look)
#   music     none | bed | prominent
#   wps       words per second of narration; the density of the writing
KIND_DEFAULTS: dict[str, dict] = {
    "news": {
        "realism": "photoreal", "voice_style": "brisk, neutral, factual",
        "voice_rate": "+0%", "music": "none", "captions": True, "wps": 2.6,
        "music_mood": "sparse neutral underscore, almost none, no melody, no vocals",
    },
    "vlog": {
        "realism": "photoreal", "voice_style": "warm, conversational, personal",
        "voice_rate": "+0%", "music": "bed", "captions": True, "wps": 2.5,
        "music_mood": "light modern acoustic bed, unobtrusive, no vocals",
    },
    "explainer": {
        "realism": "donor", "voice_style": "clear, measured, instructive",
        "voice_rate": "+0%", "music": "bed", "captions": True, "wps": 2.6,
        "music_mood": "clean minimal background music, steady pulse, no vocals",
    },
    "story": {
        "realism": "illustrated", "voice_style": "warm, measured, storytelling",
        "voice_rate": "-5%", "music": "bed", "captions": True, "wps": 2.3,
        "music_mood": "gentle cinematic underscore, warm strings, no vocals, sits under a voice",
    },
    "short_film": {
        "realism": "donor", "voice_style": "restrained, cinematic",
        "voice_rate": "+0%", "music": "prominent", "captions": True, "wps": 1.8,
        "music_mood": "cinematic score, atmospheric, no vocals, carries the scene",
    },
    "meditation": {
        "realism": "illustrated", "voice_style": "calm, soft, unhurried",
        "voice_rate": "-20%", "music": "prominent", "captions": False, "wps": 1.2,
        "music_mood": "slow ambient pad, soft drone, no percussion, no vocals, spacious",
    },
    "promo": {
        "realism": "photoreal", "voice_style": "energetic, confident",
        "voice_rate": "+5%", "music": "bed", "captions": True, "wps": 2.8,
        "music_mood": "upbeat modern bed, driving but light, no vocals",
    },
}

KINDS = tuple(KIND_DEFAULTS)

# Photoreal output needs saying out loud. The look is otherwise taken from a
# description of the donor's frames, and an image model's own pull is towards
# illustration -- so a vlog rebuilt without this comes back as a cartoon of a
# vlog.
REALISM_PHRASE = {
    "photoreal": (
        "Photographic realism: real people, natural skin texture and pores, "
        "true-to-life proportions, real fabric, natural light with shallow "
        "depth of field, shot on a full-frame camera. Not an illustration, not "
        "a cartoon, not a 3D render, not painterly"
    ),
    "illustrated": "",   # the donor's own style phrase already carries it
    "donor": "",
}


@dataclass
class SourceProfile:
    """What the pipeline decided to build, and how."""

    kind: str = "story"
    realism: str = "donor"
    voice_style: str = "warm, measured, storytelling"
    voice_rate: str = "-5%"
    music: str = "bed"
    music_mood: str = "gentle cinematic underscore, warm strings"
    captions: bool = True
    wps: float = 2.3
    lyrics: bool = False    # the words are sung, not spoken
    tone: str = ""          # the donor's delivery, in the model's words
    reason: str = ""        # why this kind was chosen, for the interface
    source: str = "default"  # detected | override | default

    @property
    def wpm(self) -> float:
        return self.wps * 60.0

    @property
    def realism_phrase(self) -> str:
        return REALISM_PHRASE.get(self.realism, "")

    @property
    def wants_voice(self) -> bool:
        """Whether this video should be narrated at all.

        A song is not narration. Words transcribed from a music track are the
        song's, not the video's, and rewriting them yields a paraphrase of
        somebody else's lyrics rather than an original script -- which is both
        the wrong output and not ours to publish. Such a source is rebuilt from
        what is on screen, and the score carries the sound.
        """
        return self.wps > 0 and not self.lyrics

    @property
    def wants_music(self) -> bool:
        return self.music != "none"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def for_kind(cls, kind: str, **overrides) -> "SourceProfile":
        base = dict(KIND_DEFAULTS.get(kind) or KIND_DEFAULTS["story"])
        base["kind"] = kind if kind in KIND_DEFAULTS else "story"
        base.update({k: v for k, v in overrides.items() if v is not None})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in base.items() if k in known})

    @classmethod
    def from_dict(cls, data: dict | None) -> "SourceProfile":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


_PROMPT = """Identify what kind of video this is, so it can be rebuilt in the
right register.

TITLE: {title}
LENGTH: {duration:.0f} seconds
HOW OFTEN IT CUTS: about every {native:.1f} seconds
WHAT ITS FRAMES LOOK LIKE: {style}

WHAT IS SAID IN IT (may be empty if it has no speech):
{transcript}

Choose exactly one "kind" from this list:
  news        a bulletin, report or piece to camera about real events
  vlog        a real person talking about their own life or day
  explainer   teaches or explains something; tutorial, listicle, breakdown
  story       a narrated story, fable, parable or motivational piece
  short_film  a cinematic piece carried by pictures, little or no narration
  meditation  sleep, calm, guided relaxation; slow and quiet by design
  promo       an advert or product piece

Also give:
  "realism"    - "photoreal" if the source shows real people and places and the
                 rebuild must look photographic; "illustrated" if it is drawn,
                 animated or stylised; "donor" if it should simply keep the
                 look already described above.
  "lyrics"     - true if the words above are SUNG, that is the words of a song
                 playing in the video, rather than someone speaking or
                 narrating. A video whose only audio is a music track with
                 vocals is lyrics: true.
  "tone"       - the delivery in a short phrase: energy, pace and register.
  "music_mood" - a specific musical brief for this video, as a phrase a
                 composer could work from: instruments, tempo, feel. It must
                 suit THIS video, not the category in general. It must be
                 instrumental: never ask for vocals, singing or a choir, because
                 a sung bed plays as a second narrator over the real one. Say
                 "none" only if music would actively hurt it.
  "reason"     - one short sentence on why this kind, for a person to check.

Return ONLY JSON:
{{"kind": "...", "realism": "...", "lyrics": false, "tone": "...",
  "music_mood": "...", "reason": "..."}}
"""


def detect_profile(brain, title: str, transcript: str, style: str,
                   duration: float, native: float,
                   donor_has_music: bool = True) -> SourceProfile:
    """Work out what kind of video this is from what ingest already gathered.

    Falls back to a narrated story rather than failing: a wrong guess produces
    a watchable video, and an exception here would waste an ingest that
    succeeded.
    """
    from .storyboard import _parse

    spoken = " ".join((transcript or "").split())[:3000] or "(no speech)"
    try:
        raw = brain._complete(_PROMPT.format(
            title=title or "untitled", duration=duration or 0.0,
            native=native or 0.0, style=style or "unknown",
            transcript=spoken,
        ))
        data = _parse(raw)
    except Exception:
        return SourceProfile.for_kind("story", source="default")

    kind = str(data.get("kind") or "").strip().lower()
    if kind not in KIND_DEFAULTS:
        kind = "story"

    profile = SourceProfile.for_kind(
        kind,
        realism=str(data.get("realism") or "").strip().lower() or None,
        tone=str(data.get("tone") or "").strip(),
        reason=str(data.get("reason") or "").strip(),
    )
    profile.source = "detected"
    profile.lyrics = bool(data.get("lyrics"))

    # A brief written for this video beats the category's generic one, which is
    # the whole reason the model is asked for it.
    mood = str(data.get("music_mood") or "").strip()
    if mood.lower() in ("none", "no music", ""):
        if mood:
            profile.music = "none"
    elif len(mood) > 8:
        profile.music_mood = mood

    if profile.realism not in REALISM_PHRASE:
        profile.realism = "donor"

    # A source that plays no music of its own should not be given a score it
    # never had. The kind still decides how loud music sits when there is any.
    if not donor_has_music and profile.kind in ("news", "vlog", "explainer"):
        profile.music = "none"

    return profile


def apply_audio_mode(profile: SourceProfile, mode: str) -> SourceProfile:
    """Force the audio shape a caller asked for.

    "auto" leaves the detected profile alone. The others exist because the
    choice is sometimes editorial rather than detectable: a narrated source can
    be wanted as a silent piece, and no amount of analysis will say so.
    """
    if mode == "voiceover":
        profile.music = "none"
    elif mode == "music":
        profile.wps = 0.0
        profile.captions = False
        if profile.music == "none":
            profile.music = "prominent"
    elif mode == "silent":
        profile.wps = 0.0
        profile.music = "none"
        profile.captions = False
    return profile

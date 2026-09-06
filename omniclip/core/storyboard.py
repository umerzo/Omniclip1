"""Storyboard: one plan that decides both what is said and what is shown.

Earlier versions derived narration from the transcript and visuals from
descriptions of the donor's frames. Two independent processes, so the words and
the pictures drifted apart — a line about two brothers played over a shot of an
empty cabin.

Here a single pass writes both together. The donor's frames still supply the
world (art style, cast, setting) and its cuts still supply the rhythm, but what
appears in each shot is chosen to illustrate the sentence spoken over it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field


class StoryboardError(RuntimeError):
    """Raised when a usable storyboard could not be written."""


def _brief(appearance: str, limit: int) -> str:
    """A character description shortened to whole clauses."""
    text = " ".join((appearance or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(", "), cut.rfind(". "), cut.rfind("; "))
    return (cut[:stop] if stop > limit * 0.4 else cut).rstrip(" ,;.")


@dataclass
class Panel:
    index: int
    start: float
    duration: float
    narration: str = ""
    narration_en: str = ""    # same line in English, for the captions
    action: str = ""          # what happens, in story terms
    shot: str = ""            # framing and camera
    characters: list = field(default_factory=list)
    # Whether the cast of this frame is actually known. False means "unstated",
    # which is not the same as "empty" and must not be drawn as empty.
    people_known: bool = True

    def image_prompt(self, style: str, cast: dict) -> str:
        """A still that illustrates this beat, in the donor's visual world.

        The pieces are labelled rather than run together. Concatenating the
        framing, the action and then a loose paragraph of appearance reads to an
        image model as a list of subjects, and it answers by drawing several of
        them — which is where duplicated and malformed figures come from. Saying
        how many people belong in the frame, once, is what prevents it.
        """
        present = [c for c in self.characters if c in cast]
        parts = []
        if self.shot:
            parts.append(self.shot.strip().rstrip("."))
        if self.action:
            parts.append(self.action.strip().rstrip("."))

        if present:
            # One character can carry its full description. Three cannot: the
            # appearance detail then outweighs the action by ten to one, and the
            # model draws a catalogue of clothing instead of a moment. Crowded
            # frames get each character in brief.
            budget = 220 if len(present) == 1 else 90
            who = "; ".join(
                f"{name} ({_brief(cast[name], budget)})" for name in present)
            count = len(present)
            noun = "person" if count == 1 else "people"
            parts.append(f"Exactly {count} {noun} in frame: {who}")
        elif self.people_known:
            parts.append("No people in frame")
        # Otherwise the frame's occupants were never established -- true of a
        # wordless source, where nothing writes a cast list per shot. Claiming
        # an empty frame there drew the scenery and left the characters out.

        if style:
            parts.append(style.strip().rstrip("."))

        # The image endpoint accepts no negative prompt, so the only place to
        # rule out a duplicated subject is here. Saying what occupies the rest
        # of a wide frame is what stops the model filling it with copies of the
        # subject; stating the absence alone did not.
        if present and len(present) == 1:
            name = present[0]
            parts.append(
                f"Composition: one single figure alone in frame, placed "
                f"off-centre, the remaining space filled by the setting itself. "
                f"No twin of {name}, no mirrored or repeated copy of {name}. If "
                f"the action mentions other people (coworkers, a crowd, "
                f"bystanders, passersby), they must be different, ordinary "
                f"background people, unrelated in appearance to {name} -- never "
                f"another copy of {name}"
            )
        elif present:
            # Crowded frames were still picking up copies of a character at the
            # edges, because the rule that names what fills the space only
            # applied when a single character was present.
            parts.append(
                f"Composition: these {count} figures and nobody else. No extra "
                "onlookers or bystanders at the edges of the frame, and no "
                "repeated copy of any of them; the remaining space is the "
                "setting itself"
            )
        # A generic ban loses whenever the action itself calls for a
        # text-bearing object -- a board being written on, a certificate being
        # held. Naming what to draw there instead of text is what a plain "no
        # lettering" could not achieve on its own.
        text_object = any(
            w in self.action.lower()
            for w in ("chalkboard", "blackboard", "whiteboard", "certificate",
                      "diploma", "newspaper", "book", "letter", "document",
                      "paper", "sign", "poster", "screen", "monitor", "note",
                      "banner", "label")
        )
        if text_object:
            parts.append(
                "Any board, page, certificate, sign or screen in the frame is "
                "rendered as dense abstract marks, scribbles or illegible "
                "squiggles that read as writing from a glance -- never as "
                "actual legible numbers, letters or words. Frame it so its "
                "surface is never squarely readable: seen at a steep oblique "
                "angle, softly out of focus behind the subject, or partly cut "
                "off by the edge of frame"
            )
        else:
            parts.append(
                "Plain surfaces, no lettering, signage or written words")
        return ". ".join(parts) + "."

    @property
    def people_count(self) -> int:
        return len(self.characters)


@dataclass
class Storyboard:
    title: str
    language: str = "en"
    style: str = ""
    logline: str = ""
    cast: dict = field(default_factory=dict)
    panels: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.panels)

    def to_json(self) -> str:
        return json.dumps(
            {
                "title": self.title, "language": self.language,
                "style": self.style, "logline": self.logline,
                "cast": self.cast,
                "panels": [asdict(p) for p in self.panels],
            },
            indent=2, ensure_ascii=False,
        )


# The clip generator tops out here. A scene longer than this cannot be filled
# by one generation, so the compositor loops the clip and the last seconds of
# every scene visibly repeat. Scenes are capped to stay inside it.
MAX_GENERATED_CLIP = 19.0


def target_scene_seconds(duration: float, cap: int = 40,
                         native: float = 0.0,
                         ceiling: float = MAX_GENERATED_CLIP) -> float:
    """How long a scene should run.

    `native` is how long the source's own shots run. When we know it, we follow
    it: a short that cuts every two seconds should be rebuilt as two-second
    scenes, because that pace is most of what makes it feel like the original.
    Deriving the length from total duration alone rebuilt such a short as four
    long takes -- a quarter of its shots and none of its rhythm.

    The floor is what keeps this affordable. Every scene is about a minute of
    generation, so a long video that cuts quickly is not given hundreds of
    scenes; it is given `cap` of them.
    """
    floor = max(2.0, duration / cap)
    if native > 0:
        return min(ceiling, max(floor, native))
    if duration <= 90:
        return min(ceiling, 6.0)
    return min(ceiling, max(12.0, duration / cap))


_PROMPT = """You are adapting a video into an original one of the same kind.

THE DONOR'S WORLD — keep the rebuilt video inside it:
  visual style: {style}
  cast: {cast}

WHAT THE DONOR IS ABOUT (its transcript; do not copy its wording):
{transcript}

Write a storyboard of exactly {count} scenes. Each line below gives the scene's
length and the number of words its narration must be close to. Hitting those
word counts matters: too few and the scene sits in silence, too many and the
speech runs past the picture.

{lengths}

For EACH scene give:
  "narration"  - what is spoken over it, {language_rule} Write close to the word
                 count given for that scene - not fewer. It must read as ONE
                 continuous story across scenes, not as separate captions.
  "narration_en" - the same line written in English, used for subtitles.
                 If the narration is already English, repeat it verbatim.
  "action"     - what is happening on screen at that moment. This must
                 ILLUSTRATE the narration for the same scene. If the narration
                 says a boy receives a bicycle, the action shows a boy receiving
                 a bicycle. Never describe something unrelated to the words.
                 At most 25 words, and ONE thing happening. Several actions at
                 once ("leaning over a sink while water runs over him holding a
                 brush") come back as a deformed figure with the objects fused
                 to it. Do not place objects at a character's mouth or eyes, and
                 never put readable words, signs or logos in the frame.
                 No apparitions. Never describe a ghost, spirit,
                 silhouette, shadow figure, reflection or second presence
                 behind or beside a character. There is no description for such
                 a figure, so the model draws the only person it has one for,
                 and the frame comes back with the character in it twice. And
                 never call a scene a "two-shot" unless two named characters
                 are actually in it.
                 Never make readable text the subject of a scene. Generated
                 images cannot render legible writing, so "solves equations on
                 the board", "reads the letter aloud" or "holds up the
                 certificate" always come back as garbled nonsense. Write the
                 beat through the person instead -- their posture, their hands,
                 the reaction of whoever is watching -- and let any board, page
                 or sign sit behind them or out of frame.
  "shot"       - ONE framing, in a few words: wide establishing, medium
                 two-shot, low-angle close-up, over-the-shoulder. Vary it
                 between scenes; do not repeat the same framing. Never give two
                 framings for one scene ("medium shot, then cuts to wide") --
                 a scene is a single picture and cannot cut to anything.
                 Never ask for a double exposure, a
                 reflection overlapping the subject, a mirror, or a split
                 screen: each of those renders as the same character appearing
                 twice, which is indistinguishable from a defect.
  "characters" - ids from the cast above who appear, as a list. [] if none.

Also give a one-line "logline" for the whole piece.

Return ONLY JSON:
{{"logline": "...", "panels": [
  {{"index": 0, "narration": "...", "narration_en": "...", "action": "...",
    "shot": "...", "characters": ["..."]}}
]}}
"""


def _parse(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace:
            raise StoryboardError(f"model did not return JSON: {raw[:200]}")
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError as exc:
            raise StoryboardError(f"malformed JSON: {exc}") from exc


def write_storyboard(
    brain,
    title: str,
    transcript: str,
    shots,
    style: str,
    cast: list[dict],
    language: str = "en",
    language_name: str = "English",
    wpm: float = 145.0,
) -> Storyboard:
    """One request that decides the words and the pictures together."""
    if not shots:
        raise StoryboardError("no shots to build a storyboard from")

    cast_map = {c["id"]: c["appearance"] for c in (cast or [])}
    cast_text = "; ".join(f"{k}: {v}" for k, v in cast_map.items()) or "none recurring"
    # Spelled out per scene, because a single words-per-minute rule gets
    # applied to the example rather than to each scene's real length, and the
    # narration comes back far too short for long scenes.
    lengths = "\n".join(
        f"  scene {i}: {s.duration:.0f}s -> about "
        f"{max(6, int(s.duration / 60 * wpm))} words"
        for i, s in enumerate(shots)
    )
    language_rule = (
        f"written in {language_name}." if language.lower() not in ("en", "")
        else "written in English."
    )

    raw = brain._complete(_PROMPT.format(
        style=style or "consistent cinematic look",
        cast=cast_text,
        transcript=" ".join(transcript.split())[:6000],
        count=len(shots),
        lengths=lengths,
        language_rule=language_rule,
        wpm=int(wpm),
    ))
    data = _parse(raw)

    board = Storyboard(
        title=title, language=language, style=style,
        logline=str(data.get("logline") or "").strip(), cast=cast_map,
    )
    panels = {int(p["index"]): p for p in (data.get("panels") or [])
              if str(p.get("index", "")).lstrip("-").isdigit()}

    for position, shot in enumerate(shots):
        raw_panel = panels.get(position) or panels.get(shot.index) or {}
        board.panels.append(Panel(
            index=position,
            start=shot.start,
            duration=shot.duration,
            narration=str(raw_panel.get("narration") or "").strip(),
            narration_en=str(raw_panel.get("narration_en")
                             or raw_panel.get("narration") or "").strip(),
            action=str(raw_panel.get("action") or "").strip(),
            shot=str(raw_panel.get("shot") or "").strip(),
            characters=[c for c in (raw_panel.get("characters") or [])
                        if c in cast_map],
        ))

    if not any(p.action for p in board.panels):
        raise StoryboardError("storyboard came back with no on-screen action")
    return board


_PERSON_WORDS = (
    "man", "woman", "boy", "girl", "child", "kid", "person", "people",
    "figure", "he ", "she ", "his ", "her ", "they ", "someone", "father",
    "mother", "old", "young", "character", "robot", "creature",
)


def _mentions_a_person(description: str) -> bool:
    text = description.lower()
    return any(word in text for word in _PERSON_WORDS)


def _cast_in(description: str, cast: dict) -> list[str]:
    """Which known characters this shot describes.

    Matched on every word of the character's id, so "old_man" does not claim a
    shot that only says "young man" -- the overlap on the shared word is what
    would otherwise put the whole cast in every frame.
    """
    text = description.lower()
    present = []
    for name in cast:
        words = [w for w in re.split(r"[_\s]+", name.lower()) if len(w) > 2]
        if words and all(w in text for w in words):
            present.append(name)
    return present


def silent_storyboard(shots, style: str, cast: list[dict], title: str) -> Storyboard:
    """A storyboard with pictures but no words.

    Used when the donor genuinely has no speech. The shots still carry the
    original's rhythm and its frames still describe what is on screen, so the
    rebuild matches a mute source instead of inventing narration for it.
    """
    cast_map = {c["id"]: c["appearance"] for c in (cast or [])}
    board = Storyboard(title=title, language="", style=style,
                       logline="visual storytelling, no narration",
                       cast=cast_map)
    for position, shot in enumerate(shots):
        description = (shot.description or style).strip()
        present = _cast_in(description, cast_map)
        board.panels.append(Panel(
            index=position, start=shot.start, duration=shot.duration,
            narration="", narration_en="",
            action=description, shot="", characters=present,
            # Only trust an empty cast when the shot mentions nobody at all.
            people_known=bool(present) or not _mentions_a_person(description),
        ))
    return board


_VISUAL_PROMPT = """You are making an ORIGINAL wordless short film, using another
film only as a reference for its shape.

THE LOOK to stay inside:
  visual style: {style}
  recurring characters: {cast}

THE REFERENCE FILM'S BEATS, in order. These tell you what KIND of story it is
and how it moves. They are NOT to be reproduced:
{beats}

Write {count} scenes that tell your OWN version of this kind of story.

HARD RULES — every one of these exists because breaking it produced a bad film:

1. INVENT. Do not restate the beats above. Same kind of story, same emotional
   arc, your own images. Never describe a specific logo, sign, brand or any
   written words appearing in frame.

2. ONE ACTION PER SCENE. Describe a single, simple thing happening. Not "he
   leans over a sink while water runs over his head and he holds a toothbrush
   in his left hand" — that is three actions at once, and it comes back as a
   deformed figure with objects fused to its face.

3. KEEP IT SHORT. Each "action" must be at most 25 words. Name the subject, what
   they are doing, and where. Leave out clothing details, colours of background
   objects, and anything a viewer would not notice.

3b. NEVER MAKE READABLE TEXT THE SUBJECT. Generated images cannot render legible
   writing, so a scene built on a board of equations, a letter being read or a
   certificate held up to camera always returns garbled nonsense. Carry the beat
   on the person instead, and leave any page or board behind them or out of
   frame.

3c. NO APPARITIONS. No apparitions. Never describe a ghost, spirit,
                 silhouette, shadow figure, reflection or second presence
                 behind or beside a character. There is no description for such
                 a figure, so the model draws the only person it has one for,
                 and the frame comes back with the character in it twice. And
                 never call a scene a "two-shot" unless two named characters
                 are actually in it.

4. NO OBJECTS AT THE FACE. Do not put things in or near a character's mouth,
   eyes or hands in close-up. Those are the frames that come back malformed.

5. TELL IT WITHOUT WORDS. There is no narration and no dialogue. The story has
   to read from the pictures alone, so each scene must show a clear, legible
   moment — a decision, a reaction, an arrival, a loss.

For EACH scene give:
  "action"     - the single thing happening on screen. At most 25 words.
  "shot"       - ONE framing in a few words: wide establishing, medium two-shot,
                 low-angle, over-the-shoulder. Vary it; never repeat one twice
                 in a row, and never give two framings for one scene.
                 Never ask for a double exposure, a
                 reflection overlapping the subject, a mirror, or a split
                 screen: each of those renders as the same character appearing
                 twice, which is indistinguishable from a defect.
  "characters" - ids from the cast above who actually appear, as a list. Use []
                 for a scene with no people in it.

Also give a one-line "logline" describing the story you invented.

Return ONLY JSON:
{{"logline": "...", "panels": [
  {{"index": 0, "action": "...", "shot": "...", "characters": ["..."]}}
]}}
"""


def write_visual_storyboard(
    brain,
    title: str,
    shots,
    style: str,
    cast: list[dict],
) -> Storyboard:
    """A storyboard for a source that says nothing.

    The donor's shots supply the rhythm and the kind of story; the model
    supplies the story itself. Without this pass the rebuild was a transcription
    of the original's frames rather than a film of its own.
    """
    if not shots:
        raise StoryboardError("no shots to build a storyboard from")

    cast_map = {c["id"]: c["appearance"] for c in (cast or [])}
    cast_text = "; ".join(f"{k}: {v}" for k, v in cast_map.items()) or "none recurring"
    beats = "\n".join(
        f"  {i}. ({s.duration:.0f}s) {' '.join((s.description or '').split())[:180]}"
        for i, s in enumerate(shots)
    )

    raw = brain._complete(_VISUAL_PROMPT.format(
        style=style or "consistent cinematic look",
        cast=cast_text,
        beats=beats,
        count=len(shots),
    ))
    data = _parse(raw)

    board = Storyboard(
        title=title, language="", style=style,
        logline=str(data.get("logline") or "").strip(), cast=cast_map,
    )
    panels = {int(p["index"]): p for p in (data.get("panels") or [])
              if str(p.get("index", "")).lstrip("-").isdigit()}

    for position, shot in enumerate(shots):
        raw_panel = panels.get(position) or panels.get(shot.index) or {}
        action = str(raw_panel.get("action") or "").strip()
        board.panels.append(Panel(
            index=position, start=shot.start, duration=shot.duration,
            narration="", narration_en="",
            action=action or (shot.description or style).strip(),
            shot=str(raw_panel.get("shot") or "").strip(),
            characters=[c for c in (raw_panel.get("characters") or [])
                        if c in cast_map],
            # Fall back to the old guess only when the model gave us nothing.
            people_known=bool(action),
        ))

    if not any(p.action for p in board.panels):
        raise StoryboardError("the visual storyboard came back empty")
    return board

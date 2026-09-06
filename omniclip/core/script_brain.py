"""Script brain.

Rewrites a source transcript into an original script of matching tone, pacing
and runtime, and emits a visual search/generation prompt for every scene.

Long transcripts are processed as a sliding window of ~5-minute chapters so no
single request has to carry a 10,000-word transcript. Each window is given the
tail of the previous one so the rewritten chapters join without seams.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from .scraper import SourceVideo, TranscriptSegment

# Agnes caps a generated clip at 20 seconds; scenes are sized to fit one clip.
MAX_SCENE_SECONDS = 20.0


class ScriptError(RuntimeError):
    """Raised when the LLM cannot produce a usable script."""


@dataclass
class StyleProfile:
    tone: str
    pacing_wpm: float
    structure: str
    visual_style: str

    def prompt_fragment(self) -> str:
        return (
            f"Tone: {self.tone}\n"
            f"Structure: {self.structure}\n"
            f"Visual style to keep constant across every scene: {self.visual_style}"
        )


@dataclass
class Scene:
    index: int
    narration: str
    visual_query: str
    start: float = 0.0
    duration: float = 0.0
    # Distinguishes repeat requests for the same prompt, so a repaired scene
    # gets a new clip rather than the cached one it is replacing.
    variant: int = 0

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class ScriptPlan:
    title: str
    style: StyleProfile
    scenes: list[Scene] = field(default_factory=list)

    @property
    def narration(self) -> str:
        return " ".join(scene.narration.strip() for scene in self.scenes)

    @property
    def word_count(self) -> int:
        return len(self.narration.split())

    @property
    def duration_seconds(self) -> float:
        return self.scenes[-1].end if self.scenes else 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "title": self.title,
                "style": asdict(self.style),
                "scenes": [asdict(scene) for scene in self.scenes],
            },
            indent=2,
            ensure_ascii=False,
        )


@dataclass
class TranscriptChunk:
    index: int
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def chunk_transcript(
    segments: list[TranscriptSegment],
    window_seconds: float = 300.0,
    overlap_seconds: float = 20.0,
) -> list[TranscriptChunk]:
    """Split a timed transcript into overlapping windows of roughly equal length."""
    if not segments:
        return []

    chunks: list[TranscriptChunk] = []
    window_start = segments[0].start
    buffer: list[TranscriptSegment] = []

    for segment in segments:
        buffer.append(segment)
        if segment.end - window_start < window_seconds:
            continue
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                text=" ".join(s.text for s in buffer),
                start=window_start,
                end=segment.end,
            )
        )
        # Carry the trailing overlap into the next window for continuity.
        cutoff = segment.end - overlap_seconds
        buffer = [s for s in buffer if s.end > cutoff]
        window_start = buffer[0].start if buffer else segment.end

    if buffer:
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                text=" ".join(s.text for s in buffer),
                start=window_start,
                end=buffer[-1].end,
            )
        )
    return chunks


_STYLE_PROMPT = """Analyse this video transcript excerpt and describe its style.

Return ONLY a JSON object with these keys:
  "tone"         - the delivery in a short phrase: energy, pace and register
  "pacing_wpm"   - estimated words spoken per minute, as a number
  "structure"    - the beat structure of the video, in order
  "visual_style" - one reusable phrase describing the look every scene should
                   share: subject matter, lighting and colour grade

Describe only what is actually present in the excerpt. Do not assume a genre,
and do not import conventions from videos this one resembles.

TRANSCRIPT EXCERPT:
{excerpt}
"""

_REWRITE_PROMPT = """You are rewriting a video into a brand-new, original script.

Write ORIGINAL prose. Cover the same subject matter and keep the same rhythm and
structure, but do not reuse the source's sentences, phrasing or distinctive images.

{style}

This is chapter {index} of {total}, covering roughly {duration:.0f} seconds of
runtime. Write about {target_words} words for it — close to that number matters,
because the narration has to fill the runtime.

Split your writing into scenes of at most {max_scene_seconds:.0f} seconds of
speech each. Give every scene a "visual_query": a concrete, filmable description
of what is on screen, always ending with the shared visual style above so the
scenes look like one coherent video.

{continuity}

Return ONLY a JSON object of the form:
{{"scenes": [{{"narration": "...", "visual_query": "..."}}]}}

SOURCE CHAPTER TO REWRITE:
{chunk}
"""


_CONTINUOUS_PROMPT = """You are rewriting a video into a brand-new, original script.

Write ORIGINAL prose covering the same subject matter as the source. Do not
reuse its sentences, phrasing, or distinctive images.

{style}

Write roughly {target_words} words — this is spoken aloud over {duration:.0f}
seconds, so length matters.

Write it as ONE continuous narration, the way it would actually be spoken:
sentences of varying length, ideas that develop across several sentences, a
beginning that hooks and an ending that lands. Do NOT write it as a list of
captions, do not number anything, and do not label scenes. It is one piece of
writing, not fourteen separate ones.

Return ONLY JSON: {{"narration": "the whole script as one string"}}

SOURCE TRANSCRIPT (subject matter only — do not copy it):
{transcript}

WHAT IS ON SCREEN, in order, for context only. Do not write to these
individually; they are here so the words suit the pictures:
{shots}
"""


_SHOT_SCRIPT_PROMPT = """You are rewriting a video into a brand-new, original script.

Write ORIGINAL prose covering the same subject matter as the source. Do not
reuse its sentences, phrasing, or distinctive images.

{style}

The video is built from the shots listed below, in order. Write the narration
that plays over each one. Match the length: about {wpm:.0f} spoken words per
minute, so a {example:.0f}-second shot needs roughly {example_words} words.
Going long is worse than going short — overrunning narration desynchronises the
edit.

The narration must read as one continuous piece across shots, not as separate
captions. {continuity}

Return ONLY JSON of the form:
{{"shots": [{{"index": <shot number>, "narration": "..."}}]}}
Include every shot listed, using its exact index.

SOURCE TRANSCRIPT (for subject matter only — do not copy it):
{transcript}

SHOTS TO WRITE FOR:
{shots}
"""


_SENTENCE = re.compile(r"[^.!?…]+[.!?…]+[\"')\]]*\s*|[^.!?…]+$")


def distribute(narration: str, shots) -> list[str]:
    """Lay a continuous script across shots in proportion to their length.

    Splitting happens at sentence boundaries so no shot begins or ends
    mid-thought. Longer shots take more sentences; every shot gets at least
    one while sentences remain.
    """
    sentences = [s.strip() for s in _SENTENCE.findall(narration) if s.strip()]
    if not sentences or not shots:
        return ["" for _ in shots]

    total_duration = sum(s.duration for s in shots) or 1.0
    total_words = sum(len(s.split()) for s in sentences)

    parts: list[list[str]] = [[] for _ in shots]
    cursor = 0
    used = 0

    for position, shot in enumerate(shots):
        remaining_shots = len(shots) - position
        if cursor >= len(sentences):
            break
        # Always leave enough sentences for the shots still to come.
        spare = len(sentences) - cursor - (remaining_shots - 1)
        if spare <= 1:
            parts[position].append(sentences[cursor])
            cursor += 1
            continue

        budget = (shot.duration / total_duration) * total_words
        taken = 0
        while cursor < len(sentences) and taken < budget and spare > 0:
            words = len(sentences[cursor].split())
            parts[position].append(sentences[cursor])
            taken += words
            used += words
            cursor += 1
            spare -= 1

    # Anything left over joins the final shot rather than being dropped.
    if cursor < len(sentences):
        parts[-1].extend(sentences[cursor:])

    return [" ".join(p).strip() for p in parts]


def _parse_json(raw: str) -> dict:
    """LLMs wrap JSON in prose and code fences more often than not."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace:
            raise ScriptError(f"Model did not return JSON: {raw[:200]}")
        return json.loads(brace.group(0))


class ScriptBrain:
    """Wraps any OpenAI-compatible endpoint (freellmapi, Gemini, OpenRouter)."""

    def __init__(
        self,
        base_url: str = "http://localhost:3001/v1",
        api_key: str = "freellmapi-local",
        model: str = "auto",
        temperature: float = 0.8,
        max_retries: int = 3,
        default_wpm: float = 145.0,
        max_completion_tokens: int = 8000,
        reasoning_effort: str = "",
        fallback: dict | None = None,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.default_wpm = default_wpm
        self.max_retries = max(1, max_retries)
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        # Used only once the primary provider is out of quota.
        self.fallback = fallback
        self._fallback_client = (
            OpenAI(base_url=fallback["base_url"], api_key=fallback["api_key"],
                   max_retries=0)
            if fallback else None
        )

    def _complete(self, prompt: str, max_tokens: int | None = None) -> str:
        """One completion, with backoff for per-minute token limits.

        Two constraints pull against each other. Reasoning models spend
        completion budget thinking before they write, so the cap must be
        generous or the reply comes back empty. But providers bill the
        *reserved* budget against the per-minute quota, so an oversized cap is
        rejected outright. Callers therefore ask for what they actually need.
        """
        from openai import APIStatusError, RateLimitError

        kwargs = {"max_completion_tokens": max_tokens or self.max_completion_tokens}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        delay = 20.0
        last_error = None

        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            except (RateLimitError, APIStatusError) as exc:
                if getattr(exc, "status_code", None) not in (413, 429):
                    raise
                last_error = exc
                time.sleep(delay)
                delay *= 2
                continue

            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            if content:
                return content

            if choice.finish_reason == "length":
                raise ScriptError(
                    "The model spent its whole token budget reasoning and returned "
                    f"nothing. Raise max_completion_tokens (currently "
                    f"{kwargs['max_completion_tokens']})."
                )
            last_error = ScriptError("Model returned empty content")

        if self._fallback_client:
            try:
                response = self._fallback_client.chat.completions.create(
                    model=self.fallback["model"],
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=self.fallback.get(
                        "max_completion_tokens", self.max_completion_tokens
                    ),
                )
                content = (response.choices[0].message.content or "").strip()
                if content:
                    return content
            except Exception as exc:
                last_error = exc

        raise ScriptError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    # The style pass returns four short strings; it needs far less headroom
    # than a chapter rewrite, and asking for less keeps it inside the quota.
    STYLE_TOKENS = 2048

    def analyse_style(self, source: SourceVideo) -> StyleProfile:
        excerpt = " ".join(source.transcript.split()[:600])
        data = _parse_json(
            self._complete(_STYLE_PROMPT.format(excerpt=excerpt), self.STYLE_TOKENS)
        )
        return StyleProfile(
            tone=data.get("tone", "neutral narration"),
            pacing_wpm=float(data.get("pacing_wpm") or source.spoken_wpm or self.default_wpm),
            structure=data.get("structure", "intro, body, outro"),
            visual_style=data.get("visual_style", "cinematic, consistent colour grade"),
        )

    def rewrite_chunk(
        self,
        chunk: TranscriptChunk,
        style: StyleProfile,
        total_chunks: int,
        continuity: str = "",
    ) -> list[dict]:
        target_words = max(30, int((chunk.duration / 60) * style.pacing_wpm))
        prompt = _REWRITE_PROMPT.format(
            style=style.prompt_fragment(),
            index=chunk.index + 1,
            total=total_chunks,
            duration=chunk.duration,
            target_words=target_words,
            max_scene_seconds=MAX_SCENE_SECONDS,
            continuity=(
                f"The previous chapter ended with: \"{continuity}\"\n"
                "Continue naturally from there without repeating it."
                if continuity
                else "This is the opening chapter. Start with a hook."
            ),
            chunk=chunk.text,
        )
        data = _parse_json(self._complete(prompt))
        scenes = data.get("scenes") or []
        if not scenes:
            raise ScriptError(f"Chapter {chunk.index + 1} produced no scenes")
        return scenes

    def write_continuous(
        self,
        source: SourceVideo,
        shots,
        style: StyleProfile,
        progress=None,
    ) -> ScriptPlan:
        """Write one flowing script, then lay it onto the shot grid.

        Writing per shot forces a sentence-sized box on every idea: a seven
        second shot holds about seventeen words, so the script comes back as a
        row of flat declaratives that never develop. Composing the whole
        narration first and splitting it afterwards keeps the prose continuous
        while the picture still cuts on the source's beats.
        """
        if not shots:
            raise ScriptError("No shots to write narration for")

        if progress:
            progress(1, 1)

        total = sum(s.duration for s in shots)
        wpm = style.pacing_wpm or source.spoken_wpm or self.default_wpm
        target = max(30, int(total / 60 * wpm))
        listing = "\n".join(
            f"  {s.index} ({s.duration:.0f}s): "
            f"{(s.description or style.visual_style)[:150]}"
            for s in shots
        )

        prompt = _CONTINUOUS_PROMPT.format(
            style=style.prompt_fragment(),
            target_words=target,
            duration=total,
            transcript=" ".join(source.transcript.split())[:6000],
            shots=listing,
        )
        narration = str(
            _parse_json(self._complete(prompt)).get("narration") or ""
        ).strip()
        if not narration:
            raise ScriptError("The model returned no narration")

        plan = ScriptPlan(title=f"{source.title} (OmniClip original)", style=style)
        for shot, text in zip(shots, distribute(narration, shots)):
            plan.scenes.append(
                Scene(
                    index=shot.index,
                    narration=text,
                    visual_query=(shot.description or style.visual_style).strip(),
                    start=shot.start,
                    duration=shot.duration,
                )
            )
        return plan

    def write_for_shots(
        self,
        source: SourceVideo,
        shots,
        style: StyleProfile,
        batch_size: int = 10,
        progress=None,
    ) -> ScriptPlan:
        """Write narration onto a shot grid taken from the source video.

        This is what keeps the rebuilt video laid out like the original: the
        shots decide where the cuts fall and how long each one runs, and the
        script is written to fit them, rather than the script inventing its own
        pacing and ignoring how the source was actually edited.
        """
        if not shots:
            raise ScriptError("No shots to write narration for")

        wpm = style.pacing_wpm or source.spoken_wpm or self.default_wpm
        transcript = " ".join(source.transcript.split())
        plan = ScriptPlan(title=f"{source.title} (OmniClip original)", style=style)
        narration_by_index: dict[int, str] = {}
        batches = [shots[i:i + batch_size] for i in range(0, len(shots), batch_size)]
        tail = ""

        for number, batch in enumerate(batches, start=1):
            if progress:
                progress(number, len(batches))

            listing = "\n".join(
                f"  {s.index}: {s.duration:.1f}s"
                f" — {(s.description or style.visual_style)[:200]}"
                f" (about {max(4, int(s.duration / 60 * wpm))} words)"
                for s in batch
            )
            # A long transcript would swamp the request; the shots carry the
            # detail, so the source only has to supply subject matter.
            budget = 6000 if len(batches) == 1 else 3000
            prompt = _SHOT_SCRIPT_PROMPT.format(
                style=style.prompt_fragment(),
                wpm=wpm,
                example=batch[0].duration,
                example_words=max(4, int(batch[0].duration / 60 * wpm)),
                continuity=(
                    f'The previous shot ended with: "{tail}" Continue from there.'
                    if tail else "This is the opening — start with a hook."
                ),
                transcript=transcript[:budget],
                shots=listing,
            )

            data = _parse_json(self._complete(prompt))
            for entry in data.get("shots") or []:
                try:
                    index = int(entry.get("index"))
                except (TypeError, ValueError):
                    continue
                text = (entry.get("narration") or "").strip()
                if text:
                    narration_by_index[index] = text
            if narration_by_index:
                last = narration_by_index.get(batch[-1].index, "")
                tail = " ".join(last.split()[-25:])

        for shot in shots:
            narration = narration_by_index.get(shot.index, "")
            plan.scenes.append(
                Scene(
                    index=shot.index,
                    narration=narration,
                    visual_query=(shot.description or style.visual_style).strip(),
                    start=shot.start,
                    duration=shot.duration,
                )
            )

        missing = [s for s in plan.scenes if not s.narration]
        if len(missing) == len(plan.scenes):
            raise ScriptError("The model returned no narration for any shot")

        # A skipped index would otherwise reach TTS as empty text. Ask again
        # for just those shots rather than leaving silent gaps in the edit.
        for scene in missing:
            words = max(4, int(scene.duration / 60 * wpm))
            prompt = (
                f"{style.prompt_fragment()}\n\n"
                f"Write about {words} words of original narration for one shot "
                f"lasting {scene.duration:.1f} seconds, showing: "
                f"{scene.visual_query[:300]}\n\n"
                f"It sits inside this video: {transcript[:1200]}\n\n"
                'Return ONLY JSON: {"narration": "..."}'
            )
            try:
                scene.narration = str(
                    _parse_json(self._complete(prompt)).get("narration") or ""
                ).strip()
            except ScriptError:
                continue

        return plan

    def build(
        self,
        source: SourceVideo,
        window_seconds: float = 300.0,
        overlap_seconds: float = 20.0,
        progress=None,
    ) -> ScriptPlan:
        """Run the full sliding-window rewrite over a source video."""
        chunks = chunk_transcript(source.segments, window_seconds, overlap_seconds)
        if not chunks:
            raise ScriptError("Source video has no transcript to rewrite")

        style = self.analyse_style(source)
        plan = ScriptPlan(title=f"{source.title} (OmniClip original)", style=style)
        continuity = ""

        for chunk in chunks:
            if progress:
                progress(chunk.index + 1, len(chunks))
            for raw_scene in self.rewrite_chunk(chunk, style, len(chunks), continuity):
                narration = (raw_scene.get("narration") or "").strip()
                if not narration:
                    continue
                plan.scenes.append(
                    Scene(
                        index=len(plan.scenes),
                        narration=narration,
                        visual_query=(raw_scene.get("visual_query") or style.visual_style).strip(),
                    )
                )
            if plan.scenes:
                continuity = " ".join(plan.scenes[-1].narration.split()[-25:])

        _lay_out_timeline(plan, style.pacing_wpm)
        return plan


def _lay_out_timeline(plan: ScriptPlan, wpm: float) -> None:
    """Assign provisional start/duration to each scene from its word count.

    These are estimates for planning and asset sizing only. The subtitle
    aligner replaces them with real word-level timings once TTS has run.
    """
    cursor = 0.0
    for scene in plan.scenes:
        words = len(scene.narration.split())
        scene.duration = min(MAX_SCENE_SECONDS, max(1.5, (words / wpm) * 60))
        scene.start = cursor
        cursor += scene.duration

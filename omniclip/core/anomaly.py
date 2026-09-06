"""Check the stills before any clip is generated from them.

Every defect that reached a finished video was visible in its still first: a
character drawn twice, a chalkboard of garbled script, a blank frame. Each one
cost a full clip generation and a render to discover, and each was caught in the
end by a person looking at a contact sheet. That does not scale past one user.

Stills are cheap and instant to remake; clips are neither. So the checking
belongs here, between them.

Two kinds of check. The cheap ones run locally on pixels and cost nothing. The
rest need a model to actually look, and those are batched -- fourteen images to
a request -- so examining a whole video costs one or two calls.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

# Two stills closer than this are, to the eye, the same picture. Measured: in a
# working render unrelated scenes sat above 20, a genuinely repeated one under 5.
SAME_PICTURE = 6
# Below this mean luma a frame is black rather than merely dark.
BLANK_LUMA = 8.0


@dataclass
class Finding:
    scene: int
    code: str      # duplicate_person | text | anatomy | blank | repeat
    detail: str

    def __str__(self) -> str:
        return f"scene {self.scene}: {self.detail}"


def _thumbnail(path: Path, width: int = 384) -> str:
    """A still as a small data URL, because full frames are token-heavy."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if image.width > width:
        image = image.resize((width, int(image.height * width / image.width)),
                             Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _hash(path: Path) -> int | None:
    """Difference hash of a still, for spotting repeats between scenes."""
    from PIL import Image

    try:
        image = Image.open(path).convert("L").resize((9, 8))
    except Exception:
        return None
    pixels = list(image.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if pixels[row * 9 + col] > pixels[row * 9 + col + 1] else 0)
    return bits


def _luma(path: Path) -> float | None:
    from PIL import Image

    try:
        image = Image.open(path).convert("L").resize((32, 32))
    except Exception:
        return None
    data = list(image.getdata())
    return sum(data) / len(data)


def local_findings(records) -> list[Finding]:
    """The checks that need no model: blank frames and repeated pictures."""
    findings: list[Finding] = []
    hashes: dict[int, int] = {}

    for record in records:
        path = Path(record.still_path or "")
        if not path.exists():
            continue
        luma = _luma(path)
        if luma is not None and luma < BLANK_LUMA:
            findings.append(Finding(record.index, "blank",
                                    "the frame is black or nearly black"))
            continue
        digest = _hash(path)
        if digest is not None:
            hashes[record.index] = digest

    seen = sorted(hashes.items())
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            a, b = seen[i], seen[j]
            if bin(a[1] ^ b[1]).count("1") <= SAME_PICTURE:
                findings.append(Finding(
                    b[0], "repeat",
                    f"shows the same picture as scene {a[0]}"))
    return findings


_PROMPT = """You are checking generated images for defects before they are
animated. Judge only what is visibly wrong with the picture; do not comment on
style, taste or composition.

For EACH image, in the order given, report:
  "people"   - how many distinct human or character figures are visible. Count
               a figure reflected or repeated in the frame separately. Use 0
               for a frame with nobody in it.
  "repeated" - true if the SAME individual appears more than once in the frame
               (a twin, a clone, a duplicate at the edge), false otherwise.
  "text"     - true if any readable or attempted lettering, numbers, words or
               signage appears anywhere in the frame, false otherwise.
  "anatomy"  - true if any figure is malformed: extra or missing limbs or
               fingers, fused objects, a distorted or melted face.

Be strict about "repeated" and "text": those are the defects being looked for.

Return ONLY a JSON array with one object per image, in order:
[{{"people": 0, "repeated": false, "text": false, "anatomy": false}}]
"""


def _parse_array(raw: str) -> list[dict]:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [p for p in parsed if isinstance(p, dict)]


def vision_findings(records, expected_people: dict[int, int], cfg: dict,
                    batch_size: int = 12, progress=None) -> list[Finding]:
    """Ask a model to look at each still and say what is wrong with it.

    A still whose figure count does not match what the scene asked for is the
    single most common defect, and the one a person spots instantly and code
    cannot spot at all without looking.
    """
    from openai import OpenAI

    usable = [r for r in records if r.still_path and Path(r.still_path).exists()]
    if not usable:
        return []

    client = OpenAI(base_url=cfg["script"]["base_url"],
                    api_key=cfg["script"]["api_key"], max_retries=1)
    model = cfg["shots"]["vision_model"]
    findings: list[Finding] = []

    for start in range(0, len(usable), batch_size):
        batch = usable[start:start + batch_size]
        if progress:
            progress(min(start + len(batch), len(usable)), len(usable))
        content: list[dict] = [{"type": "text", "text": _PROMPT}]
        for record in batch:
            content.append({"type": "image_url",
                            "image_url": {"url": _thumbnail(Path(record.still_path))}})
        try:
            reply = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": content}],
                max_completion_tokens=cfg["shots"].get("vision_max_tokens", 6000),
            )
            verdicts = _parse_array(reply.choices[0].message.content or "")
        except Exception:
            # A failed inspection must not fail the run. The still stands, and
            # the worst case is the behaviour we had before any checking.
            continue

        for record, verdict in zip(batch, verdicts):
            if verdict.get("repeated"):
                findings.append(Finding(record.index, "duplicate_person",
                                        "the same character appears more than once"))
            if verdict.get("text"):
                findings.append(Finding(record.index, "text",
                                        "readable lettering appears in the frame"))
            if verdict.get("anatomy"):
                findings.append(Finding(record.index, "anatomy",
                                        "a figure is malformed"))
            wanted = expected_people.get(record.index)
            seen_people = verdict.get("people")
            if (wanted is not None and isinstance(seen_people, int)
                    and wanted > 0 and seen_people > wanted):
                findings.append(Finding(
                    record.index, "duplicate_person",
                    f"{seen_people} figures in frame but the scene calls for {wanted}"))
    return findings


# What to add to a prompt to correct each defect. Kept as data so the wording
# lives next to the check that triggers it.
CORRECTIONS = {
    "duplicate_person": (
        "CRITICAL: exactly one of each named character, and no one else. No "
        "duplicate, twin, reflection or repeated copy of any figure anywhere in "
        "the frame, including at its edges and in the background"
    ),
    "text": (
        "CRITICAL: absolutely no lettering, numbers, words, signage or symbols "
        "anywhere. Any board, page or sign is blank, blurred, or turned away "
        "from the camera"
    ),
    "anatomy": (
        "CRITICAL: correct anatomy. Exactly two arms, two legs and five fingers "
        "per hand. No fused, missing or extra limbs, and no object touching or "
        "overlapping a face"
    ),
    "repeat": (
        "Make this frame visibly distinct from the other scenes: change the "
        "camera angle, the distance and what fills the background"
    ),
    "blank": "A fully lit frame with its subject clearly visible",
}


def corrected_prompt(prompt: str, findings: list[Finding]) -> str:
    """The original prompt plus a targeted correction for each defect found."""
    codes: list[str] = []
    for finding in findings:
        if finding.code not in codes:
            codes.append(finding.code)
    additions = [CORRECTIONS[c] for c in codes if c in CORRECTIONS]
    if not additions:
        return prompt
    return prompt.rstrip(". ") + ". " + ". ".join(additions) + "."


def summarise(findings: list[Finding], scenes: int, fixed: int = 0) -> str:
    if not findings:
        return f"{scenes} stills checked, nothing wrong found"
    scenes_hit = len({f.scene for f in findings})
    line = f"{scenes} stills checked, {scenes_hit} with problems"
    if fixed:
        line += f", {fixed} fixed automatically"
    return line

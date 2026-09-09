"""Cinema-grade production contact sheet generator for OmniClip."""

from __future__ import annotations

import html
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from .jobstore import Job


def build_contact_sheet(job: Job, output_filename: str = "contact_sheet.jpg") -> Path | None:
    """Generates an aesthetic cinema contact sheet of all scene stills with metadata."""
    stills = [s for s in job.scenes if s.has_still or (s.still_path and Path(s.still_path).exists())]
    if not stills:
        return None

    card_w = 480
    # Use first still to determine aspect ratio
    sample_img = Image.open(stills[0].still_path)
    aspect = sample_img.height / max(1, sample_img.width)
    card_h = int(card_w * aspect)
    meta_h = 70
    total_card_h = card_h + meta_h

    cols = min(3, len(stills))
    rows = (len(stills) + cols - 1) // cols
    margin = 24
    pad = 16

    header_h = 80
    sheet_w = cols * card_w + (cols - 1) * pad + 2 * margin
    sheet_h = header_h + rows * total_card_h + (rows - 1) * pad + 2 * margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), (13, 16, 21))
    draw = ImageDraw.Draw(sheet)

    # Title & Header
    title = (job.source.get("title") if job.source else None) or job.directory.name
    total_dur = sum(s.duration for s in job.scenes)
    header_text = f"🎬 OMNICLIP PRODUCTION CONTACT SHEET  |  {len(stills)} SCENES  |  {total_dur:.1f}s TOTAL"
    draw.text((margin, margin), header_text, fill=(216, 162, 74))
    draw.text((margin, margin + 28), title[:85], fill=(230, 235, 245))

    for idx, scene in enumerate(stills):
        r_idx = idx // cols
        c_idx = idx % cols
        x = margin + c_idx * (card_w + pad)
        y = margin + header_h + r_idx * (total_card_h + pad)

        # Draw still
        try:
            with Image.open(scene.still_path) as img:
                thumb = img.convert("RGB").resize((card_w, card_h), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x, y))
        except Exception:
            draw.rectangle([x, y, x + card_w, y + card_h], fill=(30, 35, 45))

        # Bottom metadata bar
        meta_y = y + card_h
        draw.rectangle([x, meta_y, x + card_w, meta_y + meta_h], fill=(21, 26, 34))
        draw.rectangle([x, y, x + card_w, meta_y + meta_h], outline=(40, 48, 62), width=1)

        # Scene badge & info
        purpose = (scene.purpose or "SCENE").upper()
        badge_text = f"#{scene.index + 1}  {purpose}  ·  {scene.duration:.1f}s"
        draw.text((x + 10, meta_y + 8), badge_text, fill=(216, 162, 74))

        # Visual query / prompt text
        prompt = (scene.visual_query or "").replace("\n", " ").strip()
        if len(prompt) > 85:
            prompt = prompt[:82] + "..."
        draw.text((x + 10, meta_y + 34), prompt, fill=(160, 170, 184))

    dest = job.directory / output_filename
    job.directory.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=92)
    return dest

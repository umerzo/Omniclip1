"""Pieces of interface used on more than one page.

Kept together so a job looks the same wherever it appears. The progress bar in
particular is not Streamlit's: a run is nine stages of very unequal length, and
a bar that treats them as equal spends most of a build claiming the same
percentage while clip generation grinds through sixty percent of the work.
"""

from __future__ import annotations

import html
import time
from pathlib import Path

import streamlit as st

from ..core.jobstore import STAGE_LABELS, STAGE_WEIGHTS, STAGES, Job
from . import labels, runner
from . import queue_store as q


def overall_progress(job: Job) -> tuple[float, str]:
    """How far through a run is, weighted by how long each stage really takes.

    Clip generation is most of the wait, so it is most of the bar. Within it the
    fraction of finished scenes counts, otherwise the bar would sit still for
    forty minutes.
    """
    # A stage that came before the furthest finished one counts as finished,
    # even if this job has no record of it. Jobs built before a stage existed
    # would otherwise sit at partial progress for ever, which is what a job
    # from yesterday did the moment the review stage was added.
    reached = -1
    for position, stage in enumerate(STAGES):
        if job.done(stage):
            reached = position

    done = 0.0
    current = "complete"
    for position, stage in enumerate(STAGES):
        weight = STAGE_WEIGHTS[stage]
        if job.done(stage) or position < reached:
            done += weight
            continue
        current = stage
        if stage == "visuals" and job.scenes:
            share = sum(1 for s in job.scenes if s.has_asset) / len(job.scenes)
            done += weight * share
        elif stage == "stills" and job.scenes:
            share = sum(1 for s in job.scenes if s.has_still) / len(job.scenes)
            done += weight * share
        break
    return min(done, 100.0) / 100.0, current


def stage_track(job: Job) -> None:
    """Every stage as its own segment, sized by how long it takes."""
    _, current = overall_progress(job)
    segments = []
    for stage in STAGES:
        state = "done" if job.done(stage) else ("now" if stage == current else "")
        segments.append(
            f'<div class="oc-seg {state}" style="flex:{STAGE_WEIGHTS[stage]}"></div>')
    st.markdown('<div class="oc-track">' + "".join(segments) + "</div>",
                unsafe_allow_html=True)


def stage_caption(job: Job) -> str:
    fraction, current = overall_progress(job)
    if current == "complete":
        return "Finished"
    label = STAGE_LABELS.get(current, current.title())
    detail = ""
    if job.scenes:
        if current == "visuals":
            detail = f" {sum(1 for s in job.scenes if s.has_asset)}/{len(job.scenes)}"
        elif current == "stills":
            detail = f" {sum(1 for s in job.scenes if s.has_still)}/{len(job.scenes)}"
    return f"{label}{detail} · {fraction * 100:.0f}%"


def metrics(pairs: list[tuple[str, str]]) -> None:
    blocks = "".join(
        f'<div class="oc-metric"><div class="v">{html.escape(str(v))}</div>'
        f'<div class="k">{html.escape(k)}</div></div>'
        for k, v in pairs
    )
    st.markdown(f'<div class="oc-metrics">{blocks}</div>', unsafe_allow_html=True)


def log_view(out_dir: Path, lines: int = 30) -> None:
    """The run's own output, filtered so no provider name reaches the screen."""
    text = labels.mask(runner.tail_log(out_dir, lines)) or "Waiting for output."
    safe = html.escape(text)
    # Stage headers are what a person scans for, so they are the only thing lit.
    safe = safe.replace("== ", '<span class="hl">== ')
    safe = "\n".join(
        line + "</span>" if '<span class="hl">' in line else line
        for line in safe.split("\n")
    )
    st.markdown(f'<div class="oc-term">{safe}</div>', unsafe_allow_html=True)


def profile_summary(job: Job) -> None:
    """What the engine decided this video is, and how it will sound."""
    profile = job.profile or {}
    if not profile:
        return
    kind = str(profile.get("kind", "")).replace("_", " ").title()
    audio = ("Silent" if not profile.get("wps") else
             "Voice and music" if profile.get("music") != "none" else "Voice only")
    row = [
        ("Kind", kind or "Unknown"),
        ("Look", str(profile.get("realism", "")).title()),
        ("Audio", audio),
        ("Captions", "On" if profile.get("captions") else "Off"),
    ]
    st.markdown(
        " ".join(f'<span class="oc-pill">{html.escape(k)}: {html.escape(str(v))}</span>'
                 for k, v in row),
        unsafe_allow_html=True)
    if profile.get("reason"):
        st.caption(labels.mask(str(profile["reason"])))


def estimate_note(job: Job) -> None:
    """Say what a run will cost before it spends it."""
    shots = job.stages.get("shots") or {}
    if not shots.get("est_total_minutes"):
        return
    minutes = shots["est_total_minutes"]
    length = f"{minutes / 60:.1f} hours" if minutes >= 90 else f"{minutes:.0f} minutes"
    text = f"{shots.get('est_scenes', '?')} scenes, about {length} of generation."
    if shots.get("est_over_budget"):
        text += (f" That is more than the {shots.get('est_scene_cap')} scene budget. "
                 f"Trim the source or raise the cap to spend less.")
        st.markdown(f'<div class="oc-note warn">{html.escape(text)}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="oc-note">{html.escape(text)}</div>',
                    unsafe_allow_html=True)


def quality_summary(job: Job) -> None:
    """What the automatic check found, in one line."""
    review = job.stages.get("review") or {}
    if not review.get("done"):
        return
    checked = review.get("checked", 0)
    fixed = review.get("fixed", 0)
    flagged = len(review.get("problems") or [])
    text = f"{checked} stills checked"
    if fixed:
        text += f", {fixed} fixed automatically"
    text += f", {flagged} flagged" if flagged else ", none flagged"
    kind = "warn" if flagged else ""
    st.markdown(f'<div class="oc-note {kind}">{html.escape(text)}</div>',
                unsafe_allow_html=True)


# A run writes to its log at every step. If nothing has been written for longer
# than this, it is not working, whatever the progress bar suggests. Generation
# is slow, so the threshold is generous: a single clip can take two minutes.
STALL_MINUTES = 6.0


def minutes_since_output(out_dir) -> float | None:
    """How long since this run last wrote anything, or None if it never did."""
    log = Path(out_dir) / runner.LOG_FILE
    if not log.exists():
        return None
    return (time.time() - log.stat().st_mtime) / 60.0


def looks_stalled(out_dir) -> bool:
    """Alive, but producing nothing. The case a progress bar cannot show."""
    if not runner.is_running(out_dir):
        return False
    idle = minutes_since_output(out_dir)
    return idle is not None and idle > STALL_MINUTES


def state_explain(entry: dict, out_dir=None) -> tuple[str, str]:
    """One honest sentence about this job, and how serious it is.

    Everything a person needs is what is happening and what happens next. The
    wording lives with the queue so every screen agrees.
    """
    state = entry.get("state", q.WAITING)
    text = q.STATE_EXPLAINS.get(state, "")

    if state == q.HOLDING:
        due = entry.get("retry_at") or 0
        wait = max(0, due - time.time())
        when = f"{wait / 60:.0f} minutes" if wait > 60 else "shortly"
        return f"{text} Next attempt in {when}.", "warn"

    if state == q.RUNNING and out_dir is not None and looks_stalled(out_dir):
        idle = minutes_since_output(out_dir) or 0
        return (f"Running, but nothing has come back for {idle:.0f} minutes. "
                f"It is most likely waiting on generation capacity.", "warn")

    kind = {q.FAILED: "bad", q.REVIEW: "warn", q.DONE: ""}.get(state, "")
    return text, kind

def relative_time(stamp: float | None) -> str:
    if not stamp:
        return ""
    delta = max(0, time.time() - float(stamp))
    if delta < 90:
        return f"{delta:.0f}s ago"
    if delta < 5400:
        return f"{delta / 60:.0f}m ago"
    if delta < 172800:
        return f"{delta / 3600:.0f}h ago"
    return f"{delta / 86400:.0f}d ago"


def state_pill(state: str) -> str:
    kind = {q.RUNNING: "run", q.DONE: "ok", q.FAILED: "bad",
            q.REVIEW: "warn", q.HOLDING: "warn", q.CANCELLED: ""}.get(state, "")
    return (f'<span class="oc-pill {kind}">'
            f'{html.escape(q.STATE_LABEL.get(state, state))}</span>')


# ----------------------------------------------------------- Studio Components

def render_contact_sheet(job: Job) -> None:
    """Renders an aesthetic cinema contact sheet of all scene stills."""
    stills = [s for s in job.scenes if s.has_still or s.still_path or s.still_url]
    if not stills:
        return

    st.markdown("### 🎬 Production Contact Sheet")
    st.caption("Visual anchor stills established for this production. Clips are animated directly from these frames.")

    cols = st.columns(min(4, len(stills)))
    for idx, scene in enumerate(stills):
        col = cols[idx % len(cols)]
        with col:
            img_src = scene.still_path if (scene.still_path and Path(scene.still_path).exists()) else scene.still_url
            if img_src:
                try:
                    st.image(img_src, use_container_width=True)
                except Exception:
                    st.caption("[Image unavailable]")
            purpose = (scene.purpose or "Scene").upper()
            st.markdown(
                f'<div style="background:#151A22; padding:6px 10px; border-radius:6px; margin-top:-10px; border:1px solid rgba(255,255,255,0.08);">'
                f'<span style="color:#D8A24A; font-weight:700; font-size:0.75rem;">#{scene.index + 1} {html.escape(purpose)}</span> '
                f'<span style="color:#8E99A8; font-size:0.72rem;">· {scene.duration:.1f}s</span>'
                f'<div style="font-size:0.7rem; color:#A0AAB8; margin-top:3px; max-height:36px; overflow:hidden; text-overflow:ellipsis;">'
                f'{html.escape(scene.visual_query[:90])}...</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Offer downloadable consolidated contact sheet
    try:
        from PIL import Image, ImageDraw
        sheet_path = job.directory / "contact_sheet.jpg"
        if not sheet_path.exists() and any(s.still_path and Path(s.still_path).exists() for s in stills):
            valid_imgs = [Image.open(s.still_path).convert("RGB") for s in stills if s.still_path and Path(s.still_path).exists()]
            if valid_imgs:
                w, h = 400, int(400 * (valid_imgs[0].height / valid_imgs[0].width))
                thumb_imgs = [img.resize((w, h)) for img in valid_imgs]
                cols_count = min(3, len(thumb_imgs))
                rows_count = (len(thumb_imgs) + cols_count - 1) // cols_count
                sheet = Image.new("RGB", (cols_count * w, rows_count * h), (14, 17, 20))
                for i, t in enumerate(thumb_imgs):
                    r_idx = i // cols_count
                    c_idx = i % cols_count
                    sheet.paste(t, (c_idx * w, r_idx * h))
                sheet.save(sheet_path, quality=90)

        if sheet_path.exists():
            with open(sheet_path, "rb") as f:
                st.download_button(
                    "📥 Download Composite Contact Sheet (JPG)",
                    f.read(),
                    file_name=f"{job.id}_contact_sheet.jpg",
                    mime="image/jpeg",
                )
    except Exception:
        pass


def render_pipeline_stepper(job: Job) -> None:
    """Visual horizontal pipeline stepper with glowing nodes and status indicators."""
    stages = [
        ("ingest", "Ingest"),
        ("shots", "DNA & Shots"),
        ("storyboard", "Script Deck"),
        ("stills", "Anchor Stills"),
        ("visuals", "Video Clips"),
        ("qa", "VideoQA"),
        ("render", "Compositor"),
    ]
    _, current_stage = overall_progress(job)

    nodes_html = []
    for code, label in stages:
        done = job.done(code)
        active = (code == current_stage or (code == "qa" and current_stage == "visuals" and all(s.has_asset for s in job.scenes)))
        status_cls = "done" if done else ("active" if active else "pending")
        icon = "✓" if done else ("⚡" if active else "•")
        nodes_html.append(
            f'<div class="oc-step-node {status_cls}">'
            f'<div class="oc-step-circle">{icon}</div>'
            f'<div class="oc-step-label">{html.escape(label)}</div>'
            f'</div>'
        )

    st.html('<div class="oc-stepper">' + "".join(nodes_html) + "</div>")


def render_key_pool_telemetry() -> None:
    """Telemetry badge displaying Agnes 25-key pool status and engine health."""
    from omniclip.pipeline import agnes_keys
    keys = agnes_keys()
    count = len(keys)
    st.html(
        f'<div class="oc-telemetry">'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
        f'<div><span style="color:#63B98D; font-size:0.8rem; font-weight:700;">🟢 MULTI-KEY POOL READY</span>'
        f'<div style="color:#98A1AE; font-size:0.75rem; margin-top:2px;">{count} Agnes AI Keys Active · LRU Cooldown Protection</div></div>'
        f'<div style="text-align:right;"><span style="color:#D8A24A; font-weight:700; font-size:0.9rem;">{count} Workers</span>'
        f'<div style="color:#6B7482; font-size:0.7rem;">Stock Fallback: Pexels + Pixabay</div></div>'
        f'</div>'
        f'</div>'
    )


def render_qa_inspector(job: Job) -> None:
    """Inspector for VideoQA scores and individual scene regeneration triggers."""
    qa = job.qa_report or {}
    if not qa:
        return

    score = qa.get("overall_score", 0)
    passed = qa.get("passed", True)
    summary = qa.get("summary", "")
    badge_cls = "ok" if passed and score >= 75 else "warn"

    st.html(
        f'<div style="display:flex; align-items:center; gap:1rem; margin:0.8rem 0;">'
        f'<div class="oc-qa-badge {badge_cls}">'
        f'<span>QA SCORE:</span> <span>{score} / 100</span>'
        f'</div>'
        f'<div style="font-size:0.85rem; color:#98A1AE;">{html.escape(summary)}</div>'
        f'</div>'
    )


def render_sidebar_header() -> None:
    """Renders a visual, futuristic branding header in the sidebar."""
    st.html(
        """<div class="oc-sidebar-brand">
  <div class="oc-brand-badge">
    <div class="oc-brand-icon">🎬</div>
    <div>
      <div class="oc-brand-title">OMNICLIP <span class="oc-brand-ver">v2.4</span></div>
      <div class="oc-brand-subtitle">AI CONTENT STUDIO</div>
    </div>
  </div>
  <div class="oc-brand-status">
    <span class="oc-led-pulse"></span>
    <span class="oc-status-text">NEURAL ENGINE ONLINE</span>
  </div>
</div>"""
    )


def render_sidebar_service_lights(running_builds: int = 0) -> None:
    """Renders simple, clean glowing service lights for key system resources and components."""
    from omniclip.pipeline import agnes_keys
    slots = len(agnes_keys())

    st.html(
        f"""<div class="oc-service-card">
  <div class="oc-service-header">
    <span>SERVICE LIGHTS</span>
    <span style="color:#22C55E; font-size:0.68rem; font-weight:700;">● ALL SYSTEMS GO</span>
  </div>
  <div class="oc-service-row">
    <div class="oc-service-left">
      <span class="oc-service-dot"></span>
      <span>APIs & Brain</span>
    </div>
    <span class="oc-service-status">Connected</span>
  </div>
  <div class="oc-service-row">
    <div class="oc-service-left">
      <span class="oc-service-dot"></span>
      <span>Video Model</span>
    </div>
    <span class="oc-service-status">{slots} Keys Active</span>
  </div>
  <div class="oc-service-row">
    <div class="oc-service-left">
      <span class="oc-service-dot"></span>
      <span>Audio Model</span>
    </div>
    <span class="oc-service-status">Connected</span>
  </div>
  <div class="oc-service-row">
    <div class="oc-service-left">
      <span class="oc-service-dot"></span>
      <span>Music Generator</span>
    </div>
    <span class="oc-service-status">Connected</span>
  </div>
    <div class="oc-service-row">
    <div class="oc-service-left">
      <span class="oc-service-dot"></span>
      <span>Video Renderer</span>
    </div>
    <span class="oc-service-status">Ready</span>
  </div>
</div>"""
    )


def render_publishing_kit(job: Job) -> None:
    """Renders the interactive YouTube Publishing Kit (Titles, Chapters, SEO Description)."""
    from ..core.publisher import generate_publishing_kit

    st.markdown("### 📦 YouTube Publishing Kit & Retention Suite")
    st.caption("AI-generated high-CTR title variations, accurate timestamped chapters, and SEO description ready for YouTube Studio.")

    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn2:
        regen = st.button("🔄 Regenerate Kit", key=f"regen_kit_{job.id}", use_container_width=True, help="Force re-run AI title and chapter synthesis")

    cached_kit = job.stages.get("publish")
    if regen or not cached_kit:
        with st.spinner("✨ Synthesizing high-CTR titles, accurate chapters, and SEO metadata with Gemini..."):
            kit = generate_publishing_kit(job, use_ai=True, force=bool(regen))
    else:
        kit = cached_kit

    if not kit or not kit.get("titles"):
        st.warning("Publishing kit could not be generated yet. Ensure storyboard or scenes are present.")
        return

    st.markdown("#### 🎯 High-CTR YouTube Title Variations")
    st.caption("Crafted to maximize click-through rate across different audience psychology triggers:")

    tcols = st.columns(3)
    icons = {"Curiosity Hook": "🎣", "Direct Value & Search": "🎯", "Contrarian / Intrigue": "🔥"}
    for idx, t in enumerate(kit.get("titles", [])):
        with tcols[idx % 3]:
            style_name = t.get("style", f"Option {idx+1}")
            icon = icons.get(style_name, "✨")
            char_count = t.get("char_count", len(t.get("title", "")))
            char_badge = f"{char_count} chars" if char_count <= 70 else f"{char_count} chars (long)"
            st.markdown(
                f"""<div class="oc-card" style="padding:14px; min-height:160px; border-left:3px solid var(--accent);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-weight:700; font-size:0.85rem; color:var(--accent);">{icon} {html.escape(style_name.upper())}</span>
                        <span class="oc-pill ok" style="font-size:0.7rem;">{char_badge}</span>
                    </div>
                    <div style="font-weight:600; font-size:0.95rem; line-height:1.35; margin-bottom:8px;">{html.escape(t.get('title', ''))}</div>
                    <div style="font-size:0.78rem; color:var(--muted); line-height:1.3;">{html.escape(t.get('ctr_explanation', ''))}</div>
                </div>""",
                unsafe_allow_html=True
            )
            st.code(t.get("title", ""), language="")

    st.markdown("---")

    ch_col, desc_col = st.columns([1, 1])

    with ch_col:
        st.markdown("#### ⏱️ Timestamped YouTube Chapters")
        st.caption("YouTube-ready chapter timestamps calculated from narrative scene timings (starts at 00:00).")

        chapter_text = kit.get("chapter_text", "")
        if not chapter_text and kit.get("chapters"):
            chapter_text = "\n".join(f"{c['timestamp']} {c['title']}" for c in kit["chapters"])

        st.code(chapter_text, language="text")

        with st.expander("🔍 Chapter Breakdown Timeline", expanded=False):
            for c in kit.get("chapters", []):
                st.markdown(f"**`{c['timestamp']}`** — {html.escape(c['title'])}")

    with desc_col:
        st.markdown("#### 📝 SEO Description Template")
        st.caption("Pre-formatted video description with hook, chapters, and hashtags.")
        st.code(kit.get("description", ""), language="text")

    st.markdown("#### 🏷️ Video Tags (YouTube Studio)")
    tags_list = kit.get("tags", [])
    if tags_list:
        tags_str = ", ".join(tags_list)
        st.code(tags_str, language="text")
        st.caption("Copy and paste directly into the 'Tags' input box in YouTube Studio.")

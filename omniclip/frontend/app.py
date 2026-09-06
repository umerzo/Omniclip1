"""OmniClip: paste a link, get an original video.

Five pages rather than one long form. A build has genuinely separate concerns
(what to make, what is being made, what needs a person, what was made, and how
the tool is set up) and stacking them into a single scrolling page is what made
the previous version hard to use as soon as more than one job existed.

Nothing here runs a build. The page starts a supervisor process and then only
reads the files it writes, so closing the browser leaves the work untouched.
"""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omniclip.core.jobstore import Job  # noqa: E402
from omniclip.core.profile import KINDS  # noqa: E402
from omniclip.frontend import components as ui  # noqa: E402
from omniclip.frontend import labels, reinforce, runner, theme, worker  # noqa: E402
from omniclip.frontend import queue_store as q  # noqa: E402
from omniclip.pipeline import agnes_keys  # noqa: E402
from omniclip.utils.config import load_settings, resolve_path  # noqa: E402

st.set_page_config(page_title="OmniClip AI Studio", page_icon="🎬", layout="wide")
theme.apply()

REFRESH_SECONDS = 6


@st.cache_data(ttl=600, show_spinner=False)
def voices_for_language(code: str) -> list[str]:
    """Voice names for a language. Cached because the list is a network call."""
    from omniclip.core.language import voices_for

    try:
        return voices_for(code) or []
    except Exception:
        return []


def slug_for(url: str, taken: set[str]) -> str:
    """A folder name from the link, kept unique so two builds never collide."""
    base = url.rstrip("/").split("/")[-1].split("?")[0].split("&")[0]
    base = "".join(c for c in base if c.isalnum() or c in "-_") or "video"
    name, suffix = base, 2
    while name in taken:
        name, suffix = f"{base}-{suffix}", suffix + 1
    return name


# ---------------------------------------------------------------- new build
def page_new() -> None:
    theme.head(
        "New build",
        "Paste a YouTube link or topic. Gemini 3.8 Flash extracts its Content DNA, "
        "determines the exact video format, and creates an original, high-retention video.")

    cfg = load_settings()

    mcol1, mcol2 = st.columns([1, 1])
    with mcol1:
        workflow_mode = st.radio(
            "Workflow Mode",
            ["🔄 AI Remake / Recreate (Preserve Cast & Action)", "💡 Original Concept Spin-Off (DNA Only)"],
            index=0,
            key="new_workflow_mode",
            help="AI Remake preserves the original characters (e.g. boy on bicycle + goat) and scene sequence. Spin-Off crafts a completely new story from the extracted DNA."
        )
    with mcol2:
        input_mode = st.radio("Source Input", ["YouTube Video Link", "Original Topic / Concept"], horizontal=True, key="new_input_mode")

    url = ""
    topic = ""
    if input_mode == "YouTube Video Link":
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=... or https://www.youtube.com/shorts/...",
            key="new_url_input",
            help="Video can be a Short or Long-form video across any genre.")
    else:
        topic = st.text_area(
            "Topic or Creative Concept",
            placeholder="e.g. 2025 Porsche 911 GT3 RS mountain pass showcase, or How Quantum Computers actually calculate...",
            key="new_topic_input",
            help="Describe the subject, goal, and tone for the video.")

    col_plat, col_inst = st.columns([1, 2])
    with col_plat:
        target_platform = st.selectbox(
            "Target Platform",
            ["YouTube Shorts (9:16)", "YouTube Long-form (16:9)", "TikTok (9:16)", "Instagram Reels (9:16)"],
            index=0 if (url and "shorts" in url.lower()) else 1,
            key="new_target_platform",
        )
    with col_inst:
        user_instructions = st.text_input(
            "Creative Guidance (Optional)",
            placeholder="e.g. Focus on high-speed kinetic cuts, no people, dramatic lighting",
            key="new_user_instructions",
            help="Directives for the script, visual pacing, or tone.")

    # Content DNA & Concept Inspection Button
    active_source = url.strip() or topic.strip()
    if active_source:
        if st.button("🔍 Analyze Content DNA & Format", key="new_btn_analyze", use_container_width=True):
            with st.spinner("Analyzing content format and extracting Content DNA via Gemini 3.8 Flash..."):
                try:
                    from omniclip.core.analyzer import SourceAnalyzer
                    analyzer = SourceAnalyzer()
                    report = analyzer.analyze(
                        url=url.strip() if url.strip() else f"concept://{slug_for(topic[:30], set())}",
                        user_target=target_platform,
                    )
                    st.session_state["active_analysis"] = report.to_dict()
                except Exception as err:
                    st.warning(f"Analysis fallback used: {err}")

    # Render Content DNA card if present in session state
    if "active_analysis" in st.session_state:
        analysis = st.session_state["active_analysis"]
        classification = analysis.get("classification") or {}
        dna = analysis.get("content_dna") or {}
        primary = classification.get("primary_type", "General").replace("_", " ").title()
        confidence = float(classification.get("confidence", 0.9)) * 100

        st.markdown("---")
        st.markdown(f"### Content Architecture: **{primary}** ({classification.get('format', 'short').upper()})")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Format & Genre", f"{primary}", f"{confidence:.0f}% confidence")
        with mcol2:
            st.metric("Visual Dependency", classification.get("visual_dependency", "high").title())
        with mcol3:
            st.metric("Strategy", classification.get("recommended_generation_strategy", "Showcase").replace("_", " ").title())

        with st.expander("🧬 View Content DNA Blueprint", expanded=True):
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.markdown(f"**Hook Mechanism:** {dna.get('hook', {}).get('mechanism', 'Pattern interrupt')}")
                st.markdown(f"**Core Promise:** {dna.get('promise', {}).get('core_value', 'Visual engagement')}")
                st.markdown(f"**Pacing Rhythm:** {dna.get('pacing', {}).get('rhythm', 'Dynamic')}")
            with dcol2:
                st.markdown(f"**Visual Palette:** {dna.get('visual_language', {}).get('palette', 'Cinematic')}")
                avoid_list = ", ".join(dna.get("things_to_avoid") or ["AI slop", "static shots"])
                st.markdown(f"**Anti-Patterns to Avoid:** {avoid_list}")

    st.markdown("#### Sound")
    col1, col2, col3 = st.columns(3)
    with col1:
        audio = st.selectbox(
            "Soundtrack", ["auto", "voiceover", "music", "silent"],
            format_func=lambda v: {
                "auto": "Match the source / DNA",
                "voiceover": "Voice only, no music",
                "music": "Music only, no voice",
                "silent": "No sound at all",
            }[v],
            key="new_audio_select",
            help="Match the source reads the video DNA and decides. A bulletin gets "
                 "no music, a meditation is mostly music.")
    with col2:
        language = st.text_input("Spoken language", value="",
                                 placeholder="Same as the source",
                                 key="new_language_input",
                                 help="A language code such as en, ur or es.")
    with col3:
        caption_language = st.text_input("Caption language", value="en", key="new_caption_language_input")

    voice_options = voices_for_language(language.strip() or "en")
    vcol1, vcol2 = st.columns([3, 1])
    with vcol1:
        voice = st.selectbox("Voice", ["Choose automatically"] + voice_options,
                             key="new_voice_select",
                             help="Automatic picks a voice that suits the language.")
    with vcol2:
        st.write("")
        if st.button("Hear it", key="new_btn_hear_voice", disabled=voice == "Choose automatically",
                     use_container_width=True):
            _preview_voice(voice)

    captions_on = st.toggle(
        "Burn in captions", value=True,
        key="new_captions_toggle",
        help="Captions use the caption language whatever the voice speaks.")

    caption_style = "kinetic"
    if captions_on:
        cap_col1, cap_col2 = st.columns([1, 1])
        with cap_col1:
            caption_style_choice = st.radio(
                "Subtitle Animation Style",
                ["⚡ Kinetic Pop-in Subtitles", "📄 Standard Subtitles"],
                index=0,
                horizontal=True,
                key="new_caption_style_choice",
                help="Kinetic Pop-in: Dynamic, high-engagement word-by-word bounce animation (viral TikTok/Reels style). Standard: Clean, classic bottom subtitles."
            )
            caption_style = "kinetic" if "Kinetic" in caption_style_choice else "standard"
        with cap_col2:
            if caption_style == "kinetic":
                st.caption("⚡ **Kinetic Active**: High-impact word pop-in scaling, dynamic safe-zone vertical placement, and punchy 3-word cadence.")
            else:
                st.caption("📄 **Standard Active**: Classic bottom-aligned subtitle bar with smooth multi-word phrasing.")

    st.markdown("#### Picture")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        aspect_default = "9:16" if "9:16" in target_platform else "16:9"
        aspect = st.selectbox("Frame", ["9:16", "16:9", "1:1"], index=0 if aspect_default == "9:16" else 1, key="new_aspect_select")
    with pcol2:
        quality = st.selectbox("Quality", list(labels.MODEL_VALUE), index=0, key="new_quality_select")
    with pcol3:
        kind = st.selectbox(
            "Kind of video", ["Detect automatically"] + list(KINDS),
            format_func=lambda v: v.replace("_", " ").title()
            if v != "Detect automatically" else v,
            key="new_kind_select",
            help="The kind sets pace, voice, music, captions and whether the "
                 "result is photographic or illustrated.")

    style_preset = st.selectbox(
        "Visual Style Preset",
        [
            "Source Visual Match (Keep Original Aesthetic)",
            "🎨 Pixar / 3D Stylized Animation",
            "🎬 Moody Anamorphic Cinema (35mm Film)",
            "🏎️ Automotive / Commercial (Hyper-Kinetic)",
            "📼 90s VHS Retro Analog Aesthetic",
            "🌿 Hyper-Realistic Documentary (8K Wildlife / Nature)",
            "🌆 Cyberpunk Neon Noir (Rainy Tokyo Glow)",
        ],
        index=0,
        key="new_style_preset_select",
        help="Applies cinematic color grading, art direction, and lighting aesthetics across all scene stills and clips."
    )

    allow_stock = st.toggle(
        "Allow stock footage as a fallback",
        value=bool(cfg["visuals"].get("allow_stock", True)),
        key="new_allow_stock_toggle",
        help="Off means every shot is AI generated via Agnes. On lets a "
             "scene fall back to Pexels/Pixabay if Agnes capacity is spent.")

    with st.expander("Advanced"):
        acol1, acol2, acol3 = st.columns(3)
        with acol1:
            scene_cap = st.number_input(
                "Scene budget", 4, 200, int(cfg["shots"].get("scene_cap", 40)),
                key="new_scene_cap_input",
                help="One scene is roughly a minute of generation.")
            start_at = st.number_input("Skip opening seconds", 0.0, 600.0, 0.0,
                                       step=1.0,
                                       key="new_start_at_input")
        with acol2:
            trim = st.number_input("Use only the first N seconds", 0.0, 7200.0,
                                   0.0, step=10.0,
                                   key="new_trim_input")
            max_scenes = st.number_input("Stop after N scenes (testing)", 0, 200, 0, key="new_max_scenes_input")
        with acol3:
            review = st.toggle(
                "Check stills before generating clips", value=True,
                key="new_review_toggle",
                help="Finds duplicated figures, stray lettering and blank frames.")
            music = st.toggle("Allow music", value=True, key="new_music_toggle")
            pause = st.selectbox("Pause for me after",
                                 ["never", "storyboard", "stills"],
                                 key="new_pause_select")

    st.divider()
    active_item = url.strip() or topic.strip()
    if active_item:
        st.caption("1 build ready to queue. Production pipeline will follow Content DNA.")

    if st.button("Add to queue & Build", type="primary", disabled=not active_item,
                 key="btn_add_to_queue",
                 use_container_width=True):
        taken = {e["name"] for e in q.entries()}
        root = runner.output_root()
        if root.exists():
            taken |= {p.name for p in root.iterdir() if p.is_dir()}

        stored_analysis = st.session_state.get("active_analysis") or {}
        is_remake = workflow_mode.startswith("🔄")
        style_val = None if style_preset.startswith("Source") else style_preset

        options = {
            "aspect": aspect,
            "no_stock": not allow_stock,
            "audio": audio,
            "no_captions": not captions_on,
            "caption_style": caption_style,
            "voice": None if voice == "Choose automatically" else voice,
            "kind": None if kind == "Detect automatically" else kind,
            "language": language.strip() or None,
            "caption_language": caption_language.strip() or "en",
            "no_music": not music,
            "no_review": not review,
            "scene_cap": int(scene_cap),
            "video_model": labels.MODEL_VALUE[quality],
            "start": float(start_at) or None,
            "trim": float(trim) or None,
            "max_scenes": int(max_scenes) or None,
            "pause_after": None if pause == "never" else pause,
            "target_platform": target_platform,
            "instructions": user_instructions.strip() or None,
            "idea": None if is_remake else (topic.strip() or None),
            "style_phrase": style_val,
            "classification": stored_analysis.get("classification"),
            "content_dna": stored_analysis.get("content_dna"),
        }
        build_link = url.strip() if url.strip() else f"https://www.youtube.com/watch?v=custom_{slug_for(topic[:30], taken)}"
        name = slug_for(build_link, taken)
        q.add(build_link, name, options)
        worker.ensure_running()
        st.success(f"Queued build '{name}'.")
        st.session_state["selected_page"] = "Queue"
        st.rerun()


def _preview_voice(voice: str) -> None:
    """Speak one line, so a voice can be judged before a whole video uses it."""
    from omniclip.core.tts_engine import synthesize

    folder = resolve_path("assets/cache") / "voice_preview"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{voice}.mp3"
    try:
        if not path.exists():
            synthesize("This is how the narration will sound in your video.",
                       path, voice=voice)
        st.audio(str(path))
    except Exception as error:
        st.warning(f"Could not play a sample. {labels.clean_error(str(error))}")


# -------------------------------------------------------------------- queue
def page_queue() -> None:
    theme.head(
        "Queue & Live Studio Monitor",
        "Builds run one at a time. Generation is limited to about one clip a "
        "minute for the whole pool, so two at once would finish no sooner.")

    ui.render_key_pool_telemetry()

    items = q.entries()
    tally = q.counts()
    ui.metrics([
        ("Waiting", tally[q.WAITING]),
        ("Building", tally[q.RUNNING]),
        ("Held", tally[q.HOLDING]),
        ("Needs review", tally[q.REVIEW]),
        ("Done", tally[q.DONE]),
        ("Failed", tally[q.FAILED]),
    ])

    # When the service is out of capacity, say so once at the top rather than
    # making somebody infer it from a row that is not moving.
    if tally[q.HOLDING]:
        due = q.soonest_retry()
        wait = max(0, (due or 0) - time.time())
        when = f"in about {wait / 60:.0f} minutes" if wait > 60 else "shortly"
        theme.note(
            f"{tally[q.HOLDING]} build(s) are waiting for generation capacity. "
            f"Everything already built is saved, and the next attempt happens "
            f"{when} without you doing anything.", "warn")

    running_now = q.entries(q.RUNNING)
    if tally[q.WAITING] or running_now:
        remaining = _queue_remaining(running_now, tally[q.WAITING])
        if remaining:
            theme.note(f"About {remaining} of generation left in the queue.")

    ccol1, ccol2, ccol3, _ = st.columns([1.2, 1.2, 1.8, 1.8])
    with ccol1:
        if worker.is_running():
            if st.button("Pause queue", key="btn_pause_queue", use_container_width=True):
                worker.stop()
                st.rerun()
        elif st.button("Start queue", key="btn_start_queue", type="primary", use_container_width=True,
                       disabled=not tally[q.WAITING]):
            worker.ensure_running()
            time.sleep(1)
            st.rerun()
    with ccol2:
        if st.button("Clear finished", key="btn_clear_finished", use_container_width=True):
            q.clear_finished()
            st.rerun()
    with ccol3:
        if st.button("⚡ Reinforce pipeline", key="btn_reinforce_queue", use_container_width=True):
            res = reinforce.reinforce_pipeline()
            st.session_state["reinforce_result"] = res
            st.rerun()

    if "reinforce_result" in st.session_state:
        res = st.session_state["reinforce_result"]
        msg = res.get("message", "")
        tone = res.get("tone", "info")
        if tone == "success":
            st.success(msg, icon="🟢")
        elif tone == "warn":
            st.warning(msg, icon="⚡")
        elif tone == "error":
            st.error(msg, icon="⚠️")
        else:
            st.info(msg, icon="ℹ️")

    st.divider()
    if not items:
        theme.empty("Nothing queued yet. Add a link on the New build page.")
        return

    for entry in items:
        _queue_row(entry)

    if running_now:
        _live_panel(running_now[0])
        col_ref, col_reinf, _ = st.columns([1.2, 1.6, 2.2])
        with col_ref:
            if st.button("🔄 Refresh progress", key="btn_refresh_live", use_container_width=True):
                st.rerun()
        with col_reinf:
            if st.button("⚡ Reinforce pipeline", key="btn_reinforce_live", use_container_width=True):
                res = reinforce.reinforce_pipeline()
                st.session_state["reinforce_result"] = res
                st.rerun()


def _queue_remaining(running: list[dict], waiting: int) -> str:
    """A rough time to empty, from the estimate each job recorded for itself."""
    total = 0.0
    for entry in running:
        job = Job.load(runner.output_root() / entry["name"])
        if job:
            shots = job.stages.get("shots") or {}
            fraction, _ = ui.overall_progress(job)
            total += float(shots.get("est_total_minutes", 25)) * (1 - fraction)
    total += waiting * 25.0
    if total <= 0:
        return ""
    return f"{total / 60:.1f} hours" if total >= 90 else f"{total:.0f} minutes"


def _queue_row(entry: dict) -> None:
    out_dir = runner.output_root() / entry["name"]
    job = Job.load(out_dir)
    state = entry.get("state", q.WAITING)

    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            title = (job.source.get("title") if job and job.source else "") or entry["name"]
            st.markdown(
                f'<div class="oc-row"><div class="oc-grow">'
                f'<div class="oc-title">{html.escape(labels.mask(title)[:78])}</div>'
                f'<div class="oc-sub">{html.escape(entry["url"][:88])}</div></div>'
                f'{ui.state_pill(state)}</div>',
                unsafe_allow_html=True)
            if job and state in (q.RUNNING, q.REVIEW, q.DONE, q.HOLDING):
                ui.render_pipeline_stepper(job)
                st.caption(ui.stage_caption(job))
            # Never a bare state word: say what it means and what comes next.
            explain, tone = ui.state_explain(entry, out_dir)
            if explain:
                theme.note(explain, tone)
            if entry.get("message"):
                st.caption(labels.clean_error(entry["message"]))
        with right:
            if state == q.WAITING:
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button("Up", key=f"up{entry['id']}",
                                 use_container_width=True):
                        q.move(entry["id"], -1)
                        st.rerun()
                with bcol2:
                    if st.button("Down", key=f"dn{entry['id']}",
                                 use_container_width=True):
                        q.move(entry["id"], 1)
                        st.rerun()
                if st.button("Remove", key=f"rm{entry['id']}",
                             use_container_width=True):
                    q.remove(entry["id"])
                    st.rerun()
            elif state == q.RUNNING:
                if st.button("Cancel", key=f"cx{entry['id']}",
                             use_container_width=True):
                    q.update(entry["id"], state=q.CANCELLED)
                    st.rerun()
            elif state == q.HOLDING:
                if st.button("Try now", key=f"nw{entry['id']}",
                             use_container_width=True):
                    q.update(entry["id"], state=q.WAITING, message="")
                    worker.ensure_running()
                    st.rerun()
            elif state in (q.FAILED, q.CANCELLED):
                if st.button("Try again", key=f"rt{entry['id']}", type="primary",
                             use_container_width=True):
                    q.update(entry["id"], state=q.WAITING, message="", finished=None)
                    worker.ensure_running()
                    st.rerun()


def _live_panel(entry: dict) -> None:
    out_dir = runner.output_root() / entry["name"]
    st.markdown("#### Building now")
    job = Job.load(out_dir)
    if job:
        ui.render_pipeline_stepper(job)
        ui.profile_summary(job)
        ui.estimate_note(job)
        stills = [s for s in job.scenes if s.has_still or s.still_path or s.still_url]
        if stills or job.done("stills"):
            ui.render_contact_sheet(job)
    ui.log_view(out_dir, 26)


# ------------------------------------------------------------------- review
def page_review() -> None:
    theme.head(
        "Review",
        "Stills the automatic check could not fix on its own. A build waits "
        "here rather than spending an hour of generation on a frame that is "
        "already wrong, and the queue carries on without it.")

    parked = q.entries(q.REVIEW)
    if not parked:
        theme.empty("Nothing is waiting. Every build passed its own check.")
        return

    for entry in parked:
        out_dir = runner.output_root() / entry["name"]
        job = Job.load(out_dir)
        if job is None:
            continue
        review = job.stages.get("review") or {}
        problems = {int(p["scene"]): p["problems"]
                    for p in review.get("problems") or []}

        st.markdown(f"#### {labels.mask((job.source or {}).get('title') or entry['name'])}")
        ui.quality_summary(job)

        for index, reasons in sorted(problems.items()):
            record = job.scene(index)
            if record is None:
                continue
            icol, dcol = st.columns([1, 2])
            with icol:
                if record.still_path and Path(record.still_path).exists():
                    st.image(record.still_path, use_container_width=True)
            with dcol:
                st.markdown(f"**Scene {index}** at {record.duration:.1f}s")
                for reason in reasons:
                    st.markdown(
                        f'<div class="oc-note warn">{html.escape(str(reason))}</div>',
                        unsafe_allow_html=True)
                st.caption(labels.mask(record.visual_query[:240]))
                if st.button("Make this still again",
                             key=f"rg{entry['id']}_{index}",
                             use_container_width=True):
                    record.still_path = ""
                    record.still_url = ""
                    record.attempt += 1
                    job.invalidate("review")
                    job.save()
                    q.update(entry["id"], state=q.WAITING, message="")
                    worker.ensure_running()
                    st.rerun()

        acol1, acol2 = st.columns(2)
        with acol1:
            if st.button("Accept and continue", key=f"ac{entry['id']}",
                         type="primary", use_container_width=True):
                # The stills stand as they are, and the check is skipped on the
                # way back so the same frames cannot park the job again.
                job.mark("review", checked=review.get("checked", 0),
                         fixed=review.get("fixed", 0), needs_review=False,
                         problems=[], summary="accepted by hand")
                options = dict(entry.get("options") or {})
                options["no_review"] = True
                q.update(entry["id"], state=q.WAITING, message="",
                         options=options)
                worker.ensure_running()
                st.rerun()
        with acol2:
            if st.button("Abandon this build", key=f"ab{entry['id']}",
                         use_container_width=True):
                q.update(entry["id"], state=q.CANCELLED,
                         message="abandoned at review")
                st.rerun()
        st.divider()


# ------------------------------------------------------------------ library
def page_library() -> None:
    theme.head("Library", "Everything built so far.")

    jobs = runner.list_jobs()
    if not jobs:
        theme.empty("No builds yet.")
        return

    chosen = st.selectbox("Build", [j["name"] for j in jobs],
                          format_func=_job_title,
                          key="library_chosen_job")
    out_dir = runner.output_root() / chosen
    job = Job.load(out_dir)
    if job is None:
        theme.empty("That build could not be read.")
        return

    ui.profile_summary(job)
    ui.render_pipeline_stepper(job)
    st.caption(ui.stage_caption(job))

    clips = sum(1 for s in job.scenes if s.has_asset)
    ui.metrics([
        ("Scenes", len(job.scenes)),
        ("Clips", f"{clips}/{len(job.scenes)}"),
        ("Length", f"{sum(s.duration for s in job.scenes):.0f}s"),
        ("Quality", labels.model_label((job.options or {}).get("video_model", ""))),
        ("Updated", ui.relative_time(job.updated)),
    ])
    ui.render_qa_inspector(job)

    video = Path(job.video_path) if job.video_path else None
    if video and video.exists():
        st.markdown("### 🎬 Studio Screening Room")
        if job.url and ("youtube.com" in job.url or "youtu.be" in job.url):
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.markdown("##### 📌 Source Reference Video")
                st.video(job.url)
            with vcol2:
                st.markdown("##### ✨ Generated AI Video")
                st.video(str(video))
                st.download_button("📥 Download Video (MP4)", video.read_bytes(),
                                   file_name=f"{chosen}.mp4", mime="video/mp4",
                                   use_container_width=True)
        else:
            st.video(str(video))
            st.download_button("📥 Download Video (MP4)", video.read_bytes(),
                               file_name=f"{chosen}.mp4", mime="video/mp4")
    elif runner.is_running(out_dir):
        theme.note("This build is still running. The Queue page shows progress.")
        stills = [s for s in job.scenes if s.has_still or s.still_path or s.still_url]
        if stills or job.done("stills"):
            ui.render_contact_sheet(job)
    else:
        theme.note("No finished video yet.", "warn")

    tab_scenes, tab_contact, tab_story, tab_publish, tab_log = st.tabs(["Scenes & QA", "🎬 Contact Sheet", "Content DNA & Story", "📦 YouTube Publishing Kit", "Output"])

    with tab_publish:
        ui.render_publishing_kit(job)

    with tab_contact:
        ui.render_contact_sheet(job)

    with tab_scenes:
        for record in job.scenes:
            scol1, scol2 = st.columns([1, 3])
            with scol1:
                if record.still_path and Path(record.still_path).exists():
                    st.image(record.still_path, use_container_width=True)
            with scol2:
                badge = (f'<span class="oc-pill '
                         f'{labels.provider_class(record.asset_source)}">'
                         f'{html.escape(labels.provider_label(record.asset_source))}'
                         f'</span>')
                purpose_label = f"[{record.purpose.upper()}] " if record.purpose else ""
                qa_badge = f'<span class="oc-pill ok">QA {record.quality_score}/100</span>' if record.quality_score else ""
                st.markdown(
                    f'<div class="oc-row"><div class="oc-grow">'
                    f'<div class="oc-title">Scene {record.index} at '
                    f'{record.duration:.1f}s — <em>{purpose_label}</em></div></div>{qa_badge} {badge}</div>',
                    unsafe_allow_html=True)
                if record.narration:
                    st.caption(f"**Narration:** {record.narration[:220]}")
                st.caption(f"**Visual Prompt:** {labels.mask(record.visual_query[:200])}")

        st.divider()
        redo = st.text_input("Rebuild scenes", placeholder="For example 3,7",
                             help="Only these are made again. The rest are kept.")
        if st.button("Rebuild those scenes", disabled=not redo.strip()):
            wanted = [int(p) for p in redo.replace(" ", "").split(",")
                      if p.isdigit()]
            options = dict(job.options or {})
            options["repair"] = wanted
            entry = q.add(job.url, chosen, options)
            q.update(entry["id"], message=f"rebuilding scenes {wanted}")
            worker.ensure_running()
            st.success(f"Queued a rebuild of scenes {wanted}.")

    with tab_story:
        if job.classification:
            c = job.classification
            st.markdown(f"### Classification: **{c.get('primary_type', '').replace('_', ' ').title()}** ({c.get('format', 'short').upper()})")
            st.caption(f"Strategy: {c.get('recommended_generation_strategy', '')}")

        if job.content_dna:
            dna = job.content_dna
            with st.expander("🧬 Content DNA", expanded=True):
                st.markdown(f"**Hook:** {dna.get('hook', {}).get('mechanism', '')}")
                st.markdown(f"**Promise:** {dna.get('promise', {}).get('core_value', '')}")
                st.markdown(f"**Pacing:** {dna.get('pacing', {}).get('rhythm', '')}")
                st.markdown(f"**Emotional Arc:** {', '.join(dna.get('emotional_arc') or [])}")
                st.markdown(f"**Avoid:** {', '.join(dna.get('things_to_avoid') or [])}")

        if job.style_bible:
            sb = job.style_bible
            with st.expander("📖 Style Bible", expanded=False):
                st.markdown(f"**Visual Style:** {sb.get('visual_style', '')}")
                st.markdown(f"**Cinematography:** {sb.get('cinematography', '')}")
                st.markdown(f"**Color Palette:** {sb.get('color_language', '')}")
                st.markdown(f"**Lighting:** {sb.get('lighting', '')}")

        if job.style.get("logline"):
            st.markdown(f"**Logline:** {labels.mask(job.style['logline'])}")

    with tab_log:
        ui.log_view(out_dir, 200)


def _job_title(name: str) -> str:
    job = Job.load(runner.output_root() / name)
    if job and job.source.get("title"):
        return labels.mask(job.source["title"])[:70]
    return name


# ----------------------------------------------------------------- settings
def page_settings() -> None:
    theme.head("Settings", "How this copy of the tool is set up.")

    cfg = load_settings()
    slots = len(agnes_keys())
    ui.metrics([
        ("Generation slots", slots),
        ("Scene budget", cfg["shots"].get("scene_cap", 40)),
        ("Longest scene", f"{cfg['shots'].get('max_scene_seconds', 19)}s"),
        ("Frame", cfg["video"]["aspect_ratio"]),
    ])

    if not slots:
        theme.note("No generation keys are configured, so nothing can be built. "
                   "Add them to the environment and restart.", "bad")

    st.markdown("#### Queue supervisor")
    if worker.is_running():
        theme.note("Running. It will work through the queue on its own.")
    else:
        theme.note("Not running. It starts by itself when something is queued.")

    st.markdown("#### Where things live")
    st.caption(f"Builds: {runner.output_root()}")
    st.caption(f"Cache: {resolve_path(cfg['ingest']['cache_dir'])}")

    st.markdown("#### Defaults")
    st.caption("From settings.toml. Any of these can be overridden per build.")
    st.json({
        "scene_cap": cfg["shots"].get("scene_cap"),
        "max_scene_seconds": cfg["shots"].get("max_scene_seconds"),
        "aspect_ratio": cfg["video"]["aspect_ratio"],
        "fps": cfg["video"]["fps"],
        "music_pieces": cfg["visuals"].get("music_pieces"),
    })


# --------------------------------------------------------------------- shell
PAGES = {
    "New build": page_new,
    "Queue": page_queue,
    "Review": page_review,
    "Library": page_library,
    "Settings": page_settings,
}


def main() -> None:
    theme.apply()
    tally = q.counts()

    with st.sidebar:
        ui.render_sidebar_header()

        page_names = list(PAGES.keys())
        nav_icons = {
            "New build": "🎬",
            "Queue": "⚡",
            "Review": "👁️",
            "Library": "🏛️",
            "Settings": "⚙️",
        }

        def _format_nav(name: str) -> str:
            icon = nav_icons.get(name, "•")
            if name == "Review" and tally[q.REVIEW]:
                return f"{icon}  {name}  [{tally[q.REVIEW]}]"
            if name == "Queue" and (tally[q.WAITING] + tally[q.RUNNING]):
                return f"{icon}  {name}  [{tally[q.WAITING] + tally[q.RUNNING]}]"
            return f"{icon}  {name}"

        if "selected_page" not in st.session_state:
            st.session_state["selected_page"] = "New build"

        curr = st.session_state["selected_page"]
        default_idx = page_names.index(curr) if curr in page_names else 0

        choice = st.radio(
            "Go to",
            page_names,
            index=default_idx,
            format_func=_format_nav,
            label_visibility="collapsed",
            key="nav_sidebar_radio",
        )
        st.session_state["selected_page"] = choice

        if tally[q.RUNNING]:
            st.markdown(theme.pill("⚡ Studio Rendering Active", "run"), unsafe_allow_html=True)
        if tally[q.REVIEW]:
            st.markdown(theme.pill(f"⚠️ {tally[q.REVIEW]} Scenes Need Review", "warn"),
                        unsafe_allow_html=True)

        ui.render_sidebar_service_lights(running_builds=tally[q.RUNNING])

    PAGES[st.session_state["selected_page"]]()


main()

"""The run() orchestrator, kept separate so pipeline.py stays readable."""

from __future__ import annotations

import json as _json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .core.compositor import _write_concat_list, compose
from .core.jobstore import SceneRecord, open_job
from .core.language import name_of, normalise, pick_voice, word_level_possible
from .core.music import MusicError, build_bed, mix_under
from .core.profile import SourceProfile, apply_audio_mode, detect_profile
from .core.estimate import describe, estimate_run
from .core.anomaly import (
    corrected_prompt, local_findings, summarise, vision_findings,
)
from .core.scraper import SourceVideo, fetch_metadata
from .core.script_brain import Scene, ScriptBrain, ScriptPlan, StyleProfile
from .core.shot_analyzer import Shot, analyse
from .core.storyboard import (
    MAX_GENERATED_CLIP,
    Storyboard,
    silent_storyboard,
    write_visual_storyboard,
    target_scene_seconds,
    write_storyboard,
)
from .core.subtitles import build_from_lines, build_subtitles
from .core.tts_engine import narrate_plan
from .core.video_rotator import AgnesProvider, VisualAsset, VisualSourcer
from .core.classifier import ContentClassification, ContentClassifier
from .core.dna import ContentDNA, extract_content_dna
from .core.strategy import CreativeStrategist, CreativeStrategy
from .core.script_engine import ScriptEngine
from .core.scene_planner import ProductionPlan, ProductionScene, ScenePlanner, StyleBible
from .core.prompt_engine import PromptEngine
from .core.retention import RetentionOptimizer
from .core.qa import VideoQA
from .utils.config import load_settings, resolve_path
from .utils.media import run as ffmpeg_run


def plan_from_board(board: Storyboard) -> ScriptPlan:
    """Adapt a storyboard to the plan shape the renderer already understands."""
    plan = ScriptPlan(
        title=board.title,
        style=StyleProfile(tone=board.logline, pacing_wpm=0.0,
                           structure=f"{len(board.panels)} scenes",
                           visual_style=board.style),
    )
    for panel in board.panels:
        plan.scenes.append(Scene(
            index=panel.index, narration=panel.narration,
            visual_query=panel.image_prompt(board.style, board.cast),
            start=panel.start, duration=panel.duration,
        ))
    return plan


def run(
    url: str,
    out_dir: str | Path,
    cfg: dict | None = None,
    mode: str = "auto",
    aspect: str | None = None,
    trim_seconds: float | None = None,
    start_seconds: float = 0.0,
    max_scenes: int | None = None,
    prefer_ai: bool = True,
    allow_stock: bool = True,
    translate: bool = False,
    language: str | None = None,
    caption_language: str = "en",
    caption_style: str = "kinetic",
    music: bool = True,
    visual_workers: int = 4,
    video_model: str | None = None,
    fresh: bool = False,
    repair: list[int] | None = None,
    pause_after: str | None = None,
    audio_mode: str = "auto",
    captions: bool | None = None,
    voice: str | None = None,
    kind: str | None = None,
    scene_cap: int | None = None,
    review: bool = True,
    on_step=None,
    on_progress=None,
):
    from .pipeline import (
        IngestQualityError, PipelinePaused, PipelineResult, _ingest,
        _plan_from_job, _source_from_job, _store_plan, _vision_fallback,
        agnes_keys, build_providers, frame_spec, is_genuinely_silent,
        transcript_health, with_backoff,
    )

    cfg = cfg or load_settings()
    frame = frame_spec(cfg, aspect)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    options = {
        "aspect": frame["aspect"], "mode": mode, "allow_stock": allow_stock,
        "prefer_ai": prefer_ai, "translate": translate,
        "trim_seconds": trim_seconds, "max_scenes": max_scenes,
        "video_model": video_model or cfg["visuals"]["agnes_model"],
        "language": language, "caption_language": caption_language,
        "caption_style": caption_style,
        "music": music,
    }
    job = open_job(out_dir, url, options, fresh)
    job.options = options
    if repair:
        cleared = job.clear_assets(repair)
        if on_step:
            on_step(f"repairing scenes {cleared}")

    def step(name):
        if on_step:
            on_step(name)
        return time.time()

    def progress(stage):
        return (lambda i, n: on_progress(stage, i, n)) if on_progress else None

    brain = ScriptBrain(
        base_url=cfg["script"]["base_url"], api_key=cfg["script"]["api_key"],
        model=cfg["script"]["model"], temperature=cfg["script"]["temperature"],
        max_retries=cfg["script"]["max_retries"],
        max_completion_tokens=cfg["script"]["max_completion_tokens"],
        reasoning_effort=cfg["script"]["reasoning_effort"],
        default_wpm=cfg["script"]["default_wpm"],
        fallback=cfg["script"].get("fallback"),
    )

    # ---- ingest -----------------------------------------------------------
    if job.done("ingest"):
        source = _source_from_job(job)
        speech_lang = job.stages["ingest"].get("language", "en")
        silent = bool(job.stages["ingest"].get("silent"))
    else:
        t = step("ingest")
        source = _ingest(url, cfg, translate, progress("ingest"))
        if source is None:
            info = fetch_metadata(url)
            source = SourceVideo(
                url=url, video_id=info.get("id"),
                title=info.get("title") or "Untitled",
                description=info.get("description") or "",
                uploader=info.get("uploader") or info.get("channel") or "",
                duration_seconds=float(info.get("duration") or 0.0),
            )

        health = transcript_health(source.transcript, source.duration_seconds)
        # Decided before the gate below, because "barely any words" is both what
        # a failed transcription looks like and what a wordless film looks like.
        # Judging them together refused the films along with the failures.
        silent = mode == "mirror" or (
            mode == "auto" and is_genuinely_silent(health, source.duration_seconds)
        )
        if health["broken"] or (not silent and health["words"] < 8
                                and source.duration_seconds > 20):
            raise IngestQualityError(
                "The transcript is unusable, so there is nothing to rewrite.\n"
                f"  usable words : {health['words']} over "
                f"{source.duration_seconds:.0f}s ({health['rate']}/s)\n"
                f"  filler ratio : {health['foreign_ratio']:.0%} 'Foreign'\n"
                f"  audio events : {health['tags']} tags"
                f"{' (music present)' if health['has_music'] else ''}\n"
                "Speech recognition likely does not support this language. "
                "Building on this would produce a silent slideshow, so the run "
                "stops here instead."
            )

        speech_lang = normalise(language or source.language or "en")
        job.source = {
            "url": source.url, "video_id": source.video_id, "title": source.title,
            "description": source.description[:2000], "uploader": source.uploader,
            "duration_seconds": source.duration_seconds,
            "language": source.language, "transcript": source.transcript,
        }
        timings["ingest"] = time.time() - t
        job.mark("ingest", language=speech_lang, silent=silent, **health)

    speech_lang = normalise(language or speech_lang)
    caption_lang = normalise(caption_language)

    # ---- shots ------------------------------------------------------------
    if job.done("shots") and job.scenes:
        shots = [Shot(index=r.index, start=r.start, duration=r.duration,
                      description=r.visual_query) for r in job.scenes]
        style_phrase = (job.style or {}).get("visual_style", "")
        cast = (job.style or {}).get("cast", [])
    else:
        t = step("analyse shots")
        target = target_scene_seconds(trim_seconds or source.duration_seconds)
        shots, style_phrase, video_file = analyse(
            url,
            cache_dir=resolve_path(cfg["ingest"]["cache_dir"]) / "video",
            api_key=cfg["script"]["api_key"], base_url=cfg["script"]["base_url"],
            model=cfg["shots"]["vision_model"],
            threshold=cfg["shots"]["scene_threshold"],
            min_shot_seconds=max(cfg["shots"]["min_shot_seconds"], target * 0.7),
            # Both ceilings are clamped to what one generation can fill. A shot
            # longer than that gets a looped clip, and the loop point is visible
            # as the last seconds of the scene repeating.
            max_shot_seconds=min(MAX_GENERATED_CLIP, target * 1.6),
            cluster_window=cfg["shots"]["cluster_window"],
            split_after=min(MAX_GENERATED_CLIP, target * 1.8),
            max_duration=trim_seconds,
            start_seconds=start_seconds,
            max_clip_seconds=float(
                cfg["shots"].get("max_scene_seconds", MAX_GENERATED_CLIP)),
            scene_cap=int(scene_cap or cfg["shots"].get("scene_cap", 40)),
            batch_size=cfg["shots"]["vision_batch_size"],
            max_tokens=cfg["shots"]["vision_max_tokens"],
            min_interval=cfg["shots"]["vision_min_interval"],
            frame_width=cfg["shots"]["vision_frame_width"],
            fallback=_vision_fallback(cfg),
            progress=progress("analyse shots"),
        )
        if max_scenes:
            shots = shots[:max_scenes]
        cached = _json.loads(
            video_file.with_suffix(".shots.json").read_text(encoding="utf-8"))
        cast = cached.get("cast") or []
        job.set_scenes([SceneRecord(index=s.index, start=s.start,
                                    duration=s.duration, visual_query=s.description)
                        for s in shots])
        job.style = {"visual_style": style_phrase, "cast": cast}
        timings["analyse shots"] = time.time() - t
        forecast = estimate_run(len(shots), cfg, narrated=not silent)
        job.mark("shots", count=len(shots), target_scene_seconds=round(target, 1),
                 **forecast)
        # Said out loud before anything expensive starts, because the cost of a
        # long source was previously discovered only by waiting through it.
        print(f"   estimate: {describe(forecast)}", flush=True)

    # ---- classification & DNA --------------------------------------------
    if job.classification and job.content_dna:
        classification = ContentClassification.from_dict(job.classification)
        dna = ContentDNA.from_dict(job.content_dna)
    else:
        classifier = ContentClassifier()
        shots_summary = "\n".join(f"[{s.index}] {s.duration:.1f}s: {s.description}" for s in shots[:15])
        classification = classifier.classify(
            title=source.title,
            description=source.description,
            duration=trim_seconds or source.duration_seconds,
            aspect_ratio=frame["aspect"],
            transcript=source.transcript,
            shots_summary=shots_summary,
            user_target=options.get("target_platform") or options.get("format"),
        )
        dna = extract_content_dna(
            title=source.title,
            transcript=source.transcript,
            classification=classification,
            shots_summary=shots_summary,
            duration=trim_seconds or source.duration_seconds,
        )
        job.classification = classification.to_dict()
        job.content_dna = dna.to_dict()
        job.save()

    # Creative strategy (Originality Engine: what to preserve vs what must change)
    if job.creative_strategy:
        strategy = CreativeStrategy.from_dict(job.creative_strategy)
    else:
        strategist = CreativeStrategist()
        strategy = strategist.develop_strategy(
            dna=dna,
            classification=classification,
            source_title=source.title,
            source_transcript=source.transcript,
            user_idea=options.get("idea") or options.get("topic"),
            user_instructions=options.get("instructions"),
            target_platform=options.get("target_platform") or classification.format,
            cast=cast,
            shots=shots,
            style_phrase=style_phrase,
        )
        job.creative_strategy = strategy.to_dict()
        job.save()

    # ---- profile ----------------------------------------------------------
    if job.profile and not kind:
        profile = SourceProfile.from_dict(job.profile)
    else:
        t = step("profile")
        if kind:
            profile = SourceProfile.for_kind(kind)
            profile.source = "override"
            profile.reason = "chosen by the operator"
        else:
            profile = detect_profile(
                brain, title=source.title, transcript=source.transcript,
                style=style_phrase, duration=source.duration_seconds,
                native=(job.stages.get("shots") or {}).get(
                    "target_scene_seconds", 0.0),
                donor_has_music=bool(
                    (job.stages.get("ingest") or {}).get("has_music")),
            )
        profile = apply_audio_mode(profile, audio_mode)
        if captions is not None:
            profile.captions = captions
        job.profile = profile.to_dict()
        timings["profile"] = time.time() - t
        job.save()
    profile = SourceProfile.from_dict(job.profile)

    # Photoreal has to be said out loud; the donor-derived look does not carry
    # it, and an image model left to itself drifts to illustration.
    if profile.realism_phrase:
        style_phrase = f"{style_phrase.rstrip('. ')}. {profile.realism_phrase}"

    # A profile with no narration is silent, whatever the transcript said.
    silent = silent or not profile.wants_voice or classification.narration_dependency == "none"

    # ---- storyboard -------------------------------------------------------
    if job.done("storyboard"):
        plan = _plan_from_job(job)
    else:
        t = step("storyboard")
        try:
            script_engine = ScriptEngine()
            target_dur = trim_seconds or (source.duration_seconds if source.duration_seconds > 0 else 60.0)
            target_count = len(shots) if shots else max(4, int(target_dur / 4.0))
            script = script_engine.generate(
                strategy=strategy,
                classification=classification,
                dna=dna,
                target_duration=target_dur,
                target_scenes_count=target_count,
                language=speech_lang,
                cast=cast,
                shots=shots,
            )

            planner = ScenePlanner()
            prod_plan = planner.plan(
                script=script,
                strategy=strategy,
                classification=classification,
                dna=dna,
                cast=cast,
                shots=shots,
                style_phrase=style_phrase,
            )

            prompt_eng = PromptEngine()
            for sc in prod_plan.scenes:
                chk = prompt_eng.validate_and_refine(sc, prod_plan.style_bible, classification)
                if chk.refined_prompt:
                    sc.visual_prompt = chk.refined_prompt

            retention_opt = RetentionOptimizer()
            retention_opt.optimize_plan(prod_plan, is_short=(classification.format == "short"))

            (out_dir / "production_plan.json").write_text(
                _json.dumps(prod_plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            job.style_bible = prod_plan.style_bible.to_dict()

            plan = ScriptPlan(
                title=prod_plan.title,
                style=StyleProfile(
                    tone=strategy.hook_strategy,
                    pacing_wpm=dna.pacing.get("words_per_minute", 145),
                    structure=f"{len(prod_plan.scenes)} scenes",
                    visual_style=prod_plan.style_bible.visual_style,
                ),
            )
            cum_time = 0.0
            new_scenes = []
            for sc in prod_plan.scenes:
                plan.scenes.append(Scene(
                    index=sc.scene_id,
                    narration=sc.narration,
                    visual_query=sc.visual_prompt,
                    start=cum_time,
                    duration=sc.duration,
                ))
                rec = job.scene(sc.scene_id)
                if not rec:
                    rec = SceneRecord(index=sc.scene_id)
                rec.start = cum_time
                rec.duration = sc.duration
                rec.narration = sc.narration
                rec.visual_query = sc.visual_prompt
                rec.purpose = sc.purpose
                rec.visual_type = sc.visual_type
                rec.generation_mode = sc.generation_mode
                new_scenes.append(rec)
                cum_time += sc.duration
            job.scenes = new_scenes

            (out_dir / "storyboard.json").write_text(plan.to_json(), encoding="utf-8")
            job.style = {
                **(job.style or {}),
                "logline": script.logline,
                "visual_style": prod_plan.style_bible.visual_style,
            }
            job.mark("storyboard", logline=script.logline, primary_type=classification.primary_type)
        except Exception as exc:
            print(f"   storyboard error, using legacy fallback: {exc}")
            board = write_storyboard(
                brain, title=f"{source.title} (OmniClip original)",
                transcript=source.transcript, shots=shots,
                style=style_phrase, cast=cast,
                language=speech_lang, language_name=name_of(speech_lang),
                wpm=profile.wpm,
            )
            (out_dir / "storyboard.json").write_text(board.to_json(), encoding="utf-8")
            plan = plan_from_board(board)
            for panel in board.panels:
                record = job.scene(panel.index)
                if record:
                    record.narration = panel.narration
                    record.narration_en = panel.narration_en
                    record.visual_query = panel.image_prompt(board.style, board.cast)
            job.style = {**(job.style or {}), "logline": board.logline}
            job.mark("storyboard", logline=board.logline, language=speech_lang)
        timings["storyboard"] = time.time() - t
    _store_plan(job, plan)
    if pause_after == "storyboard":
        raise PipelinePaused("storyboard")

    # ---- narration and captions -------------------------------------------
    if silent:
        audio_paths = None
        job.mark("narrate", silent=True)
        job.mark("captions", silent=True)
    elif job.done("captions") and all(r.has_audio for r in job.scenes):
        audio_paths = [Path(r.audio_path) for r in job.scenes]
    else:
        t = step("narrate")
        chosen_voice = voice or pick_voice(speech_lang)
        # The profile's rate is the deliberate one -- a meditation is slow
        # because it is a meditation, not because the donor happened to be.
        # An explicit rate in settings still overrules it, as it always did.
        configured_rate = (cfg["tts"].get("rate") or "").strip()
        speech = narrate_plan(
            plan, out_dir / "audio", voice=chosen_voice,
            rate=(configured_rate
                  if configured_rate not in ("", "+0%", "0%")
                  else profile.voice_rate),
            volume=cfg["tts"]["volume"], progress=progress("narrate"),
            preserve_durations=True,
        )
        for scene, spoken in zip(plan.scenes, speech):
            record = job.scene(scene.index)
            record.audio_path = str(spoken.audio_path)
            record.start, record.duration = scene.start, scene.duration
        audio_paths = [s.audio_path for s in speech]
        timings["narrate"] = time.time() - t
        job.mark("narrate", voice=chosen_voice, language=speech_lang)

        t = step("captions")
        subtitles = None
        chosen_style = options.get("caption_style") or (job.options or {}).get("caption_style") or "kinetic"
        if not profile.captions:
            # Burned-in words are wrong for a piece meant to be listened to
            # with the eyes closed.
            pass
        elif word_level_possible(speech_lang, caption_lang):
            subtitles = build_subtitles(
                speech, out_dir, basename="captions",
                width=frame["width"], height=frame["height"],
                style=chosen_style)
        else:
            # The captions are a different language from the voice, so real
            # word timings do not exist; they are spread across each scene.
            subtitles = build_from_lines(
                [(r.narration_en or r.narration, r.start, r.duration)
                 for r in job.scenes],
                out_dir, basename="captions",
                width=frame["width"], height=frame["height"],
                style=chosen_style)
        timings["captions"] = time.time() - t
        if subtitles is None:
            job.mark("captions", skipped="captions off for this kind")
        else:
            job.mark("captions",
                     ass=str(subtitles["ass"]), srt=str(subtitles["srt"]),
                     language=caption_lang,
                     style=chosen_style,
                     word_level=word_level_possible(speech_lang, caption_lang))

    caption_file = (Path(job.stages["captions"]["ass"])
                    if job.stages["captions"].get("ass") else None)

    # ---- stills -----------------------------------------------------------
    # A still per scene anchors the clip animated from it. Without one, every
    # clip is an unrelated hallucination of a similar-sounding sentence.
    pending_stills = [r for r in job.scenes if not r.has_still]
    if pending_stills and prefer_ai:
        t = step("stills")
        still_provider = AgnesProvider(
            api_key=agnes_keys(), rpm=cfg["queue"]["requests_per_minute"],
            width=frame["gen_width"], height=frame["gen_height"],
            pool_rpm=cfg["queue"]["pool_requests_per_minute"],
        )
        still_dir = resolve_path(cfg["ingest"]["cache_dir"]) / "stills"
        size = ("1024x1024" if frame["width"] == frame["height"]
                else "1024x1792" if frame["height"] > frame["width"]
                else "1792x1024")
        # Four at a time. The image queue is separate from the video one and
        # allows considerably more than this; the cap is deliberately modest
        # because the point is to stop waiting, not to find the ceiling.
        lock = threading.Lock()
        done_count = 0

        def make_one(record):
            """One still, retried, because a lost still is a lost scene."""
            return with_backoff(
                lambda: still_provider.make_still(
                    record.visual_query, still_dir, record.variant, size),
                attempts=3, first_wait=5.0,
            )

        with ThreadPoolExecutor(max_workers=min(4, len(pending_stills))) as pool:
            futures = {pool.submit(make_one, r): r for r in pending_stills}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    made = future.result()
                except Exception:
                    made = None
                # One writer at a time: the job file is rewritten whole, so two
                # threads saving at once would race over the same file.
                with lock:
                    done_count += 1
                    if made:
                        record.still_url, made_path = made[0], made[1]
                        record.still_path = str(made_path)
                        job.save()
                    if on_progress:
                        on_progress("stills", done_count, len(pending_stills))
        timings["stills"] = time.time() - t
    job.mark("stills", made=sum(1 for r in job.scenes if r.has_still))

    # ---- review -----------------------------------------------------------
    # Between the cheap thing and the expensive thing. A bad still costs
    # seconds to replace here and a clip generation plus a render to discover
    # later, which is why every defect so far was found by a person watching
    # the finished video.
    if review and prefer_ai and not job.done("review"):
        t = step("review")
        wanted = {}
        for record in job.scenes:
            found = re.search(r"Exactly (\d+) (?:person|people) in frame",
                              record.visual_query or "")
            wanted[record.index] = int(found.group(1)) if found else None

        still_provider = AgnesProvider(
            api_key=agnes_keys(), rpm=cfg["queue"]["requests_per_minute"],
            width=frame["gen_width"], height=frame["gen_height"],
            pool_rpm=cfg["queue"]["pool_requests_per_minute"],
        )
        still_dir = resolve_path(cfg["ingest"]["cache_dir"]) / "stills"
        size = ("1024x1024" if frame["width"] == frame["height"]
                else "1024x1792" if frame["height"] > frame["width"]
                else "1792x1024")

        fixed, unresolved = 0, []
        checked = list(job.scenes)
        for attempt in range(2):
            findings = local_findings(checked) + vision_findings(
                checked, wanted, cfg, progress=progress("review"))
            by_scene = {}
            for finding in findings:
                by_scene.setdefault(finding.scene, []).append(finding)
            if not by_scene:
                checked = []
                break
            if attempt == 1:
                unresolved = [
                    {"scene": i, "problems": [f.detail for f in group]}
                    for i, group in sorted(by_scene.items())
                ]
                break
            # One correction pass: the prompt is rewritten to name the defect
            # and the still is made again from scratch.
            retry = []
            for index, group in sorted(by_scene.items()):
                record = job.scene(index)
                if record is None or record.locked:
                    continue
                record.visual_query = corrected_prompt(record.visual_query, group)
                record.attempt += 1
                try:
                    made = still_provider.make_still(
                        record.visual_query, still_dir, record.variant, size)
                except Exception:
                    made = None
                if made:
                    record.still_url, made_path = made[0], made[1]
                    record.still_path = str(made_path)
                    fixed += 1
                retry.append(record)
            job.save()
            checked = retry

        timings["review"] = time.time() - t
        job.mark("review", checked=len(job.scenes), fixed=fixed,
                 needs_review=bool(unresolved), problems=unresolved,
                 summary=summarise([], len(job.scenes), fixed) if not unresolved
                 else f"{len(unresolved)} scene(s) still need a look")

        # Stop here. Flagging a bad still and then generating its clip anyway
        # spends exactly what this stage exists to save, which is what the
        # first batch did: three scenes reported as wrong, three clips paid
        # for, a finished video nobody would keep.
        if unresolved:
            raise PipelinePaused("review")

    if pause_after == "stills":
        raise PipelinePaused("stills")

    # ---- visuals ----------------------------------------------------------
    for scene in plan.scenes:
        record = job.scene(scene.index)
        if record:
            scene.variant = record.variant

    outstanding = [s for s in plan.scenes
                   if not (job.scene(s.index) or SceneRecord(-1)).has_asset]
    if outstanding:
        t = step("visuals")
        sourcer = VisualSourcer(
            build_providers(cfg, prefer_ai, allow_stock, frame, video_model),
            resolve_path(cfg["ingest"]["cache_dir"])
            / frame["aspect"].replace(":", "x"),
        )
        for provider in sourcer.providers:
            if getattr(provider, "name", "") == "agnes":
                provider.references = {
                    r.variant: r.still_url for r in job.scenes if r.still_url}
        pending = ScriptPlan(title=plan.title, style=plan.style)
        pending.scenes = outstanding

        def remember(scene, asset):
            record = job.scene(scene.index)
            if record is not None:
                record.asset_path = str(asset.path)
                record.asset_source = asset.source
                job.save()

        sourcer.fetch_for_plan(pending, progress=progress("visuals"),
                               max_workers=visual_workers, on_asset=remember)
        timings["visuals"] = time.time() - t
    job.mark("visuals", generated=len(outstanding))

    # ---- qa ---------------------------------------------------------------
    t = step("qa")
    try:
        qa_agent = VideoQA()
        scenes_data = [
            {
                "index": s.index,
                "purpose": (job.scene(s.index) or SceneRecord(0)).purpose or "demonstrate",
                "narration": s.narration,
                "visual_prompt": s.visual_query,
                "asset_source": (job.scene(s.index) or SceneRecord(0)).asset_source,
            }
            for s in plan.scenes
        ]
        prod_plan = ProductionPlan(
            title=plan.title,
            style_bible=StyleBible.from_dict(job.style_bible or {}),
            scenes=[],
        )
        qa_report = qa_agent.evaluate_scenes(prod_plan, scenes_data)
        job.qa_report = qa_report.to_dict()
        for r in qa_report.scene_results:
            rec = job.scene(r.scene_id)
            if rec:
                rec.quality_score = r.quality_score
        job.mark("qa", score=qa_report.overall_score, passed=qa_report.passed)
        print(f"   QA verdict: {qa_report.summary} (score: {qa_report.overall_score}/100)", flush=True)
    except Exception as exc:
        print(f"   QA check skipped: {exc}", flush=True)
    timings["qa"] = time.time() - t

    assets = [
        VisualAsset(path=Path(job.scene(s.index).asset_path),
                    source=job.scene(s.index).asset_source, query=s.visual_query)
        for s in plan.scenes
    ]

    # ---- music ------------------------------------------------------------
    mixed_track = None
    # A wordless source still gets a score, because most wordless sources have
    # one. Without this the rebuild of a mute cartoon came back in dead silence.
    # Whether there is music at all is the profile's call: a bulletin gets none,
    # a meditation is mostly music. `--no-music` still overrules everything.
    if music and profile.wants_music and (audio_paths or silent):
        t = step("music")
        try:
            work = out_dir / "work"
            work.mkdir(parents=True, exist_ok=True)
            bed = build_bed(
                # A brief written for this video, not a keyword guess at one.
                profile.music_mood,
                cfg["script"]["api_key"],
                # Overrun only helps when the bed is trimmed to something else.
                plan.duration_seconds + (0 if silent else 2), work,
                progress=progress("music"),
                # Shared across every job: a mood is bought once, then reused.
                library=resolve_path(cfg["ingest"]["cache_dir"]) / "music",
                pieces_wanted=cfg["visuals"].get("music_pieces", 3))
            if audio_paths:
                joined = work / "narration_only.m4a"
                ffmpeg_run(
                    ["-f", "concat", "-safe", "0", "-i",
                     str(_write_concat_list([Path(p) for p in audio_paths],
                                            work / "audio.txt")),
                     "-c:a", "aac", "-b:a", "192k", str(joined)],
                    label="join narration")
                # A score that carries the piece sits forward; a bed
                # under a bulletin stays out of the way.
                mixed_track = mix_under(
                    joined, bed, work / "narration_mixed.m4a",
                    music_db=-12.0 if profile.music == "prominent" else -20.0)
            else:
                # Nothing to sit under, so the bed carries the whole track and
                # is not ducked the way it would be beneath a voice.
                mixed_track = bed
            timings["music"] = time.time() - t
            job.mark("music", bed=str(bed))
        except Exception as exc:
            job.mark("music", skipped=f"{type(exc).__name__}: {exc}"[:180])
    else:
        job.mark("music", skipped="disabled by request")

    # ---- render -----------------------------------------------------------
    out_path = out_dir / "video.mp4"
    if job.done("render") and out_path.exists() and not repair:
        video = out_path
    else:
        t = step("render")
        video = compose(
            plan, assets,
            [mixed_track] if mixed_track else audio_paths,
            out_path=out_path, work_dir=out_dir / "work",
            subtitles=caption_file,
            width=frame["width"], height=frame["height"], fps=frame["fps"],
            progress=progress("render"),
            single_audio=bool(mixed_track),
        )
        timings["render"] = time.time() - t
        job.video_path = str(video)
        job.mark("render", path=str(video))

    return PipelineResult(
        video_path=video, plan=plan, source=source,
        mode="silent" if silent else "narrated",
        assets=assets, timings=timings, frame=frame, job=job,
    )

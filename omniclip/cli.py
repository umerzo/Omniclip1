"""Command-line entry point.

    python -m omniclip <url> [options]

Windows consoles default to cp1252, which cannot print the emoji that appear in
most Shorts titles. Printing one raises UnicodeEncodeError and would abort the
run *after* the video was already rendered, so stdout is switched to UTF-8
before anything is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.jobstore import Job
from .core.profile import KINDS
from .core.video_rotator import QuotaExhausted
from .pipeline import PipelinePaused, agnes_keys, run
from .utils.config import load_settings, resolve_path


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def record_stop(out: Path, reason: str, detail: str = "", **extra) -> None:
    """Write why this run ended into the job file.

    Every caller -- this CLI, and the queue worker that shells out to it --
    reads that record to decide what happens next. It used to be inferred by
    matching the log tail, which made the decision depend on whether the last
    line had finished being written.

    Best effort on purpose: a run that failed before writing a job has nothing
    to record against, and failing to record must not turn a clean failure into
    a crash on the way out.
    """
    try:
        job = Job.load(out)
        if job is not None:
            job.mark_stop(reason, detail, **extra)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omniclip",
        description="Rebuild any video link as an original video.",
    )
    parser.add_argument("url", help="source video URL")
    parser.add_argument("-o", "--out", default=None,
                        help="output directory (default: output/<video id>)")
    parser.add_argument("--mode", choices=("auto", "narrated", "mirror"), default="auto",
                        help="auto picks mirror when the source barely speaks")
    parser.add_argument("--aspect", choices=("9:16", "16:9", "1:1"), default=None,
                        help="output frame (default: settings.toml)")
    parser.add_argument("--no-stock", action="store_true",
                        help="AI generation only; stop rather than use stock "
                             "footage (overrides visuals.allow_stock)")
    parser.add_argument("--no-ai", action="store_true",
                        help="stock footage only; skip AI generation")
    parser.add_argument("--translate", action="store_true",
                        help="translate a non-English source into English")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel clip fetches (default: one per Agnes key)")
    parser.add_argument("--trim", type=float, default=None,
                        help="only use the first N seconds of the source")
    parser.add_argument("--start", type=float, default=0.0,
                        help="skip the first N seconds (titles, credits)")
    parser.add_argument("--pause-after", choices=["storyboard", "stills"],
                        default=None,
                        help="stop after this stage to review before it "
                             "spends anything on clips; rerun the same "
                             "command without this flag to continue")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="stop after N scenes (useful for quick tests)")
    parser.add_argument("--language", default=None,
                        help="spoken language (default: same as the source)")
    parser.add_argument("--caption-language", default="en",
                        help="caption language (default: English)")
    parser.add_argument("--no-music", action="store_true",
                        help="skip the generated music bed")
    parser.add_argument("--audio", choices=["auto", "voiceover", "music", "silent"],
                        default="auto",
                        help="what the soundtrack is: auto follows the detected "
                             "kind; voiceover drops music; music drops the "
                             "voice; silent drops both")
    parser.add_argument("--no-captions", action="store_true",
                        help="never burn in captions, whatever the kind wants")
    parser.add_argument("--caption-style", choices=["kinetic", "standard"],
                        default="kinetic",
                        help="subtitle style: kinetic (pop-in animated) or standard (static)")
    parser.add_argument("--voice", default=None,
                        help="edge-tts voice name (default: chosen by language)")
    parser.add_argument("--kind", choices=list(KINDS), default=None,
                        help="override the detected kind of video")
    parser.add_argument("--no-review", action="store_true",
                        help="skip the automatic still check before clips")
    parser.add_argument("--scene-cap", type=int, default=None,
                        help="how many scenes this video is worth; one scene is "
                             "about a minute of generation (default: settings)")
    parser.add_argument("--video-model", default=None,
                        help="Agnes model for clip generation")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore saved progress and start the job over")
    parser.add_argument("--repair", default=None,
                        help="regenerate only these scenes, e.g. --repair 3,7")
    parser.add_argument("--lock", default=None,
                        help="mark scenes as keepers so repairs skip them, e.g. --lock 0,1")
    parser.add_argument("--status", action="store_true",
                        help="print saved job progress and exit")
    return parser


def _scene_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(p) for p in str(text).replace(" ", "").split(",") if p.strip("-").isdigit()]


def show_status(out: Path) -> int:
    job = Job.load(out)
    if not job:
        print(f"No saved job in {out}")
        return 1
    state = job.summary()
    print(f"url    : {state['url']}")
    print(f"title  : {state['title']}")
    print(f"stage  : {state['stage']}")
    print("stages : " + "  ".join(
        f"{'x' if done else '.'} {name}" for name, done in state["stages"].items()
    ))
    print(f"scenes : {state['scenes_with_clips']}/{state['scenes']} have clips, "
          f"{state['locked']} locked")
    if state["video"]:
        print(f"video  : {state['video']}")
    print()
    for record in job.scenes:
        flags = "".join(("L" if record.locked else "-",
                         "A" if record.has_asset else "-",
                         "S" if record.has_audio else "-"))
        print(f"  [{record.index:>2}] {flags} {record.duration:5.1f}s "
              f"{record.asset_source or '-':8} {record.visual_query[:56]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)
    cfg = load_settings()

    if args.out:
        out = Path(args.out)
    else:
        url_clean = args.url.rstrip("/")
        if "v=" in url_clean:
            slug = url_clean.split("v=")[1].split("&")[0]
        else:
            slug = url_clean.split("/")[-1].split("?")[0]
        slug = "".join(c for c in slug if c.isalnum() or c in "-_") or "video"
        out = resolve_path(cfg["output"]["dir"]) / slug
    if args.status:
        return show_status(out)

    if args.lock:
        job = Job.load(out)
        if not job:
            print(f"No saved job in {out}", file=sys.stderr)
            return 1
        for index in _scene_list(args.lock):
            record = job.scene(index)
            if record:
                record.locked = True
        job.save()
        print(f"locked scenes {_scene_list(args.lock)}")

    workers = args.workers or min(max(1, len(agnes_keys())), 6)
    repair = _scene_list(args.repair)

    print(f"source : {args.url}")
    print(f"output : {out}")
    print(f"workers: {workers}  (agnes keys: {len(agnes_keys())})")
    existing = Job.load(out)
    if existing and not args.fresh:
        print(f"resume : from '{existing.resume_from()}' "
              f"({existing.summary()['scenes_with_clips']}/{len(existing.scenes)} clips ready)")

    try:
        result = run(
            args.url,
            out_dir=out,
            cfg=cfg,
            mode=args.mode,
            aspect=args.aspect,
            trim_seconds=args.trim,
            start_seconds=args.start,
            pause_after=args.pause_after,
            audio_mode=args.audio,
            captions=False if args.no_captions else None,
            caption_style=args.caption_style,
            voice=args.voice,
            kind=args.kind,
            scene_cap=args.scene_cap,
            review=not args.no_review,
            max_scenes=args.max_scenes,
            prefer_ai=not args.no_ai,
            # --no-stock overrides; otherwise the shared default applies.
            allow_stock=(cfg["visuals"].get("allow_stock", True)
                         and not args.no_stock),
            translate=args.translate,
            visual_workers=workers,
            language=args.language,
            caption_language=args.caption_language,
            music=not args.no_music,
            video_model=args.video_model,
            fresh=args.fresh,
            repair=repair,
            on_step=lambda name: print(f"\n== {name}", flush=True),
            on_progress=lambda stage, i, n: print(f"   {stage} {i}/{n}", flush=True),
        )
    except PipelinePaused as paused:
        record_stop(out, "paused", f"waiting for a person after '{paused.stage}'",
                    stage=paused.stage)
        if paused.stage == "review":
            print("\n== stopped before generating clips", flush=True)
            print("Some stills could not be corrected automatically. Look at "
                  "them, then rerun with --no-review to accept them as they "
                  "are, or delete the ones you want made again.", flush=True)
        else:
            print(f"\n== paused after '{paused.stage}' for review", flush=True)
            print("Rerun this exact command (without --pause-after) to continue.",
                  flush=True)
        return 0
    except QuotaExhausted as exc:
        # Not a failure: the clips already made are on disk and the run picks
        # up from them once the plans reset. Said plainly so that whoever reads
        # the console gets the same story the job file tells.
        record_stop(out, "quota", str(exc))
        print(f"\nSTOPPED: out of generation capacity. {exc}", file=sys.stderr)
        print("The clips made so far are saved; rerun to continue from them.",
              file=sys.stderr)
        return 2
    except Exception as exc:
        record_stop(out, "error", f"{type(exc).__name__}: {exc}")
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    record_stop(out, "complete", "rendered", video=str(result.video_path))
    print("\n" + "=" * 60)
    print(f"mode    : {result.mode}")
    print(f"frame   : {result.frame.get('aspect')} "
          f"{result.frame.get('width')}x{result.frame.get('height')}")
    print(f"source  : {result.source.title}")
    print(f"length  : {result.source.duration_seconds:.1f}s original "
          f"-> {result.duration:.1f}s generated")
    print(f"scenes  : {len(result.plan.scenes)}")
    for scene, asset in zip(result.plan.scenes, result.assets):
        print(f"   [{scene.index:>2}] {asset.source:<7} {scene.duration:5.1f}s  "
              f"{scene.visual_query[:60]}")
    used = ", ".join(sorted({a.source for a in result.assets}))
    print(f"sources : {used}")
    print("timings : " + ", ".join(f"{k}={v:.0f}s" for k, v in result.timings.items())
          + f", total={sum(result.timings.values()):.0f}s")
    print(f"video   : {result.video_path} "
          f"({result.video_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

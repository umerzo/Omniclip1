"""Headless compositor.

Renders the finished video with ffmpeg: one clip per scene, cut to the length of
that scene's narration, cropped to the target aspect ratio, with the voiceover
and burned-in captions on top.

Every clip is normalised to identical codec/size/fps first, which is what lets
the concat demuxer join hundreds of scenes without re-encoding the join.
"""

from __future__ import annotations

from pathlib import Path

from ..utils.media import escape_filter_path, probe_duration, run


def _write_concat_list(paths: list[Path], list_path: Path) -> Path:
    lines = [f"file '{str(p.resolve()).replace(chr(92), '/')}'" for p in paths]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def normalize_clip(
    source: Path,
    dest: Path,
    duration: float,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> Path:
    """Crop a clip to the target frame and hold it for exactly `duration`.

    Clips shorter than the scene are looped rather than frozen, and the crop is
    centred so the subject stays in frame when a landscape source is squeezed
    into a vertical one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "-stream_loop", "-1",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-an",
            "-vf", (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},fps={fps},setsar=1"
            ),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(dest),
        ],
        label=f"normalize {source.name}",
    )
    return dest


def compose(
    plan,
    assets,
    audio_paths: list[Path] | None,
    out_path: str | Path,
    work_dir: str | Path,
    subtitles: str | Path | None = None,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    progress=None,
    single_audio: bool = False,
) -> Path:
    """Render scenes into a single mp4.

    `audio_paths` may be None, for sources that carry no narration; the result
    is then a silent video of exactly the planned length.
    """
    out_path = Path(out_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if len(plan.scenes) != len(assets):
        raise ValueError(
            f"scene/asset counts differ: {len(plan.scenes)}, {len(assets)}"
        )
    if (audio_paths is not None and not single_audio
            and len(audio_paths) != len(plan.scenes)):
        raise ValueError(
            f"scene/audio counts differ: {len(plan.scenes)}, {len(audio_paths)}"
        )

    clips = []
    for scene, asset in zip(plan.scenes, assets):
        if progress:
            progress(scene.index + 1, len(plan.scenes))
        clips.append(
            normalize_clip(
                Path(asset.path),
                work_dir / f"clip_{scene.index:04d}.mp4",
                scene.duration,
                width,
                height,
                fps,
            )
        )

    silent_video = work_dir / "video.mp4"
    run(
        ["-f", "concat", "-safe", "0",
         "-i", str(_write_concat_list(clips, work_dir / "clips.txt")),
         "-c", "copy", str(silent_video)],
        label="concat video",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []

    if audio_paths is None:
        args = ["-i", str(silent_video), "-an"]
    else:
        if single_audio:
            # Already one finished track (narration mixed with music).
            narration = Path(audio_paths[0])
        else:
            narration = work_dir / "narration.m4a"
            run(
                ["-f", "concat", "-safe", "0",
                 "-i", str(_write_concat_list([Path(p) for p in audio_paths],
                                              work_dir / "audio.txt")),
                 "-c:a", "aac", "-b:a", "192k", str(narration)],
                label="concat audio",
            )

        # Joining mp3s adds a little padding per file, so the narration ends up
        # marginally longer than the clips it was cut against. Trimming to the
        # shorter stream would clip the final words, so the last frame is held
        # instead until the audio finishes.
        video_length = probe_duration(silent_video)
        audio_length = probe_duration(narration)
        if audio_length > video_length:
            filters.append(f"tpad=stop_mode=clone:stop_duration={audio_length - video_length:.3f}")
        args = ["-i", str(silent_video), "-i", str(narration)]

    if subtitles:
        filters.append(f"ass='{escape_filter_path(subtitles)}'")

    if filters:
        args += ["-vf", ",".join(filters), "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    else:
        args += ["-c:v", "copy"]

    if audio_paths is not None:
        args += ["-c:a", "aac"]
    args.append(str(out_path))

    run(args, label="final mux")
    return out_path

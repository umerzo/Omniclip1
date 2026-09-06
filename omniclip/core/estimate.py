"""What a run is about to cost, in time, before it spends any of it.

Scene count is known the moment the shot grid exists, and clip generation is
the only part that takes real time -- so the whole wait is predictable from a
single number. Nothing reported it, which is how a forty minute source could
quietly become a three and a half hour job.

The per-clip figure is measured, not assumed. Across finished runs: 121s over
4 scenes, 98s over 7, 95s over 13, 110s over 25. It sits near 100 because
submissions are limited to roughly one a minute for the whole pool, so the
generation itself is mostly waiting for the next slot.
"""

from __future__ import annotations

# Measured across completed runs, not a guess. See the module docstring.
SECONDS_PER_CLIP = 100.0

# Stills run four at a time at about 8.4s each, so they cost roughly a quarter
# of that per scene. Narration, captions, music and the render together are a
# small and fairly flat addition.
SECONDS_PER_STILL = 8.4
STILL_WORKERS = 4
SECONDS_PER_NARRATION = 9.0
FIXED_OVERHEAD_SECONDS = 120.0


def estimate_run(scene_count: int, cfg: dict | None = None,
                 narrated: bool = True) -> dict:
    """How long this many scenes will take, and whether that is more than asked.

    Returned as plain numbers rather than a formatted string so an interface can
    present them however it likes, and so the job file stays machine-readable.
    """
    cfg = cfg or {}
    cap = int((cfg.get("shots") or {}).get("scene_cap", 40))

    clip_seconds = scene_count * SECONDS_PER_CLIP
    still_seconds = scene_count * SECONDS_PER_STILL / STILL_WORKERS
    speech_seconds = scene_count * SECONDS_PER_NARRATION if narrated else 0.0
    total = clip_seconds + still_seconds + speech_seconds + FIXED_OVERHEAD_SECONDS

    return {
        "est_scenes": scene_count,
        "est_clip_minutes": round(clip_seconds / 60, 1),
        "est_total_minutes": round(total / 60, 1),
        "est_scene_cap": cap,
        # True when the clip ceiling has stopped the budget from applying, which
        # is the case worth warning about: the run is longer than was asked for
        # and nothing else will say so.
        "est_over_budget": scene_count > cap,
    }


def describe(estimate: dict) -> str:
    """One line a person can act on."""
    minutes = estimate.get("est_total_minutes", 0)
    scenes = estimate.get("est_scenes", 0)
    if minutes >= 90:
        length = f"{minutes / 60:.1f} hours"
    else:
        length = f"{minutes:.0f} minutes"
    line = f"{scenes} scenes, about {length} of generation"
    if estimate.get("est_over_budget"):
        line += (f" -- more than the {estimate['est_scene_cap']}-scene budget. "
                 f"Trim the source or raise the cap to spend less.")
    return line

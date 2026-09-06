"""Provider Manager for intelligent video sourcing and failover."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .base import QuotaExhausted, VideoGenerationProvider, VisualAsset, VisualError


class ProviderManager:
    """Orchestrates generation providers with health tracking, prioritization, and failover."""

    def __init__(
        self,
        providers: list[VideoGenerationProvider],
        cache_dir: Path,
    ):
        self.providers = providers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_scene(
        self,
        query: str,
        duration: float,
        variant: int = 0,
        generation_mode: str = "image_to_video",
        still_path: str | None = None,
    ) -> VisualAsset:
        """Fetch or generate a visual asset trying providers in priority order."""
        last_error = None
        quota_hit = False

        for provider in self.providers:
            if not provider.available():
                continue
            try:
                asset = provider.fetch(
                    query=query,
                    min_duration=duration,
                    cache_dir=self.cache_dir,
                    variant=variant,
                    generation_mode=generation_mode,
                    still_path=still_path,
                )
                if asset is not None:
                    return asset
            except QuotaExhausted as exc:
                quota_hit = True
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue

        if quota_hit and not any(p.name in ("pexels", "pixabay") for p in self.providers):
            raise QuotaExhausted(f"All visual generation providers exhausted quota: {last_error}")
        raise VisualError(f"No provider could supply clip for {query[:60]!r}: {last_error}")

    def fetch_for_plan(
        self,
        scenes: list[Any],
        max_workers: int = 4,
        progress: Callable[[int, int], None] | None = None,
        on_asset: Callable[[Any, VisualAsset], None] | None = None,
    ) -> list[VisualAsset]:
        """Fetch clips for multiple scenes concurrently."""
        assets: dict[int, VisualAsset] = {}
        total = len(scenes)
        completed = 0
        lock = threading.Lock()

        def work(scene):
            dur = getattr(scene, "duration", 4.0)
            query = getattr(scene, "visual_query", getattr(scene, "visual_prompt", ""))
            variant = getattr(scene, "variant", getattr(scene, "index", 0))
            mode = getattr(scene, "generation_mode", "image_to_video")
            still = getattr(scene, "still_path", None)
            return scene, self.fetch_scene(
                query=query,
                duration=dur,
                variant=variant,
                generation_mode=mode,
                still_path=still,
            )

        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, total))) as pool:
            futures = {pool.submit(work, s): s for s in scenes}
            for future in as_completed(futures):
                scene, asset = future.result()
                with lock:
                    idx = getattr(scene, "index", getattr(scene, "scene_id", 0))
                    assets[idx] = asset
                    completed += 1
                    if on_asset:
                        on_asset(scene, asset)
                    if progress:
                        progress(completed, total)

        return [assets[getattr(s, "index", getattr(s, "scene_id", 0))] for s in scenes]

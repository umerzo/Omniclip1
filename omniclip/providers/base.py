"""Base interfaces and data structures for video generation providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class VisualError(RuntimeError):
    """Raised when a visual provider fails to supply a clip."""


class QuotaExhausted(VisualError):
    """Raised when an account or pool is out of capacity and waiting will not resolve it."""


@dataclass
class VisualAsset:
    path: Path
    source: str
    query: str
    width: int = 0
    height: int = 0
    duration: float = 0.0

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width


class VideoGenerationProvider(ABC):
    """Abstract interface for all video generation providers."""

    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """Check if provider is configured and currently reachable."""
        pass

    @abstractmethod
    def fetch(
        self,
        query: str,
        min_duration: float,
        cache_dir: Path,
        variant: int = 0,
        **kwargs,
    ) -> VisualAsset | None:
        """Generate or source a video clip satisfying query and duration."""
        pass

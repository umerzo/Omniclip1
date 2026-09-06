"""Settings loader.

Reads settings.toml and swaps any "${VAR}" placeholder for the matching
environment variable, so secrets stay in .env and out of the repo.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = PROJECT_ROOT / "omniclip" / "config" / "settings.toml"
DEFAULT_ENV = PROJECT_ROOT / ".env"

_PLACEHOLDER = re.compile(r"^\$\{(\w+)\}$")


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


def load_env(path: str | Path = DEFAULT_ENV) -> None:
    """Load KEY=VALUE lines from .env or Streamlit Cloud secrets without overwriting real env vars."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k, v in st.secrets.items():
                if isinstance(v, str):
                    os.environ.setdefault(k, v)
    except Exception:
        pass

    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _resolve(value):
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    if isinstance(value, str):
        match = _PLACEHOLDER.match(value)
        if match:
            return os.environ.get(match.group(1), "")
    return value


def load_settings(path: str | Path = DEFAULT_SETTINGS) -> dict:
    """Return settings with environment placeholders resolved."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"No settings file at {path}")
    load_env()
    with path.open("rb") as handle:
        return _resolve(tomllib.load(handle))


def resolve_path(value: str | Path) -> Path:
    """Turn a config path into an absolute one, relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

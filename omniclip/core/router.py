"""AI Model Router.

Dispatches prompts to Gemini 3.8 Flash as the primary high-reasoning intelligence,
with OpenRouter and Groq fallbacks, structured JSON output validation,
and automatic JSON repair.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import OpenAI

from ..utils.config import load_env, load_settings

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


def repair_json(raw: str) -> dict:
    """Extract and parse JSON from model output, handling fences, stray text, and truncation."""
    text = (raw or "").strip()
    match = _JSON_BLOCK.search(text)
    if match:
        text = match.group(1).strip()

    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try robust json_repair for truncated / unclosed JSON
    try:
        import json_repair
        repaired = json_repair.loads(text)
        if isinstance(repaired, (dict, list)):
            return repaired
    except Exception:
        pass

    # Extract outermost JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Clean trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Extract outermost JSON array
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        candidate = text[first_bracket : last_bracket + 1]
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON from output:\n{text[:400]}")


class AIModelRouter:
    """Manages AI model execution across high, medium, and fast tiers."""

    PRIMARY_MODEL = "google/gemini-2.5-flash"
    SECONDARY_MODEL = "minimax/minimax-m3"
    FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct"
    GROQ_MODEL = "openai/gpt-oss-120b"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
    ):
        load_env()
        cfg = load_settings()
        self.base_url = base_url or cfg["script"].get("base_url", "https://openrouter.ai/api/v1")
        self.api_key = api_key or cfg["script"].get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self.groq_key = os.environ.get("GROQ_API_KEY", "")

        self.primary_model = primary_model or self.PRIMARY_MODEL
        self.secondary_model = self.SECONDARY_MODEL
        self.fallback_model = fallback_model or cfg["script"].get("model", self.FALLBACK_MODEL)

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key or "missing")

    def run_prompt(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 12000,
        response_json: bool = True,
        high_reasoning: bool = True,
        max_retries: int = 3,
    ) -> dict | str:
        """Execute a prompt with retry and graceful tier failover."""
        chosen_model = model or (self.primary_model if high_reasoning else self.secondary_model)
        models_to_try = [chosen_model]
        if self.secondary_model not in models_to_try:
            models_to_try.append(self.secondary_model)
        if self.fallback_model not in models_to_try:
            models_to_try.append(self.fallback_model)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for m in models_to_try:
            for attempt in range(1, max_retries + 1):
                try:
                    kwargs: dict[str, Any] = {
                        "model": m,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if response_json:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = self.client.chat.completions.create(**kwargs)
                    raw_text = response.choices[0].message.content or ""

                    if response_json:
                        return repair_json(raw_text)
                    return raw_text

                except Exception as exc:
                    last_error = exc
                    # Retry on rate limit or transient network error
                    error_msg = str(exc).lower()
                    if "429" in error_msg or "rate limit" in error_msg or "timeout" in error_msg:
                        time.sleep(2.0 * attempt)
                        continue
                    # If response_format error on a model that doesn't support json_object mode
                    if "response_format" in error_msg:
                        try:
                            kwargs.pop("response_format", None)
                            response = self.client.chat.completions.create(**kwargs)
                            raw_text = response.choices[0].message.content or ""
                            if response_json:
                                return repair_json(raw_text)
                            return raw_text
                        except Exception as inner_exc:
                            last_error = inner_exc
                            break
                    break

        raise RuntimeError(f"AIModelRouter failed across models {models_to_try}: {last_error}") from last_error


# Singleton default router
_default_router: AIModelRouter | None = None


def get_router() -> AIModelRouter:
    global _default_router
    if _default_router is None:
        _default_router = AIModelRouter()
    return _default_router

"""Thin OpenRouter chat-completions client used by LLM-backed agents."""
from __future__ import annotations

from typing import List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from core.logging_config import get_logger

logger = get_logger(__name__)


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.base_url = settings.openrouter_base_url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def chat(self, messages: List[dict], temperature: float = 0.3, max_tokens: int = 600) -> str:
        if not self.api_key:
            logger.warning("openrouter_no_api_key", msg="Falling back to local stub response")
            return _stub_response(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-org/hermes-ads-research",
            "X-Title": "Hermes Ads Research",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            logger.error("openrouter_request_failed", error=str(exc))
            raise


def _stub_response(messages: List[dict]) -> str:
    """Deterministic offline fallback so the pipeline still runs without a key."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return f"[stub-llm] Acknowledged prompt of {len(last_user)} chars. No API key configured."


llm_client = OpenRouterClient()

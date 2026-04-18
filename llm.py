"""
novaprospect/core/llm.py
========================
Centralized LLM client for xAI Grok via the Responses API.

Uses the OpenAI Python SDK pointed at xAI's base URL. All agents call
through this module instead of rolling their own HTTP requests.

Three call modes:
    llm_call()           → standard single-agent reasoning (most agents)
    llm_vision_call()    → same model, with images attached
    llm_orchestrate()    → multi-agent orchestrator (reasoning agent only)

NOTES on Grok 4.x:
  - Does NOT accept `reasoning_effort` — sending it 4xx's.
  - Multimodal is unified: same model handles images. No separate vision model.
  - Reasoning eats tokens. Default max_tokens is 16000.
  - Reasoning calls can run minutes — default timeout is 600s.
  - Multi-agent mode is a model variant, not a different API surface.
"""

from __future__ import annotations

import time
import threading
from typing import Optional, List, Dict, Any

from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError

from config.settings import (
    GROK_API_KEY,
    GROK_API_BASE,
    ACTIVE_CONFIG,
    LLM_TIMEOUT_S,
)
from core.logger import get_logger

log = get_logger("llm")

_lock = threading.Lock()
_call_count = 0
_client: Optional[OpenAI] = None


# ── Client singleton ─────────────────────────────────────────────────────────

def _get_client() -> Optional[OpenAI]:
    """Lazy singleton — avoids constructing the client at import time."""
    global _client
    if _client is None:
        if not GROK_API_KEY:
            return None
        _client = OpenAI(
            api_key=GROK_API_KEY,
            base_url=GROK_API_BASE,
            timeout=LLM_TIMEOUT_S,
        )
    return _client


# ── Input builders ───────────────────────────────────────────────────────────

def _build_input(
    system: str,
    user: str,
    images: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Build the `input` array for the Responses API.

    images: optional list of {"label": str, "b64": str} dicts — base64-encoded
    PNGs to attach to the user turn for vision tasks.
    """
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]

    if images:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": user}]
        for img in images:
            label = img.get("label", "image").upper()
            content.append({"type": "input_text", "text": f"\n[{label}]"})
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img['b64']}",
            })
        msgs.append({"role": "user", "content": content})
    else:
        msgs.append({"role": "user", "content": user})

    return msgs


def _extract_text(response: Any) -> Optional[str]:
    """Pull assistant text from a Responses-API response object."""
    text = getattr(response, "output_text", None)
    if text:
        return text

    out_blocks = getattr(response, "output", None) or []
    chunks: List[str] = []
    for block in out_blocks:
        content = getattr(block, "content", None) or []
        for item in content:
            t = getattr(item, "text", None) or (
                item.get("text") if isinstance(item, dict) else None
            )
            if t:
                chunks.append(t)
    return "".join(chunks) if chunks else None


# ── Budget management ────────────────────────────────────────────────────────

def _check_budget() -> Optional[int]:
    """Increment and return call count, or None if budget exhausted."""
    global _call_count
    max_calls = ACTIVE_CONFIG.get("max_api_calls", 200)
    with _lock:
        if _call_count >= max_calls:
            log.warning("API budget exhausted", limit=max_calls)
            return None
        _call_count += 1
        return _call_count


def reset_budget() -> None:
    """Reset the per-run API call counter."""
    global _call_count
    with _lock:
        _call_count = 0


def get_call_count() -> int:
    with _lock:
        return _call_count


# ── Core API call ────────────────────────────────────────────────────────────

def _raw_call(
    model: str,
    msgs: List[Dict[str, Any]],
    max_tokens: int,
    retries: int = 2,
    call_n: int = 0,
) -> Optional[str]:
    """Make a Responses-API call. Returns text content or None."""

    client = _get_client()
    if client is None:
        log.error("GROK_API_KEY not set — cannot make LLM call")
        return None

    log.debug("LLM request", call_n=call_n, model=model, max_tokens=max_tokens)

    for attempt in range(1, retries + 2):
        try:
            t0 = time.time()
            response = client.responses.create(
                model=model,
                input=msgs,
                max_output_tokens=max_tokens,
                # NOTE: do NOT pass reasoning_effort — Grok 4.x rejects it.
            )
            elapsed = round(time.time() - t0, 2)

            text = _extract_text(response)
            if not text:
                log.warning("LLM returned empty text", model=model,
                            elapsed_s=elapsed, attempt=attempt)
                if attempt <= retries:
                    time.sleep(2 ** attempt)
                    continue
                return None

            # Log usage stats
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "input_tokens", None) if usage else None
            out_tok = getattr(usage, "output_tokens", None) if usage else None
            reason_tok = None
            if usage and hasattr(usage, "output_tokens_details"):
                details = usage.output_tokens_details
                reason_tok = getattr(details, "reasoning_tokens", None)

            log.info("LLM response", call_n=call_n, model=model,
                     elapsed_s=elapsed, tokens_in=in_tok, tokens_out=out_tok,
                     reasoning_tokens=reason_tok, text_len=len(text))
            return text

        except APITimeoutError:
            log.warning("LLM timeout", model=model, attempt=attempt,
                        timeout_s=LLM_TIMEOUT_S)
            if attempt <= retries:
                time.sleep(2 ** attempt)
                continue
            return None

        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            body = str(getattr(e, "message", e))[:300]
            log.warning("LLM API error", status=status, body=body, attempt=attempt)
            if status in (429, 500, 502, 503, 504) and attempt <= retries:
                time.sleep(2 ** attempt)
                continue
            return None

        except APIConnectionError as e:
            log.warning("LLM connection error", err=str(e)[:200], attempt=attempt)
            if attempt <= retries:
                time.sleep(2 ** attempt)
                continue
            return None

        except Exception as e:
            log.exception("LLM call failed", exc=e, attempt=attempt)
            return None

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — these are what agents call
# ══════════════════════════════════════════════════════════════════════════════

def llm_call(
    system: str,
    user: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    retries: int = 2,
) -> Optional[str]:
    """Standard single-agent reasoning call. Used by most agents.

    Equivalent to the old call_grok_structural() / similar per-agent functions,
    but centralized. Agents no longer need to import requests or manage HTTP.

    Args:
        system:     System prompt (agent role, instructions)
        user:       User prompt (the actual question / data)
        model:      Override model (defaults to ACTIVE_CONFIG["model"])
        max_tokens: Override max tokens (defaults to ACTIVE_CONFIG["max_tokens"])

    Returns:
        The model's text response, or None on failure.
    """
    call_n = _check_budget()
    if call_n is None:
        return None

    model = model or ACTIVE_CONFIG["model"]
    max_tokens = max_tokens or ACTIVE_CONFIG["max_tokens"]
    msgs = _build_input(system, user)

    return _raw_call(model, msgs, max_tokens, retries, call_n)


def llm_vision_call(
    system: str,
    user: str,
    images: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    retries: int = 2,
) -> Optional[str]:
    """Vision call — text + images. Used by vision/lidar agents.

    Args:
        images: list of {"label": "hillshade", "b64": "<base64 PNG>"}

    Grok 4.x is unified multimodal — uses the same model as text.
    """
    call_n = _check_budget()
    if call_n is None:
        return None

    model = model or ACTIVE_CONFIG["vision_model"]
    max_tokens = max_tokens or ACTIVE_CONFIG["max_tokens"]
    msgs = _build_input(system, user, images)

    return _raw_call(model, msgs, max_tokens, retries, call_n)


def llm_orchestrate(
    system: str,
    user: str,
    max_tokens: Optional[int] = None,
    retries: int = 2,
) -> Optional[str]:
    """Multi-agent orchestrator call. Used ONLY by the top-level reasoning agent.

    This uses grok-4.20-multi-agent-beta-0309 which spawns multiple internal
    agents that research, cross-check, debate, and synthesize a final answer.
    More expensive and slower, but dramatically better for the final
    prospectivity ranking where synthesis quality matters most.

    Same Responses API endpoint — multi-agent is a model variant, not a
    different API surface.
    """
    call_n = _check_budget()
    if call_n is None:
        return None

    model = ACTIVE_CONFIG["orchestrator_model"]
    max_tokens = max_tokens or ACTIVE_CONFIG["max_tokens"]
    msgs = _build_input(system, user)

    return _raw_call(model, msgs, max_tokens, retries, call_n)

"""Minimal, synthetic grounded retrieval demonstration for Miranda.

No garment-care claims are inferred from images. The path-based facts below are
DEMO DATA ONLY. Never use them as real laundry guidance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch


import json
from urllib import request, error


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def call_ollama_chat_json(prompt: str, model: str, timeout: int = 120) -> dict:
    """Call the local Ollama chat API; parse content, never thinking.

    The MLX-backed Ollama runner may return HTTP 501 for /api/generate
    while supporting /api/chat. No proxy is used for localhost traffic.
    """
    payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "stream": False,
    "format": "json",
    "think": False,
    }
    
    req = request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            outer = json.load(response)
    except error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise GroundingError(
            f"Ollama chat HTTP {exc.code}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise GroundingError(f"Cannot reach Ollama chat API: {exc.reason}") from exc

    message = outer.get("message")
    if not isinstance(message, dict):
        raise GroundingError("Ollama chat response has no message object")
    raw = message.get("content")
    if not isinstance(raw, str) or not raw.strip():
        raise GroundingError("Ollama chat returned empty message.content")
    raw = raw.strip()
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GroundingError(
            f"Ollama chat returned invalid JSON: {raw[:300]!r}"
        ) from exc
    if not isinstance(data, dict):
        raise GroundingError("Ollama chat JSON is not an object")
    return data


PROMPT_TEMPLATE = '''You are Miranda's conservative fashion retrieval intent parser.
The catalog supports similarity search and exactly ONE verifiable hard constraint:
synthetic tumble_dry_allowed (true or false). No other factual metadata exists.

Return ONLY a JSON object with EXACTLY these fields:
{{"domain":"fashion_search|out_of_scope", "query_en":"English visual garment description", "is_followup":false, "decision":"SEARCH|UNSUPPORTED|UNCERTAIN", "tumble_dry":"REQUIRED|FORBIDDEN|NONE", "reason":"short explanation"}}

Rules:
- SEARCH only when every user-imposed condition can be honored. Normal positive
  visual descriptions (color, garment type, pattern) are similarity hints, NOT guarantees.
- REQUIRED means the garment MUST be allowed to tumble dry. FORBIDDEN means
  the garment MUST NOT be tumble dried. NONE means no mandatory dryer requirement.
- Mere statements about habits ("I never use a dryer") do NOT impose garment requirements.
- Preferences about dryer suitability ("I'd prefer tumble-dryable") are UNSUPPORTED:
  this prototype cannot rank a soft care preference. Do not turn them into hard constraints.
- UNSUPPORTED for any negative visual exclusion ("not black", "without stripes"),
  additional factual requirement ("organic", "machine washable", "waterproof", "in stock"),
  or request to guarantee a visual property. Do not silently drop a condition.
- UNCERTAIN when intent, polarity, or required eligibility is ambiguous.
- For SEARCH with REQUIRED/FORBIDDEN, REMOVE dryer language from query_en.
- For SEARCH without dryer requirement, query_en is a short positive visual description.
- If the request is not about fashion retrieval, use out_of_scope.
- For a short follow-up, incorporate the previous visual description if appropriate;
  never assume a previous hard constraint still holds without explicit confirmation.
- Translate Chinese/Japanese requests into English visual queries.

Examples:
"grey zip sweater" => SEARCH, NONE, "grey zip sweater"
"grey zip sweater, must be tumble-dryable" => SEARCH, REQUIRED, "grey zip sweater"
"grey zip sweater, must never be tumble-dried" => SEARCH, FORBIDDEN, "grey zip sweater"
"I never use a dryer. Find a sweater" => SEARCH, NONE, "sweater"
"I'd prefer a tumble-dryable sweater" => UNSUPPORTED
"black shirt without stripes" => UNSUPPORTED
"a shirt that is NOT black" => UNSUPPORTED
"machine washable cardigan" => UNSUPPORTED
"找灰色有拉鍊的毛衣，但一定要可以烘乾" => SEARCH, REQUIRED, "grey zip sweater"
"write C++ hello world" => out_of_scope
"T-shirt with C++ code printed on it" => SEARCH, NONE, "T-shirt with C++ code print"

PREVIOUS VISUAL QUERY: {previous}
CURRENT REQUEST: {message}
Return JSON only.'''


class GroundingError(ValueError):
    """Fail closed when an intent response cannot be trusted."""


def interpret(message: str, previous_query: str | None,
              call_json: Callable[[str], dict]) -> dict:
    """Return validated intent; call_json is an injected Ollama JSON adapter."""
    prompt = PROMPT_TEMPLATE.format(previous=previous_query or 'NONE', message=message)
    data = call_json(prompt)
    if not isinstance(data, dict):
        raise GroundingError('Intent parser returned a non-object')
    required = {'domain', 'query_en', 'is_followup', 'decision', 'tumble_dry', 'reason'}
    if not required.issubset(data):
        raise GroundingError('Intent parser omitted required fields')
    if data['domain'] not in {'fashion_search', 'out_of_scope'}:
        raise GroundingError('Invalid domain')
    if data['decision'] not in {'SEARCH', 'UNSUPPORTED', 'UNCERTAIN'}:
        raise GroundingError('Invalid decision')
    if data['tumble_dry'] not in {'REQUIRED', 'FORBIDDEN', 'NONE'}:
        raise GroundingError('Invalid tumble_dry state')
    if type(data['is_followup']) is not bool:
        raise GroundingError('Invalid follow-up flag')
    if not isinstance(data['query_en'], str) or not isinstance(data['reason'], str):
        raise GroundingError('Invalid text fields')
    if data['domain'] == 'fashion_search' and data['decision'] == 'SEARCH':
        if not data['query_en'].strip():
            raise GroundingError('Missing retrieval query')
        # Avoid CLIP treating care requirements as visual similarity features.
        # This is a validation guard, not a semantic classifier.
        if data['tumble_dry'] != 'NONE' and any(
            token in data['query_en'].lower()
            for token in ('tumble', 'dryer', 'dryable', 'dry-able')
        ):
            raise GroundingError('Care constraint leaked into visual query')
    return data


def synthetic_tumble_dry_allowed(path: str) -> bool:
    """DEMO ONLY: filenames containing 'wool' or 'cardigan' are False."""
    name = str(path).lower()
    return not ('wool' in name or 'cardigan' in name)


def apply_tumble_dry_mask(scores: torch.Tensor, paths: list[str], state: str):
    """Apply an exact eligibility filter BEFORE Top-K; return scores and counts.

    Missing/unknown values would be ineligible for either constraint; this
    synthetic demo currently assigns every catalog entry a Boolean value.
    """
    if state not in {'REQUIRED', 'FORBIDDEN', 'NONE'}:
        raise GroundingError('Unknown hard constraint')
    if state == 'NONE':
        return scores, None
    if scores.ndim != 2 or scores.shape[0] != 1 or scores.shape[1] != len(paths):
        raise GroundingError('Score and path shapes differ')
    allowed = torch.tensor(
        [synthetic_tumble_dry_allowed(p) for p in paths],
        dtype=torch.bool, device=scores.device,
    ).unsqueeze(0)
    eligible = allowed if state == 'REQUIRED' else ~allowed
    filtered = scores.masked_fill(~eligible, float('-inf'))
    return filtered, {
        'eligible': int(eligible.sum().item()),
        'rejected': int((~eligible).sum().item()),
        'rule': 'SYNTHETIC DEMO ONLY: wool/cardigan => tumble_dry_allowed=False',
    }

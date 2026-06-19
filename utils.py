"""
Output parsing utilities. parse_sentences() cleans raw model responses
before they are passed to downstream steps.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def parse_sentences(raw: str, expected: int = 5) -> tuple[list[str], list[str]]:
    """
    Extract sentences from a model response.
    Tries JSON first ({"sentences": [...]}), then falls back to line-by-line parsing.
    """
    def _try_json(text: str) -> list[str] | None:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "sentences" in data:
                return [str(s).strip() for s in data["sentences"] if str(s).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    items = _try_json(raw)
    if items is None:
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence_match:
            items = _try_json(fence_match.group(1))
    if items is None:
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            items = _try_json(brace_match.group(0))

    if items is not None:
        sentences = items[:expected]
        warnings = []
        if len(items) != expected:
            warnings.append(f"JSON response had {len(items)} sentences; expected {expected}.")
            logger.warning(warnings[0])
        return sentences, warnings

    # Plain-text fallback
    lines = raw.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        line = line.strip()
        if not line:
            continue
        if not line[0].isupper():
            logger.debug("Dropped non-sentence line: %r", line)
            continue
        if not re.search(r"[a-zA-Z]", line):
            continue
        cleaned.append(line)

    sentences = cleaned[:expected]

    warnings = []
    if len(cleaned) > expected:
        warnings.append(f"Model returned {len(cleaned)} lines; keeping first {expected}.")
    elif len(cleaned) < expected:
        warnings.append(f"Model returned only {len(cleaned)} valid lines; expected {expected}.")

    if warnings:
        for w in warnings:
            logger.warning(w)

    return sentences, warnings

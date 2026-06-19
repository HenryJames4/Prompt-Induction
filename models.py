"""
OpenRouter API interface. All model calls go through generate_text().
API key is read from the OPENROUTER_API_KEY environment variable (set in .env).
"""

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

import config

load_dotenv()

logger = logging.getLogger(__name__)

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    return OpenAI(api_key=api_key, base_url=config.OPENROUTER_BASE_URL)


def generate_text(
    model_name: str,
    prompt: str,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    extra_body: dict | None = None,
    response_format: dict | None = None,
) -> str:
    """Call a model via OpenRouter and return the stripped response string."""
    client = _get_client()
    logger.debug(
        "Calling model=%s  temp=%.2f  top_p=%s  top_k=%s  min_p=%s  prompt_len=%d",
        model_name, temperature, top_p, top_k, min_p, len(prompt),
    )

    # top_k and min_p are non-standard; merge them into extra_body
    merged_extra: dict = dict(extra_body or {})
    if top_k is not None:
        merged_extra["top_k"] = top_k
    if min_p is not None:
        merged_extra["min_p"] = min_p

    kwargs: dict = dict(
        model=model_name,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        extra_body=merged_extra,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)

    text = response.choices[0].message.content
    if text is None:
        raise ValueError(f"Model {model_name} returned an empty response.")

    logger.debug("Response length: %d chars", len(text))
    return text.strip()

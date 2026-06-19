"""
Core experiment logic: generate_l1, run_induction, compute_run_metrics.
Each function corresponds to one stage of the L1 → induction → L2 pipeline.
"""

import logging
from datetime import datetime, timezone

import config
from models import generate_text
from prompts import (
    build_l1_prompt, build_induction_prompt, build_l2_prompt,
    build_l1_prompt_neg_evidence, build_induction_prompt_neg_evidence,
)
from utils import parse_sentences

VARIANTS = (None, "neg_evidence")
logger = logging.getLogger(__name__)


def generate_l1(model_key: str, template: str, variant: str | None = None) -> dict:
    model_id    = config.MODELS[model_key]
    model_type  = "LRM" if model_key in config.LRM_MODELS else "LLM"
    extra_body  = config.MODEL_EXTRA_PARAMS.get(model_key, {})
    sampling    = config.MODEL_SAMPLING_PARAMS.get(model_key, {"temperature": 1.0})
    resp_fmt    = {"type": "json_object"} if model_type == "LLM" else None

    result = {
        "model_key":         model_key,
        "model_id":          model_id,
        "model_type":        model_type,
        "template":          template,
        "variant":           variant,
        "l1_raw":            None,
        "l1_sentences":      [],
        "l1_parse_warnings": [],
        "timestamp":         _now(),
        "error":             None,
    }

    logger.info("[L1] model=%s (%s)  template=%s  variant=%s", model_key, model_type, template, variant)

    try:
        if variant == "neg_evidence":
            prompt = build_l1_prompt_neg_evidence(template)
        else:
            prompt = build_l1_prompt(template)

        raw = generate_text(model_id, prompt, **sampling, extra_body=extra_body,
                            response_format=resp_fmt)
        result["l1_raw"] = raw

        sentences, warnings = parse_sentences(raw, expected=config.NUM_SENTENCES)
        result["l1_sentences"]      = sentences
        result["l1_parse_warnings"] = warnings

        # For neg_evidence: first 4 sentences are correct, last 1 is incorrect.
        if variant == "neg_evidence" and len(sentences) >= 4:
            result["l1_correct_sentences"] = sentences[:4]
            result["l1_neg_sentences"]     = sentences[4:5]

        if not sentences:
            logger.warning("[L1] No sentences parsed  model=%s  template=%s", model_key, template)

    except Exception as exc:
        logger.error("[L1] FAILED  model=%s  template=%s: %s", model_key, template, exc)
        result["error"] = str(exc)

    return result


def run_induction(model_key: str, template: str, l1_sentences: list[str],
                  variant: str | None = None, l1_correct_sentences: list[str] | None = None) -> dict:
    model_id    = config.MODELS[model_key]
    model_type  = "LRM" if model_key in config.LRM_MODELS else "LLM"
    extra_body  = config.MODEL_EXTRA_PARAMS.get(model_key, {})
    sampling    = config.MODEL_SAMPLING_PARAMS.get(model_key, {"temperature": 1.0})
    resp_fmt    = {"type": "json_object"} if model_type == "LLM" else None

    result = {
        "model_key":              model_key,
        "model_id":               model_id,
        "model_type":             model_type,
        "template":               template,
        "variant":                variant,
        "induction_input":        [],
        "induced_prompt_raw":     None,
        "induced_prompt":         None,
        "induced_prompt_chars":   None,
        "induced_prompt_words":   None,
        "l1_char_total":          None,
        "prompt_shorter_than_l1": None,
        "l2_raw":                 None,
        "l2_sentences":           [],
        "l2_parse_warnings":      [],
        "timestamp":              _now(),
        "error":                  None,
    }

    if not l1_sentences:
        result["error"] = "No L1 sentences available; skipping induction."
        logger.warning(result["error"])
        return result

    induction_input           = l1_sentences[:config.NUM_SENTENCES]
    result["induction_input"] = induction_input

    logger.info("[INDUCTION] model=%s (%s)  template=%s  variant=%s", model_key, model_type, template, variant)

    try:
        if variant == "neg_evidence":
            induction_prompt = build_induction_prompt_neg_evidence(induction_input)
        else:
            induction_prompt = build_induction_prompt(induction_input)

        if variant == "neg_evidence" and l1_correct_sentences:
            result["induction_input_correct"] = l1_correct_sentences

        induced_raw                    = generate_text(model_id, induction_prompt, **sampling, extra_body=extra_body)
        result["induced_prompt_raw"]   = induced_raw
        induced_prompt                 = induced_raw.strip()
        result["induced_prompt"]       = induced_prompt
        result["induced_prompt_chars"] = len(induced_prompt)
        result["induced_prompt_words"] = len(induced_prompt.split())

        l1_char_total                    = sum(len(s) for s in induction_input)
        result["l1_char_total"]          = l1_char_total
        result["prompt_shorter_than_l1"] = len(induced_prompt) < l1_char_total

        logger.info(
            "[INDUCTION] induced prompt (%d chars, %d words): %r",
            len(induced_prompt), len(induced_prompt.split()), induced_prompt[:120],
        )

        l2_raw                      = generate_text(model_id, build_l2_prompt(induced_prompt), **sampling, extra_body=extra_body,
                                                    response_format=resp_fmt)
        result["l2_raw"]            = l2_raw
        l2_sentences, l2_warnings   = parse_sentences(l2_raw, expected=config.NUM_SENTENCES)
        result["l2_sentences"]      = l2_sentences
        result["l2_parse_warnings"] = l2_warnings

    except Exception as exc:
        logger.error("[INDUCTION] FAILED  model=%s  template=%s: %s", model_key, template, exc)
        result["error"] = str(exc)

    return result


def compute_run_metrics(l1_result: dict, induction_result: dict) -> dict:
    sim = induction_result.get("similarity", {})
    return {
        "model_key":              l1_result["model_key"],
        "model_type":             l1_result.get("model_type", ""),
        "template":               l1_result["template"],
        "run_idx":                induction_result.get("run_idx"),
        "l1_sentences_count":     len(l1_result.get("l1_sentences", [])),
        "induced_prompt_chars":   induction_result.get("induced_prompt_chars"),
        "induced_prompt_words":   induction_result.get("induced_prompt_words"),
        "prompt_shorter_than_l1": induction_result.get("prompt_shorter_than_l1"),
        "l2_sentences_count":     len(induction_result.get("l2_sentences", [])),
        "mean_sim":               sim.get("mean_sim"),
        "mean_max_sim":           sim.get("mean_max_sim"),
        "l1_error":               l1_result.get("error"),
        "induction_error":        induction_result.get("error"),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

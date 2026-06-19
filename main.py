"""
Prompt induction experiment — entry point (LRM vs LLM).

Usage:
    python main.py                              # all models, all templates
    python main.py --template center_embedding
    python main.py --model deepseek-r1 kimi-k2
    python main.py --workers 3
    python main.py --variant neg_evidence       # saves to results_neg_evidence/

Resume: re-running after a crash picks up where it left off (existing run files are skipped).
Cancel a specific model mid-run: touch results/<model_key>/CANCEL
"""

import json
import logging
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config
from experiment import generate_l1, run_induction, compute_run_metrics

VARIANT_RESULTS_DIR = {
    "neg_evidence": "results_neg_evidence",
}

os.makedirs(config.RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

_save_lock = threading.Lock()


def _load_raw_runs() -> list[dict]:
    """Load all saved runs, deduplicating by (model_key, template, run_idx)."""
    import glob
    seen: dict[tuple, dict] = {}
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "*", "*", "run_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            key = (r.get("model_key"), r.get("template"), r.get("run_idx"))
            seen[key] = r
        except Exception:
            pass
    return list(seen.values())


def _save_raw_runs(runs: list[dict]) -> None:
    # Re-scan before writing so parallel processes don't overwrite each other.
    merged = {
        (r.get("model_key"), r.get("template"), r.get("run_idx")): r
        for r in _load_raw_runs()
    }
    for r in runs:
        key = (r.get("model_key"), r.get("template"), r.get("run_idx"))
        merged[key] = r
    with open(config.RAW_RUNS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(merged.values()), f, ensure_ascii=False, indent=2)


def _save_semantic_matrices(runs: list[dict]) -> None:
    matrices = [
        {
            "model_key":    r.get("model_key"),
            "model_type":   r.get("model_type"),
            "template":     r.get("template"),
            "matrix":       r.get("similarity", {}).get("matrix", []),
            "l1_sentences": r.get("l1_sentences", []),
            "l2_sentences": r.get("l2_sentences", []),
        }
        for r in runs if r.get("similarity")
    ]
    with open(config.SEMANTIC_MATRICES_PATH, "w", encoding="utf-8") as f:
        json.dump(matrices, f, ensure_ascii=False, indent=2)


def _save_summary_csv(metrics_rows: list[dict]) -> None:
    pd.DataFrame(metrics_rows).to_csv(config.SUMMARY_CSV_PATH, index=False)


def _cancel_path(model_key: str) -> str:
    return os.path.join(config.RESULTS_DIR, model_key, "CANCEL")


def _is_cancelled(model_key: str) -> bool:
    return os.path.exists(_cancel_path(model_key))


def _run_model(
    model_key: str,
    template: str,
    run_idx: int,
    all_results: list[dict],
    variant: str | None = None,
) -> dict | None:
    """Run L1 → induction → L2 for one model/template/run triple."""
    run_path = config.model_run_path(model_key, template, run_idx)

    if os.path.exists(run_path):
        logger.info("⏭ SKIP   model=%s  template=%s  run=%02d  (already saved)", model_key, template, run_idx)
        try:
            with open(run_path, encoding="utf-8") as f:
                existing = json.load(f)
            with _save_lock:
                all_results.append(existing)
            return existing
        except Exception:
            logger.warning("Could not reload %s — will re-run", run_path)

    if _is_cancelled(model_key):
        logger.warning("⛔ CANCEL model=%s  template=%s  run=%02d", model_key, template, run_idx)
        return None

    model_type = "LRM" if model_key in config.LRM_MODELS else "LLM"
    logger.info("▶ START  model=%s (%s)  template=%s  run=%02d", model_key, model_type, template, run_idx)

    l1 = generate_l1(model_key, template, variant=variant)

    if l1["error"]:
        logger.error("✗ L1 failed  model=%s  template=%s  run=%02d: %s", model_key, template, run_idx, l1["error"])
        return None

    logger.info("✓ L1 done  model=%s  run=%02d  sentences=%d", model_key, run_idx, len(l1["l1_sentences"]))

    if _is_cancelled(model_key):
        logger.warning("⛔ CANCEL model=%s  template=%s  run=%02d  (after L1)", model_key, template, run_idx)
        return None

    ind = run_induction(
        model_key, template, l1["l1_sentences"],
        variant=variant,
        l1_correct_sentences=l1.get("l1_correct_sentences"),
    )

    ind["l1_sentences"] = l1["l1_sentences"]
    ind["l1_raw"]       = l1["l1_raw"]
    ind["run_idx"]      = run_idx

    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(ind, f, ensure_ascii=False, indent=2)

    with _save_lock:
        all_results.append(ind)
        _save_raw_runs(all_results)

    logger.info(
        "✓ DONE   model=%s  run=%02d  mean_max=%.3f",
        model_key, run_idx,
        ind.get("similarity", {}).get("mean_max_sim", 0),
    )
    return ind


def run_experiment(
    filter_models: list[str] | None = None,
    filter_templates: list[str] | None = None,
    max_workers: int | None = None,
    variant: str | None = None,
) -> None:
    models_to_run    = filter_models    or list(config.MODELS.keys())
    templates_to_run = filter_templates or config.TEMPLATES
    num_runs         = config.NUM_RUNS

    lrms = [m for m in models_to_run if m in config.LRM_MODELS]
    llms = [m for m in models_to_run if m in config.LLM_MODELS]
    total_tasks = len(models_to_run) * len(templates_to_run) * num_runs
    workers = max_workers or len(models_to_run)

    logger.info("=" * 60)
    logger.info("Prompt Induction Experiment  (LRM vs LLM)")
    logger.info("Variant    : %s", variant or "baseline")
    logger.info("LRMs       : %s", lrms)
    logger.info("LLMs       : %s", llms)
    logger.info("Templates  : %s", templates_to_run)
    logger.info("Runs/model : %d", num_runs)
    logger.info("Workers    : %d", workers)
    logger.info("Total tasks: %d", total_tasks)
    logger.info("=" * 60)

    all_results: list[dict] = _load_raw_runs()
    metrics_rows: list[dict] = []

    for template in templates_to_run:
        logger.info("── Template: %s ──", template)

        futures_map = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Interleave submissions so all models get workers even when some are slow.
            for run_idx in range(1, num_runs + 1):
                for model_key in models_to_run:
                    fut = pool.submit(_run_model, model_key, template, run_idx, all_results, variant)
                    futures_map[fut] = (model_key, run_idx)

            for fut in as_completed(futures_map):
                model_key, run_idx = futures_map[fut]
                try:
                    result = fut.result()
                    if result is not None:
                        l1_stub = {
                            "model_key":    result["model_key"],
                            "model_type":   result["model_type"],
                            "template":     result["template"],
                            "l1_sentences": result.get("l1_sentences", []),
                            "error":        None,
                        }
                        metrics_rows.append(compute_run_metrics(l1_stub, result))
                except Exception as exc:
                    logger.error("✗ Worker exception  model=%s  run=%02d: %s", model_key, run_idx, exc)

    _save_semantic_matrices(all_results)
    _save_summary_csv(metrics_rows)

    logger.info("=" * 60)
    logger.info("Experiment complete. %d runs saved.", len(all_results))
    logger.info("  per-model → %s/<model>/<template>.json", config.RESULTS_DIR)
    logger.info("  raw runs  → %s", config.RAW_RUNS_PATH)
    logger.info("  summary   → %s", config.SUMMARY_CSV_PATH)
    logger.info("  matrices  → %s", config.SEMANTIC_MATRICES_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompt induction: LRM vs LLM.")
    parser.add_argument("--model",       nargs="+", choices=list(config.MODELS.keys()), default=None)
    parser.add_argument("--template",    nargs="+", choices=config.TEMPLATES,           default=None)
    parser.add_argument("--workers",     type=int,  default=None,
                        help="Max parallel threads. Defaults to number of models.")
    parser.add_argument("--results-dir", type=str,  default=None,
                        help="Override output directory.")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["neg_evidence"],
                        help="Prompt variant. Results saved to results_<variant>/ by default.")
    args = parser.parse_args()

    rdir = args.results_dir or (VARIANT_RESULTS_DIR[args.variant] if args.variant else None)
    if rdir:
        config.RESULTS_DIR            = rdir
        config.RAW_RUNS_PATH          = os.path.join(rdir, "raw_runs.json")
        config.SUMMARY_CSV_PATH       = os.path.join(rdir, "summary.csv")
        config.SEMANTIC_MATRICES_PATH = os.path.join(rdir, "semantic_matrices.json")
        config.LOG_PATH               = os.path.join(rdir, "experiment.log")
        os.makedirs(rdir, exist_ok=True)
        logger.info("Output redirected → %s", rdir)

    run_experiment(
        filter_models=args.model,
        filter_templates=args.template,
        max_workers=args.workers,
        variant=args.variant,
    )

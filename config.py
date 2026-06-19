"""
Experiment configuration: models, sampling params, templates, output paths.
LRM = reasoning model (chain-of-thought); LLM = standard instruction-tuned model.
"""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

RANDOM_SEED = 42

MODELS = {
    # LRMs
    "deepseek-r1":              "deepseek/deepseek-r1",
    "qwen3-30b-a3b-think":      "qwen/qwen3-30b-a3b-thinking-2507",
    "glm-4.5-think":            "z-ai/glm-4.5",
    "claude-sonnet-4.6-think":  "anthropic/claude-sonnet-4.6",
    "gemini-3.5-flash-think":   "google/gemini-3.5-flash",
    # LLMs
    "deepseek-v3":              "deepseek/deepseek-chat",
    "qwen3-30b-a3b":            "qwen/qwen3-30b-a3b-instruct-2507",
    "glm-4.5":                  "z-ai/glm-4.5",
    "claude-sonnet-4.6":        "anthropic/claude-sonnet-4.6",
    "gemini-3.5-flash":         "google/gemini-3.5-flash",
}

LRM_MODELS = {"deepseek-r1", "qwen3-30b-a3b-think", "glm-4.5-think", "claude-sonnet-4.6-think", "gemini-3.5-flash-think"}
LLM_MODELS = {"deepseek-v3", "qwen3-30b-a3b", "glm-4.5", "claude-sonnet-4.6", "gemini-3.5-flash"}

MODEL_SAMPLING_PARAMS: dict[str, dict] = {
    "deepseek-r1":             {"temperature": 0.6, "top_p": 0.95},
    "qwen3-30b-a3b-think":     {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    "glm-4.5-think":           {"temperature": 0.6, "top_p": 0.95},
    "claude-sonnet-4.6-think": {"temperature": 1.0, "top_p": 0.95},
    "gemini-3.5-flash-think":  {"temperature": 1.0, "top_p": 0.95},
    "deepseek-v3":             {"temperature": 1.0, "top_p": 0.95},
    "qwen3-30b-a3b":           {"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "min_p": 0},
    "glm-4.5":                 {"temperature": 0.6, "top_p": 0.95},
    "claude-sonnet-4.6":       {"temperature": 1.0, "top_p": 0.95},
    "gemini-3.5-flash":        {"temperature": 1.0, "top_p": 0.95},
}

# Non-standard fields forwarded verbatim to OpenRouter (reasoning control).
MODEL_EXTRA_PARAMS: dict[str, dict] = {
    "glm-4.5-think":           {"reasoning": {"enabled": True}},
    "glm-4.5":                 {"reasoning": {"enabled": False}},
    "claude-sonnet-4.6-think": {"reasoning": {"effort": "high"}},
    "claude-sonnet-4.6":       {"reasoning": {"effort": "none"}},
    "gemini-3.5-flash-think":  {"reasoning": {"effort": "high"}},
    "gemini-3.5-flash":        {"reasoning": {"effort": "minimal"}},
}

TEMPLATES = [
    "center_embedding",
    "center_embedding_depth2",
]

NUM_SENTENCES = 5
NUM_RUNS      = 30

RESULTS_DIR            = "results"
RAW_RUNS_PATH          = "results/raw_runs.json"
SUMMARY_CSV_PATH       = "results/summary.csv"
SEMANTIC_MATRICES_PATH = "results/semantic_matrices.json"
LOG_PATH               = "results/experiment.log"


def model_run_path(model_key: str, template: str, run_idx: int) -> str:
    """Path: results/<model_key>/<template>/run_01.json"""
    import os
    return os.path.join(RESULTS_DIR, model_key, template, f"run_{run_idx:02d}.json")

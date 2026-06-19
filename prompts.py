"""
Prompt builders for the three experiment stages.

Baseline:     build_l1_prompt, build_induction_prompt, build_l2_prompt
Neg-evidence: build_l1_prompt_neg_evidence, build_induction_prompt_neg_evidence

Templates:
  center_embedding        — 4 noun phrases + 4 verbs (depth 4)
  center_embedding_depth2 — 2 noun phrases + 2 verbs (depth 2)
"""

_FORMAT_RULES = """\
Return a JSON object with a single key "sentences" containing an array of exactly 5 strings.
Example: {"sentences": ["Sentence one.", "Sentence two.", "Sentence three.", "Sentence four.", "Sentence five."]}
Output only valid JSON. No extra text.
"""


_L1_INSTRUCTIONS = {
    "center_embedding": (
        "Each sentence must match this center-embedded pattern:\n"
        "Example: the report the editor the assistant the intern consulted revised approved disappeared\n"
        "Each sentence has exactly 4 noun phrases followed by exactly 4 verbs.\n"
        "NOUN must be a singular common noun. VERB must be a single-word verb.\n"
    ),
    "center_embedding_depth2": (
        "Each sentence must match this center-embedded pattern:\n"
        "Example: the proposal the reviewer approved passed\n"
        "Each sentence has exactly 2 noun phrases followed by exactly 2 verbs.\n"
        "NOUN must be a singular common noun. VERB must be a single-word verb.\n"
    ),
}


def build_l1_prompt(template: str) -> str:
    if template not in _L1_INSTRUCTIONS:
        raise ValueError(f"Unknown template: {template!r}")
    return f"{_L1_INSTRUCTIONS[template]}\n\n{_FORMAT_RULES}"


def build_induction_prompt(sentences: list[str]) -> str:
    sentence_block = "\n".join(sentences)
    return (
        "You are given a list of sentences. "
        "Infer a short prompt that could generate the syntactic structure of these sentences. "
        "The prompt must be shorter than the original list. "
        "Do not copy the sentences. "
        "Return only the prompt.\n\n"
        f"Sentences:\n{sentence_block}"
    )


def build_l2_prompt(induced_prompt: str) -> str:
    return (
        "Use the following prompt to generate exactly 5 sentences. "
        "Follow the prompt's instructions precisely.\n\n"
        f"Prompt: {induced_prompt}\n\n"
        f"{_FORMAT_RULES}"
    )


_L1_NEG_INSTRUCTIONS = {
    "center_embedding": (
        "Generate 5 sentences with center-embedded structure.\n"
        "The first 4 must be CORRECT and match this pattern:\n"
        "Example: the report the editor the assistant the intern consulted revised approved disappeared\n"
        "Each sentence has exactly 4 noun phrases followed by exactly 4 verbs.\n"
        "The last sentence must be INCORRECT:\n"
        "  - Sentence 5: wrong number of phrase-verb pairs, using only 3 NOUN phrases and 3 VERBs.\n"
        "Return the sentences in order: 4 correct first, then 1 incorrect.\n\n"
    ),
    "center_embedding_depth2": (
        "Generate 5 sentences with center-embedded structure.\n"
        "The first 4 must be CORRECT and match this pattern:\n"
        "Example: the proposal the reviewer approved passed\n"
        "Each sentence has exactly 2 noun phrases followed by exactly 2 verbs.\n"
        "The last sentence must be INCORRECT:\n"
        "  - Sentence 5: wrong number of phrase-verb pairs, using 3 NOUN phrases and 3 VERBs.\n"
        "Return the sentences in order: 4 correct first, then 1 incorrect.\n\n"
    ),
}


def build_l1_prompt_neg_evidence(template: str) -> str:
    if template not in _L1_NEG_INSTRUCTIONS:
        raise ValueError(f"Unknown template for neg_evidence: {template!r}")
    return f"{_L1_NEG_INSTRUCTIONS[template]}\n\n{_FORMAT_RULES}"


def build_induction_prompt_neg_evidence(sentences: list[str]) -> str:
    """First 4 sentences are labelled CORRECT, last 1 is labelled INCORRECT."""
    correct = sentences[:4]
    neg = sentences[4:5]
    correct_block = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(correct))
    neg_block = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(neg))
    return (
        "You are given labelled examples of a linguistic structure.\n\n"
        f"CORRECT examples:\n{correct_block}\n\n"
        f"INCORRECT examples (showing what to avoid):\n{neg_block}\n\n"
        "Infer a short prompt that would generate sentences like the correct examples "
        "and explicitly avoids the errors shown in the incorrect examples. "
        "The prompt must be shorter than the original list. "
        "Do not copy the sentences. "
        "Return only the prompt."
    )

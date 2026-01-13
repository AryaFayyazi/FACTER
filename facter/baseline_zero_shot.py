"""
baseline_zero_shot.py: Baselines for paper-aligned comparisons.

We provide two variants:

1) ZeroShot_OpenEnded:
   - Same open-ended Top-K generation task as FACTER
   - No conformal validator
   - No prompt repair
   - Neutral system prompt

2) ZeroShot_CandidateRanker (optional):
   - Given a candidate set (retrieved externally), ask LLM to rank candidates.
   - This is a different task; use only if you explicitly want candidate-ranking comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from .config import Config
from .utils import generate_recommendations, parse_ranked_list


NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful recommendation assistant.\n"
    "Recommend items based on the user's watch history.\n"
    f"Return ONLY a JSON array of exactly {Config.TOP_K_RECS} item titles (strings), ranked best-first.\n"
)


@dataclass
class ZeroShotResult:
    recs: List[List[str]]              # open-ended raw recs
    mapped_recs: Optional[List[List[str]]] = None
    valid_at_k: Optional[List[float]] = None


def run_zero_shot_openended(
    df: pd.DataFrame,
    tokenizer,
    model,
    *,
    prompt_col: str = "prompt",
    system_msg: str = NEUTRAL_SYSTEM_PROMPT,
) -> List[List[str]]:
    prompts = df[prompt_col].tolist()
    return generate_recommendations(prompts, system_msg, tokenizer, model)


def run_zero_shot_candidate_ranker(
    df: pd.DataFrame,
    tokenizer,
    model,
    candidate_col: str = "candidates",
    context_col: str = "context",
    *,
    k: int = 10,
) -> List[List[str]]:
    """
    Expects df[candidate_col] is a list[str] of candidates per row.
    Builds a ranking prompt that forces ranking from candidates.
    """
    prompts = []
    for _, row in df.iterrows():
        ctx = str(row.get(context_col, ""))
        cands = row.get(candidate_col, [])
        cands = cands if isinstance(cands, list) else []
        cand_block = "\n".join([f"{i+1}. {c}" for i, c in enumerate(cands)])
        p = (
            ctx
            + "\n\nCandidates:\n"
            + cand_block
            + f"\n\nTask: Rank the top {k} candidates best-first.\n"
            + f"Return ONLY a JSON array of exactly {k} candidate titles."
        )
        prompts.append(p)

    system_msg = (
        "You are a helpful recommendation assistant.\n"
        f"Return ONLY a JSON array of exactly {k} titles."
    )

    # Use the same generator but parse list strictly
    recs = generate_recommendations(prompts, system_msg, tokenizer, model)
    # ensure length k
    out = []
    for r in recs:
        r = r if isinstance(r, list) else []
        out.append(r[:k])
    return out

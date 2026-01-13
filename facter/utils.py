"""
utils.py: Generation, parsing, and metrics for FACTER (paper-aligned).
- Generates Top-K ranked lists (open-vocabulary) and parses JSON arrays.
- Computes HitRate@K and NDCG@K for next-item prediction.
"""
from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch

from .config import Config

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


# -------------------------
# Prompt formatting (chat template if available)
# -------------------------
def _format_chat(tokenizer, system_msg: str, user_msg: str) -> torch.Tensor:
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # fallback
    text = f"<system>\n{system_msg}\n</system>\n<user>\n{user_msg}\n</user>\n<assistant>\n"
    return tokenizer(text, return_tensors="pt").input_ids


def _best_fuzzy_match(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def parse_ranked_list(text: str, k: int) -> List[str]:
    """
    Parse a model output into a list of titles.
    Prefers JSON array; otherwise parse numbered/bulleted lines.
    """
    if not text:
        return []

    # try JSON array
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                arr = [str(x).strip() for x in arr if str(x).strip()]
                # unique preserve order
                seen, out = set(), []
                for x in arr:
                    if x not in seen:
                        out.append(x)
                        seen.add(x)
                    if len(out) >= k:
                        break
                return out
        except Exception:
            pass

    # fallback parse lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        ln = re.sub(r"^\s*[\-\*\d\.\)\:]+\s*", "", ln).strip()
        if ln:
            out.append(ln)
        if len(out) >= k:
            break

    # unique preserve order
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq[:k]


def generate_recommendations(
    prompts: List[str],
    system_msg: str,
    tokenizer,
    model,
) -> List[List[str]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_recs: List[List[str]] = []

    for i in range(0, len(prompts), Config.BATCH_SIZE):
        batch = [p for p in prompts[i : i + Config.BATCH_SIZE] if p is not None]
        if not batch:
            continue

        # build batch input ids
        input_ids_list = [_format_chat(tokenizer, system_msg, p) for p in batch]
        # pad manually
        max_len = max(x.shape[-1] for x in input_ids_list)
        input_ids = torch.full((len(input_ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long)
        for j, x in enumerate(input_ids_list):
            input_ids[j, -x.shape[-1] :] = x[0]
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                repetition_penalty=Config.REPETITION_PENALTY,
                do_sample=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for txt in decoded:
            recs = parse_ranked_list(txt, Config.TOP_K_RECS)
            all_recs.append(recs)

    # if any prompts were None, keep alignment by returning empty lists for them
    if len(all_recs) != len(prompts):
        # best-effort: pad
        while len(all_recs) < len(prompts):
            all_recs.append([])
        all_recs = all_recs[: len(prompts)]
    return all_recs


# -------------------------
# Metrics (@K)
# -------------------------
def hitrate_ndcg_at_k(preds: List[str], gold: str, k: int) -> Tuple[float, float]:
    if not preds:
        return 0.0, 0.0
    gold = gold.strip()
    for rank, p in enumerate(preds[:k], start=1):
        if _best_fuzzy_match(p, gold) >= 0.85:
            # Hit@K = 1; NDCG@K = 1/log2(rank+1)
            return 1.0, 1.0 / np.log2(rank + 1)
    return 0.0, 0.0


def evaluate_at_k(df, k: int = 10) -> dict:
    hits, ndcgs = [], []
    for _, row in df.iterrows():
        preds = row["recs"] if isinstance(row["recs"], list) else []
        gold = str(row["target_title"])
        h, n = hitrate_ndcg_at_k(preds, gold, k)
        hits.append(h)
        ndcgs.append(n)
    return {
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }

def _best_fuzzy_match(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower().strip(), (b or "").lower().strip()).ratio()


def hitrate_ndcg_at_k(preds: List[str], gold: str, k: int) -> Tuple[float, float]:
    if not preds:
        return 0.0, 0.0
    gold = (gold or "").strip()
    for rank, p in enumerate(preds[:k], start=1):
        if p and _best_fuzzy_match(p, gold) >= 0.85:
            return 1.0, 1.0 / np.log2(rank + 1)
    return 0.0, 0.0


def evaluate_at_k_from_lists(
    rec_lists: List[List[str]],
    gold_titles: List[str],
    k: int = 10,
) -> Dict[str, float]:
    hits, ndcgs = [], []
    for recs, gold in zip(rec_lists, gold_titles):
        recs = recs if isinstance(recs, list) else []
        h, n = hitrate_ndcg_at_k(recs, str(gold), k)
        hits.append(h)
        ndcgs.append(n)
    return {
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def evaluate_valid_at_k(valid_at_k_list: List[float], k: int = 10) -> Dict[str, float]:
    # valid_at_k_list is already per-example fraction valid among top-k mapped
    return {f"Valid@{k}": float(np.mean(valid_at_k_list)) if valid_at_k_list else 0.0}

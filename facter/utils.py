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
    """Render one chat turn to a 1-D LongTensor of prompt token ids.

    ``apply_chat_template`` returns a plain tensor on some transformers versions
    and a BatchEncoding on others; normalise both to a tensor here so callers
    can rely on ``.shape``.
    """
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    if hasattr(tokenizer, "apply_chat_template"):
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(enc, torch.Tensor):
            enc = enc["input_ids"]
        return enc.reshape(-1)
    text = f"<system>\n{system_msg}\n</system>\n<user>\n{user_msg}\n</user>\n<assistant>\n"
    return tokenizer(text, return_tensors="pt").input_ids.reshape(-1)


# (an earlier duplicate of _best_fuzzy_match lived here, shadowed by the
# None-safe definition below; removed)


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
    """Batched Top-K generation.

    Three correctness requirements this path has to meet, all of which are easy
    to get wrong and all of which silently corrupt every downstream metric:

    1. **Decode only the completion.**  ``batch_decode`` over the full output
       tensor returns prompt + completion, and the prompt embeds the user's
       watch history as a numbered list -- exactly the format
       :func:`parse_ranked_list` looks for.  Parsing the full text therefore
       returns the user's *own history* as the recommendation list, for every
       method.  Because a counterfactual prompt differs from its base only in a
       few attribute words, the two echoed lists are near-identical and CFR
       collapses toward zero regardless of the method under test.  We slice off
       the prompt before decoding.

    2. **Pass an attention mask.**  Padding is applied on the left, so without a
       mask the model attends to pad tokens as if they were context.

    3. **Have a pad token.**  Llama-3 ships no pad token; fall back to EOS.
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    all_recs: List[List[str]] = []

    for i in range(0, len(prompts), Config.BATCH_SIZE):
        batch = [p for p in prompts[i : i + Config.BATCH_SIZE] if p is not None]
        if not batch:
            continue

        ids_list = [_format_chat(tokenizer, system_msg, p) for p in batch]
        ids_list = [x[-Config.MAX_PROMPT_LENGTH :] for x in ids_list]
        max_len = max(x.shape[-1] for x in ids_list)

        input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        for j, x in enumerate(ids_list):  # left-pad
            input_ids[j, -x.shape[-1] :] = x
            attention_mask[j, -x.shape[-1] :] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                repetition_penalty=Config.REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=pad_id,
            )

        # completion only -- see requirement (1) above
        completions = outputs[:, input_ids.shape[1] :]
        for txt in tokenizer.batch_decode(completions, skip_special_tokens=True):
            all_recs.append(parse_ranked_list(txt, Config.TOP_K_RECS))

    if len(all_recs) != len(prompts):
        while len(all_recs) < len(prompts):
            all_recs.append([])
        all_recs = all_recs[: len(prompts)]
    return all_recs


# -------------------------
# Metrics (@K)
# -------------------------
# (an earlier duplicate of hitrate_ndcg_at_k lived here and was silently
# shadowed by the definition below; removed)


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


def _as_gold_list(gold) -> List[str]:
    """Accept a single title, a list, or a '||'-joined relevance set."""
    if isinstance(gold, (list, tuple)):
        return [str(g).strip() for g in gold if str(g).strip()]
    g = str(gold or "").strip()
    if not g:
        return []
    return [t.strip() for t in g.split("||") if t.strip()]


def recall_ndcg_at_k(preds: List[str], gold, k: int) -> Tuple[float, float, float]:
    """Multi-target Recall@k, NDCG@k and HitRate@k under binary relevance.

    Recall@k = |R_rel ∩ R_k| / |R_rel|
    NDCG@k   = DCG@k / IDCG@k with gain 1 for a relevant item

    With a single target this reduces to the usual next-item metrics, where
    Recall@k == HitRate@k and NDCG@k <= Recall@k.  A published table showing
    NDCG@k > Recall@k therefore implies |R_rel| > 1, which is why the relevance
    window is now an explicit, configurable part of the protocol
    (Config.RELEVANCE_WINDOW) rather than an unstated choice.
    """
    golds = _as_gold_list(gold)
    if not preds or not golds:
        return 0.0, 0.0, 0.0, 0.0

    matched, hit = set(), 0.0
    dcg = 0.0
    for rank, p in enumerate(preds[:k], start=1):
        if not p:
            continue
        for gi, g in enumerate(golds):
            if gi in matched:
                continue
            if _best_fuzzy_match(p, g) >= 0.85:
                matched.add(gi)
                dcg += 1.0 / np.log2(rank + 1)
                hit = 1.0
                break

    ideal = min(len(golds), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
    recall = len(matched) / len(golds)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    # Precision@k is reported alongside because the published tables' "NDCG@10"
    # column matches this quantity, not NDCG (see REPRODUCIBILITY_NOTE.md §5).
    precision = len(matched) / float(k)
    return float(recall), float(ndcg), float(hit), float(precision)


def evaluate_at_k_from_lists(
    rec_lists: List[List[str]],
    gold_titles: List,
    k: int = 10,
) -> Dict[str, float]:
    """Recall@k / NDCG@k / HitRate@k. ``gold_titles`` may be single or multi-target."""
    recalls, ndcgs, hits, precs = [], [], [], []
    for recs, gold in zip(rec_lists, gold_titles):
        recs = recs if isinstance(recs, list) else []
        r, n, h, pr = recall_ndcg_at_k(recs, gold, k)
        recalls.append(r); ndcgs.append(n); hits.append(h); precs.append(pr)
    return {
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"Precision@{k}": float(np.mean(precs)) if precs else 0.0,
    }


def evaluate_valid_at_k(valid_at_k_list: List[float], k: int = 10) -> Dict[str, float]:
    # valid_at_k_list is already per-example fraction valid among top-k mapped
    return {f"Valid@{k}": float(np.mean(valid_at_k_list)) if valid_at_k_list else 0.0}

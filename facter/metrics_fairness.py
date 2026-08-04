"""
metrics_fairness.py: Fairness metric proxies for black-box LLM recommendations.

Implements:
- SNSR / SNSV as group-disparity proxies computed from embeddings of recommendations.
  * SNSR: max pairwise group distance (worst-case)
  * SNSV: mean pairwise group distance (average disparity)
- CFR as counterfactual "attribute flip on same context" proxy:
  For a given user context x, generate recs under attributes a and a',
  then compute distance between recommendation sets (embedding-based).

These are black-box computable proxies (no internal weights/activations required).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import util

from .catalog_map import rewrite_prompt_attrs, strip_prompt_attrs
from .config import Config

logger = logging.getLogger(__name__)


def _group_key_from_row(row: pd.Series) -> str:
    return "|".join([f"{a}={row[a]}" for a in Config.PROTECTED_ATTRIBUTES])


def _embed_texts(embedder, texts: List[str]) -> torch.Tensor:
    # returns torch tensor [n, d]
    embs = embedder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    if not isinstance(embs, torch.Tensor):
        embs = torch.tensor(embs)
    return embs


def _pool_recs_embedding(embedder, recs: List[str]) -> torch.Tensor:
    """
    Pool a list of titles into a single vector (mean of embeddings).
    Empty recs -> zeros vector of correct dimension (inferred from model with a dummy).
    """
    recs = [r for r in recs if isinstance(r, str) and r.strip()]
    if not recs:
        # infer dim
        dummy = _embed_texts(embedder, ["dummy"])
        return torch.zeros_like(dummy[0])
    E = _embed_texts(embedder, recs)
    return torch.mean(E, dim=0)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - util.cos_sim(a, b).item())


@dataclass
class SNSMetrics:
    SNSR: float
    SNSV: float
    details: Dict[str, float]


def compute_snsr_snsv(
    df: pd.DataFrame,
    embedder,
    recs_col: str = "mapped_recs",   # list[str]
    group_mode: str = "tuple",       # "tuple" or a single attribute name like "gender"
    min_group_size: int = 30,
) -> SNSMetrics:
    """
    Compute SNSR/SNSV over groups.

    group_mode:
      - "tuple": group by the full protected attribute tuple (default).
      - "<attr>": group by one protected attribute (e.g., "gender") to control one at a time.
    """
    if df is None or df.empty:
        return SNSMetrics(SNSR=0.0, SNSV=0.0, details={})

    if group_mode == "tuple":
        groups = df.groupby(df.apply(_group_key_from_row, axis=1))
    else:
        if group_mode not in df.columns:
            raise ValueError(f"group_mode='{group_mode}' not found in df columns")
        groups = df.groupby(df[group_mode].astype(str))

    # pool each group's recommendations into one embedding per example then mean within group
    group_vecs = {}
    for gname, gdf in groups:
        if len(gdf) < min_group_size:
            continue
        pooled = []
        for recs in gdf[recs_col].tolist():
            recs = recs if isinstance(recs, list) else []
            pooled.append(_pool_recs_embedding(embedder, recs))
        G = torch.stack(pooled, dim=0)
        group_vecs[str(gname)] = torch.mean(G, dim=0)

    names = list(group_vecs.keys())
    if len(names) < 2:
        return SNSMetrics(SNSR=0.0, SNSV=0.0, details={"n_groups_used": float(len(names))})

    # pairwise distances
    dists = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            di = _cosine_distance(group_vecs[names[i]], group_vecs[names[j]])
            dists.append(di)

    SNSR = float(np.max(dists)) if dists else 0.0
    SNSV = float(np.mean(dists)) if dists else 0.0
    return SNSMetrics(SNSR=SNSR, SNSV=SNSV, details={"n_groups_used": float(len(names))})


@dataclass
class CFRMetrics:
    CFR: float
    valid_rate: float
    n_pairs: int


def compute_cfr(
    df: pd.DataFrame,
    embedder,
    generate_fn: Callable[[List[str], str], List[List[str]]],
    system_msg_neutral: str = "",
    *,
    k: int = 10,
    n_samples: int = 200,
    flip_mode: str = "tuple",   # "tuple" or single attribute name e.g. "gender"
    attr_value_sampler: Optional[Callable[[str, pd.DataFrame], str]] = None,
    recs_distance: Optional[str] = None,  # "l2" (Eq. 14) | "pooled_cos" | "set_cos"
    prompt_col: str = "prompt",
    system_msg: Optional[str] = None,
    prompt_transform: Optional[Callable[[str, Dict[str, str]], str]] = None,
) -> CFRMetrics:
    """Counterfactual Fairness Ratio.

    IMPORTANT (measurement correctness).  Earlier releases always generated both
    arms with the *neutral* system prompt and the *unmodified* user prompt, for
    every method.  Under that setup FACTER's repaired prompt never entered the
    measurement at all, so CFR could not differ between the zero-shot baseline
    and FACTER by anything other than sampling noise -- which is what an external
    reproduction observed.  CFR must be measured under whatever prompt the method
    being evaluated would actually deploy.

    ``system_msg``       system prompt of the method under evaluation
                         (falls back to ``system_msg_neutral`` for the baseline).
    ``prompt_transform`` optional ``(prompt, attrs) -> prompt`` hook applying the
                         method's user-prompt repair.  It is applied to the
                         counterfactual arm using the *counterfactual* group, so
                         the intervention is evaluated as it would be served.
    """
    """
    CFR proxy via counterfactual attribute flip on the SAME context:
    - sample examples
    - create counterfactual prompt with flipped attribute(s)
    - generate rec lists for original and counterfactual
    - compute semantic distance between lists

    flip_mode:
      - "tuple": resample a different protected tuple from the dataset distribution
      - "<attr>": flip only one attribute; other attrs stay unchanged

    recs_distance:
      - "pooled_cos": pool embeddings of list and use cosine distance
      - "set_cos": compute mean of top-1 cosine distances across corresponding ranks
    """
    if df is None or df.empty:
        return CFRMetrics(CFR=0.0, valid_rate=0.0, n_pairs=0)

    rng = random.Random(Config.RANDOM_SEED)

    # Prepare available attribute values for sampling
    def default_sampler(attr: str, data: pd.DataFrame) -> str:
        vals = data[attr].astype(str).unique().tolist()
        return rng.choice(vals)

    sampler = attr_value_sampler or default_sampler

    rows = df.sample(n=min(n_samples, len(df)), replace=False, random_state=Config.RANDOM_SEED)
    cfr_vals = []
    valid_pairs = 0

    for _, row in rows.iterrows():
        base_prompt = row[prompt_col]
        if not isinstance(base_prompt, str) or not base_prompt.strip():
            continue

        base_attrs = {a: str(row[a]) for a in Config.PROTECTED_ATTRIBUTES}

        if flip_mode == "tuple":
            # sample a different tuple from dataset
            # pick a random row with different tuple key
            base_key = _group_key_from_row(row)
            # try a few times
            cf_attrs = None
            for _try in range(10):
                rr = df.iloc[rng.randrange(len(df))]
                key2 = _group_key_from_row(rr)
                if key2 != base_key:
                    cf_attrs = {a: str(rr[a]) for a in Config.PROTECTED_ATTRIBUTES}
                    break
            if cf_attrs is None:
                continue
        else:
            if flip_mode not in Config.PROTECTED_ATTRIBUTES:
                raise ValueError(f"flip_mode must be 'tuple' or one of {Config.PROTECTED_ATTRIBUTES}")
            cf_attrs = dict(base_attrs)
            cf_attrs[flip_mode] = sampler(flip_mode, df)

            # Ensure it actually changes
            if cf_attrs[flip_mode] == base_attrs[flip_mode]:
                continue

        cf_prompt = rewrite_prompt_attrs(base_prompt, cf_attrs)

        # Apply the evaluated method's prompt repair to each arm under its own
        # group, so the intervention is measured as it would be deployed.
        a_prompt, b_prompt = base_prompt, cf_prompt
        if prompt_transform is not None:
            a_prompt = prompt_transform(base_prompt, base_attrs)
            b_prompt = prompt_transform(cf_prompt, cf_attrs)

        sys_msg = system_msg if system_msg is not None else system_msg_neutral
        recs_pair = generate_fn([a_prompt, b_prompt], sys_msg)
        if not (isinstance(recs_pair, list) and len(recs_pair) == 2):
            continue
        recs_a, recs_b = recs_pair[0], recs_pair[1]
        recs_a = recs_a[:k] if isinstance(recs_a, list) else []
        recs_b = recs_b[:k] if isinstance(recs_b, list) else []

        if not recs_a or not recs_b:
            continue

        # compute distance
        mode = recs_distance or getattr(Config, "CFR_DISTANCE", "l2")
        if mode == "l2":
            # Eq. 14: CFR = E[ || f(x) - f(x_not_s) ||_2 ].  The paper states an
            # L2 norm; earlier code used cosine distance, a different scale.
            va = _pool_recs_embedding(embedder, recs_a)
            vb = _pool_recs_embedding(embedder, recs_b)
            va = va / (torch.norm(va) + 1e-12)
            vb = vb / (torch.norm(vb) + 1e-12)
            dist = float(torch.norm(va - vb, p=2).item())
        elif mode == "pooled_cos":
            va = _pool_recs_embedding(embedder, recs_a)
            vb = _pool_recs_embedding(embedder, recs_b)
            dist = _cosine_distance(va, vb)
        elif mode == "set_cos":
            # mean rank-wise cosine distance (top-k aligned)
            m = min(len(recs_a), len(recs_b), k)
            if m == 0:
                continue
            Ea = _embed_texts(embedder, recs_a[:m])
            Eb = _embed_texts(embedder, recs_b[:m])
            # 1 - cos for each rank
            dist = float(torch.mean(1.0 - torch.diag(util.cos_sim(Ea, Eb))).item())
        else:
            raise ValueError("recs_distance must be 'l2', 'pooled_cos' or 'set_cos'")

        cfr_vals.append(dist)
        valid_pairs += 1

    CFR = float(np.mean(cfr_vals)) if cfr_vals else 0.0
    valid_rate = float(valid_pairs) / float(len(rows)) if len(rows) > 0 else 0.0
    return CFRMetrics(CFR=CFR, valid_rate=valid_rate, n_pairs=valid_pairs)


# ---------------------------------------------------------------------------
# SNSR / SNSV, as actually defined by FaiRLLM (Zhang et al., 2023)
# ---------------------------------------------------------------------------
@dataclass
class SNSMetricsFairLLM:
    SNSR: float                       # mean over attributes of max_a Sim(a) - min_a Sim(a)
    SNSV: float                       # mean over attributes of std_a Sim(a)
    per_attribute: Dict[str, Dict[str, float]]
    n_users: int


def _jaccard(a: List[str], b: List[str]) -> float:
    A = {x for x in a if x}
    B = {x for x in b if x}
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def compute_snsr_snsv_fairllm(
    df: pd.DataFrame,
    embedder,
    generate_fn: Callable[[List[str], str], List[List[str]]],
    *,
    system_msg: str = "",
    prompt_transform: Optional[Callable[[str, Dict[str, str]], str]] = None,
    map_fn: Optional[Callable[[List[str]], List[str]]] = None,
    attributes: Optional[List[str]] = None,
    k: int = 10,
    n_users: int = 60,
    similarity: str = "jaccard",
    prompt_col: str = "prompt",
) -> SNSMetricsFairLLM:
    """SNSR / SNSV per FaiRLLM (Zhang et al., 2023), the metric FACTER cites.

        Sim_bar(a) = mean over users of Sim(R_neutral, R_a)
        SNSR@K     = max_a Sim_bar(a) - min_a Sim_bar(a)
        SNSV@K     = std_a Sim_bar(a)

    ``R_neutral`` is generated from a prompt with the protected-attribute block
    removed, and ``R_a`` from the same context with attribute value ``a``
    injected.  Both are required by the definition.

    This replaces an earlier implementation that computed pairwise cosine
    distance between group-mean pooled embeddings.  That quantity never
    referenced a neutral prompt, is not a range over attribute values, and is
    not SNSR; it also cannot move much, because averaging recommendation
    embeddings within a group washes out exactly the per-user differences the
    metric is meant to expose.
    """
    if df is None or df.empty:
        return SNSMetricsFairLLM(0.0, 0.0, {}, 0)

    attributes = attributes or list(Config.PROTECTED_ATTRIBUTES)
    rows = df.sample(n=min(n_users, len(df)), replace=False,
                     random_state=Config.RANDOM_SEED)

    def _mapped(recs: List[str]) -> List[str]:
        recs = recs[:k] if isinstance(recs, list) else []
        return map_fn(recs) if map_fn is not None else recs

    # similarity accumulators: attr -> value -> [per-user similarity]
    acc: Dict[str, Dict[str, List[float]]] = {a: {} for a in attributes}
    values = {a: sorted(df[a].astype(str).unique().tolist()) for a in attributes}

    for _, row in rows.iterrows():
        base_prompt = row[prompt_col]
        if not isinstance(base_prompt, str) or not base_prompt.strip():
            continue
        base_attrs = {a: str(row[a]) for a in Config.PROTECTED_ATTRIBUTES}

        # build the neutral prompt and one prompt per (attribute, value)
        neutral = strip_prompt_attrs(base_prompt)
        variants: List[Tuple[str, str, str]] = []   # (attr, value, prompt)
        for a in attributes:
            for v in values[a]:
                at = dict(base_attrs); at[a] = v
                variants.append((a, v, rewrite_prompt_attrs(base_prompt, at)))

        prompts = [neutral] + [p for _, _, p in variants]
        attrs_for_transform = [base_attrs] + [
            {**base_attrs, a: v} for a, v, _ in variants
        ]
        if prompt_transform is not None:
            prompts = [prompt_transform(p, at)
                       for p, at in zip(prompts, attrs_for_transform)]

        outs = generate_fn(prompts, system_msg)
        if not outs or len(outs) != len(prompts):
            continue

        r_neutral = _mapped(outs[0])
        if similarity == "embedding":
            v_neutral = _pool_recs_embedding(embedder, r_neutral)

        for (a, v, _), out in zip(variants, outs[1:]):
            r_a = _mapped(out)
            if similarity == "jaccard":
                sim = _jaccard(r_neutral, r_a)
            elif similarity == "embedding":
                sim = 1.0 - _cosine_distance(v_neutral,
                                             _pool_recs_embedding(embedder, r_a))
            else:
                raise ValueError("similarity must be 'jaccard' or 'embedding'")
            acc[a].setdefault(v, []).append(float(sim))

    per_attr: Dict[str, Dict[str, float]] = {}
    snsrs, snsvs = [], []
    for a in attributes:
        means = [float(np.mean(vals)) for vals in acc[a].values() if vals]
        if len(means) < 2:
            continue
        snsr = float(np.max(means) - np.min(means))
        snsv = float(np.std(means))
        per_attr[a] = {"SNSR": snsr, "SNSV": snsv, "n_values": float(len(means))}
        snsrs.append(snsr); snsvs.append(snsv)

    return SNSMetricsFairLLM(
        SNSR=float(np.mean(snsrs)) if snsrs else 0.0,
        SNSV=float(np.mean(snsvs)) if snsvs else 0.0,
        per_attribute=per_attr,
        n_users=int(len(rows)),
    )

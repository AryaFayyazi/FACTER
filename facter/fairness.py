"""
fairness.py: Conformal fairness calibration + validation for FACTER (paper-aligned).
Implements:
- S = d + λΔ (paper)
- Cross-group neighborhoods for Δ (ai != aj)
- Online threshold update with exponential decay (paper Eq. 11)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import util

from .config import Config

logger = logging.getLogger(__name__)


def _group_key(attrs: Dict[str, str]) -> str:
    return "|".join([f"{k}={attrs.get(k,'')}" for k in Config.PROTECTED_ATTRIBUTES])


def _cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    # 1 - cosine similarity
    return float(1.0 - util.cos_sim(a, b).item())


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a - b, p=2).item())


@dataclass
class ViolationRecord:
    context: str
    prompt: str
    recs: List[str]
    group: str
    score: float
    threshold: float
    features: List[str]


class ConformalFairnessValidator:
    """
    Offline calibrate a conformal threshold on nonconformity scores.
    Online: compute S_new and flag violation if S_new > Q.
    On violation: update Q via exponential decay (paper Eq. 11) and store violation.
    """

    def __init__(self, embedder, item_db: Optional[Dict] = None):
        self.embedder = embedder
        self.item_db = item_db or {}

        # calibration stores
        self.cal_contexts: List[str] = []
        self.cal_prompts: List[str] = []
        self.cal_groups: List[str] = []
        self.cal_yhat_embeds: Optional[torch.Tensor] = None
        self.cal_context_embeds: Optional[torch.Tensor] = None

        self.adaptive_threshold: Optional[float] = None
        self.violation_memory: List[ViolationRecord] = []
        self.violation_count: int = 0

    # -------------------------
    # Feature extraction (simple)
    # -------------------------
    def _extract_features(self, rec_titles: List[str]) -> List[str]:
        """
        Minimal feature extractor:
        - MovieLens: use genre if we can match titles back to item_db entries.
        - Otherwise: keyword tokens from titles.
        """
        feats = []
        # build a reverse map title->genre if available
        title_to_genre = {}
        for mid, info in self.item_db.items():
            t = str(info.get("title", "")).lower()
            g = str(info.get("genre", "")).strip()
            if t and g:
                title_to_genre[t] = g

        for t in rec_titles[: min(5, len(rec_titles))]:
            tl = t.lower().strip()
            if tl in title_to_genre:
                # use first genre token
                g = title_to_genre[tl].split("|")[0]
                feats.append(f"genre:{g}")
            else:
                toks = [w for w in tl.replace(":", " ").replace("-", " ").split() if len(w) >= 4]
                feats.extend([f"kw:{w}" for w in toks[:2]])
        # de-duplicate
        return list(dict.fromkeys(feats))[:10]

    # -------------------------
    # Scoring: S = d + λΔ
    # -------------------------
    def _neighbors_cross_group(self, context_embed: torch.Tensor, group: str) -> List[int]:
        """
        Find cross-group neighbors among calibration set by context similarity,
        then filter to ai != aj and similarity >= τ.
        """
        assert self.cal_context_embeds is not None
        sims = util.cos_sim(context_embed.unsqueeze(0), self.cal_context_embeds).squeeze(0)

        # topK candidates then filter
        k = min(Config.N_REFERENCE * 3, sims.shape[0])
        top_idx = torch.topk(sims, k=k).indices.tolist()

        out = []
        for j in top_idx:
            if self.cal_groups[j] == group:
                continue
            if float(sims[j].item()) < Config.BASE_SIMILARITY:
                continue
            out.append(j)
            if len(out) >= Config.N_REFERENCE:
                break
        return out

    def _score_S(
        self,
        context: str,
        group: str,
        y_hat_title: str,
        y_true_title: Optional[str],
    ) -> float:
        """
        Compute S = d + λΔ.
        d: predictive error (embedding-based) if y_true is available else 0.
        Δ: max L2 distance from cross-group neighbors' y_hat embeddings.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        ctx_e = self.embedder.encode(context, convert_to_tensor=True, show_progress_bar=False).to(device)
        yhat_e = self.embedder.encode(y_hat_title, convert_to_tensor=True, show_progress_bar=False).to(device)

        # d term
        d = 0.0
        if y_true_title is not None and str(y_true_title).strip():
            ytrue_e = self.embedder.encode(str(y_true_title), convert_to_tensor=True, show_progress_bar=False).to(device)
            d = _cos_dist(yhat_e, ytrue_e)

        # Δ term
        delta = 0.0
        if self.cal_context_embeds is not None and self.cal_yhat_embeds is not None:
            nbrs = self._neighbors_cross_group(ctx_e, group)
            if nbrs:
                nbr_embeds = self.cal_yhat_embeds[nbrs].to(device)
                # max L2 to neighbors
                dists = torch.norm(nbr_embeds - yhat_e.unsqueeze(0), p=2, dim=1)
                delta = float(torch.max(dists).item())

        return float(d + Config.LAMBDA_FAIRNESS * delta)

    # -------------------------
    # Conformal calibration
    # -------------------------
    @staticmethod
    def _conformal_quantile(scores: Sequence[float], alpha: float) -> float:
        """
        Finite-sample conformal quantile:
          Q = score_(k) where k = ceil((n+1)*(1-alpha))
        """
        s = np.asarray(list(scores), dtype=float)
        s = np.sort(s)
        n = len(s)
        if n == 0:
            return float("inf")
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        return float(s[k - 1])

    def calibrate(
        self,
        cal_contexts: List[str],
        cal_prompts: List[str],
        cal_groups: List[str],
        cal_recs: List[List[str]],
        cal_targets: List[str],
    ) -> None:
        """
        Offline calibration over a calibration set.
        We embed:
          - contexts for neighbor search
          - y_hat (rank-1 title) for Δ
        Then compute S_i and set Q_alpha.
        """
        assert len(cal_contexts) == len(cal_prompts) == len(cal_groups) == len(cal_recs) == len(cal_targets)

        self.cal_contexts = cal_contexts
        self.cal_prompts = cal_prompts
        self.cal_groups = cal_groups

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Embedding calibration contexts...")
        self.cal_context_embeds = self.embedder.encode(
            cal_contexts, convert_to_tensor=True, show_progress_bar=True
        ).to(device)

        # rank-1 rec for calibration
        yhat_titles = [r[0] if (isinstance(r, list) and len(r) > 0) else "" for r in cal_recs]
        logger.info("Embedding calibration rank-1 recommendations...")
        self.cal_yhat_embeds = self.embedder.encode(
            yhat_titles, convert_to_tensor=True, show_progress_bar=True
        ).to(device)

        logger.info("Computing calibration S scores...")
        scores = []
        for i in range(len(cal_contexts)):
            s_i = self._score_S(
                context=cal_contexts[i],
                group=cal_groups[i],
                y_hat_title=yhat_titles[i],
                y_true_title=cal_targets[i],
            )
            scores.append(s_i)

        self.adaptive_threshold = self._conformal_quantile(scores, Config.ALPHA)
        logger.info(f"Calibration complete: Q_alpha={self.adaptive_threshold:.4f} (n={len(scores)})")

    # -------------------------
    # Online update + validation
    # -------------------------
    def _update_threshold_on_violation(self, s_new: float) -> None:
        """
        Exponential update: Q <- γ Q + (1-γ) s_new
        """
        if self.adaptive_threshold is None:
            self.adaptive_threshold = s_new
            return
        self.adaptive_threshold = float(
            Config.QUANTILE_DECAY * self.adaptive_threshold + (1.0 - Config.QUANTILE_DECAY) * s_new
        )

    def validate(
        self,
        context: str,
        prompt: str,
        attrs: Dict[str, str],
        recs: List[str],
        y_true_title: Optional[str] = None,
    ) -> Tuple[bool, float, float]:
        """
        Returns (is_violation, S_new, current_threshold).
        On violation: stores violation + updates threshold.
        """
        if self.adaptive_threshold is None:
            raise RuntimeError("Validator must be calibrated before validate().")

        group = _group_key(attrs)
        yhat_title = recs[0] if recs else ""
        s_new = self._score_S(context=context, group=group, y_hat_title=yhat_title, y_true_title=y_true_title)

        is_violation = bool(s_new > float(self.adaptive_threshold))
        if is_violation:
            self.violation_count += 1
            feats = self._extract_features(recs)
            self._store_violation(context, prompt, recs, group, s_new, float(self.adaptive_threshold), feats)
            self._update_threshold_on_violation(s_new)

        return is_violation, float(s_new), float(self.adaptive_threshold)

    def _store_violation(
        self,
        context: str,
        prompt: str,
        recs: List[str],
        group: str,
        score: float,
        threshold: float,
        features: List[str],
    ) -> None:
        if len(self.violation_memory) >= Config.VIOLATION_MEMORY_SIZE:
            self.violation_memory.pop(0)
        self.violation_memory.append(
            ViolationRecord(
                context=context,
                prompt=prompt,
                recs=recs,
                group=group,
                score=score,
                threshold=threshold,
                features=features,
            )
        )

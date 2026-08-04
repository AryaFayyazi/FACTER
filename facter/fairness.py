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
        # Q^(0): the calibration threshold, frozen. Violations are reported
        # against both this and the moving Q^(t) -- see Config.REPORT_FIXED_THRESHOLD.
        self.fixed_threshold: Optional[float] = None
        self.violation_memory: List[ViolationRecord] = []
        self.violation_count: int = 0          # against the adaptive Q^(t)
        self.violation_count_fixed: int = 0    # against the frozen Q^(0)
        self.n_validated: int = 0

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

        q_raw = self._conformal_quantile(scores, Config.ALPHA)
        # Eq. 15: Q = Quantile(1-alpha; {s_i}) + C/sqrt(n).  The paper leaves C
        # unspecified, but Eq. 18 uses sqrt(log(2/delta)/(2n)) as its
        # finite-sample slack, so C = sqrt(log(2/delta)/2) reproduces it exactly.
        # Earlier code omitted the correction entirely.
        self.correction = 0.0
        if getattr(Config, "USE_EQ15_CORRECTION", True):
            n = max(1, len(scores))
            C = float(np.sqrt(np.log(2.0 / Config.CONFORMAL_DELTA) / 2.0))
            self.correction = float(C / np.sqrt(n))
        self.adaptive_threshold = float(q_raw + self.correction)

        # Keep the calibration score distribution so the exchangeability
        # assumption can be checked directly against the test scores rather
        # than inferred from the violation count.
        qs = [1, 5, 10, 25, 50, 75, 80, 90, 95, 99]
        self.cal_score_quantiles = {
            f"p{q}": float(np.percentile(scores, q)) for q in qs
        }
        self.cal_score_mean = float(np.mean(scores))
        self.cal_score_std = float(np.std(scores))
        self.cal_n = int(len(scores))
        # Freeze the calibration threshold.  Everything downstream reports
        # violations against BOTH this and the moving threshold.
        self.fixed_threshold = float(self.adaptive_threshold)
        logger.info(
            f"Calibration complete: Q_alpha={self.adaptive_threshold:.4f} (n={len(scores)}) "
            f"[frozen Q^(0)={self.fixed_threshold:.4f}]"
        )

    # -------------------------
    # Online update + validation
    # -------------------------
    def _update_threshold(self, s_new: float, is_violation: bool) -> None:
        """Advance the online threshold by the configured rule.

        See ``Config.THRESHOLD_UPDATE`` for why the default is the two-sided ACI
        update rather than the one-sided exponential rule shipped previously.
        The short version: a violation means ``s_new > Q``, so the one-sided rule
        only ever moves ``Q`` upward, and a violation count measured against a
        monotonically rising bar falls whether or not the recommendations got
        any fairer.
        """
        if self.adaptive_threshold is None:
            self.adaptive_threshold = s_new
            return

        mode = getattr(Config, "THRESHOLD_UPDATE", "aci")

        if mode == "aci":
            # Gibbs & Candes (2021): Q_{t+1} = Q_t + eta * (alpha - 1{violation}).
            # Tightens on a conforming step, loosens on a violation, and tracks
            # the alpha-quantile in the long run instead of drifting.
            eta = float(getattr(Config, "ACI_STEP", 0.05))
            err = 1.0 if is_violation else 0.0
            self.adaptive_threshold = float(
                self.adaptive_threshold + eta * (err - Config.ALPHA)
            )
            # keep the threshold in a sane range for a bounded score
            self.adaptive_threshold = float(max(0.0, self.adaptive_threshold))
            return

        if not is_violation:
            return  # the legacy and Eq.11 rules only fire on violations

        if mode == "paper_eq11":
            # Exactly as printed: min(Q, s_new) == Q whenever s_new > Q, so this
            # is a no-op on violations.  Retained so the published equation can
            # be run and inspected.
            self.adaptive_threshold = float(
                Config.QUANTILE_DECAY * self.adaptive_threshold
                + (1.0 - Config.QUANTILE_DECAY) * min(self.adaptive_threshold, s_new)
            )
        elif mode == "legacy":
            self.adaptive_threshold = float(
                Config.QUANTILE_DECAY * self.adaptive_threshold
                + (1.0 - Config.QUANTILE_DECAY) * s_new
            )
        else:
            raise ValueError(f"unknown THRESHOLD_UPDATE {mode!r}")

    def score_only(
        self,
        context: str,
        attrs: Dict[str, str],
        recs: List[str],
        y_true_title: Optional[str] = None,
    ) -> float:
        """Nonconformity score for a candidate output, with no side effects.

        Used to score baselines against the same frozen threshold Q^(0) as
        FACTER.  Unlike :meth:`validate` this does not touch the adaptive
        threshold, the violation buffer, or any counter.
        """
        if self.adaptive_threshold is None:
            raise RuntimeError("Validator must be calibrated before score_only().")
        return self._score_S(
            context=context,
            group=_group_key(attrs),
            y_hat_title=recs[0] if recs else "",
            y_true_title=y_true_title,
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
        self.n_validated += 1

        # Independent accounting against the frozen calibration threshold.  This
        # is the number that answers "did the recommendations change?" as opposed
        # to "did the bar move?".
        if self.fixed_threshold is not None and s_new > float(self.fixed_threshold):
            self.violation_count_fixed += 1

        if is_violation:
            self.violation_count += 1
            feats = self._extract_features(recs)
            self._store_violation(context, prompt, recs, group, s_new, float(self.adaptive_threshold), feats)

        # The threshold advances on every step, not only on violations, so that
        # it can tighten as well as loosen (see _update_threshold).
        self._update_threshold(s_new, is_violation)

        return is_violation, float(s_new), float(self.adaptive_threshold)

    def summary(self) -> Dict[str, float]:
        """Violation counts under both the adaptive and the frozen threshold."""
        n = max(1, self.n_validated)
        return {
            "n_validated": float(self.n_validated),
            "violations_adaptive": float(self.violation_count),
            "violations_fixed": float(self.violation_count_fixed),
            "violation_rate_adaptive": self.violation_count / n,
            "violation_rate_fixed": self.violation_count_fixed / n,
            "Q_fixed": float(self.fixed_threshold) if self.fixed_threshold is not None else float("nan"),
            "cal_score_quantiles": getattr(self, "cal_score_quantiles", None),
            "cal_score_mean": getattr(self, "cal_score_mean", None),
            "cal_score_std": getattr(self, "cal_score_std", None),
            "Q_adaptive": float(self.adaptive_threshold) if self.adaptive_threshold is not None else float("nan"),
        }

    def reset_counts(self) -> None:
        """Zero the per-iteration counters (the threshold and buffer persist)."""
        self.violation_count = 0
        self.violation_count_fixed = 0
        self.n_validated = 0

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

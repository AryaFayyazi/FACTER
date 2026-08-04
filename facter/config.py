"""
config.py: Centralized configuration and hyperparameters for FACTER (paper-aligned).
"""
from __future__ import annotations

import os as _os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # -------------------------
    # Data
    # -------------------------
    DATASETS = {
        "ml-1m": {
            "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "paths": ["ratings.dat", "users.dat", "movies.dat"],
        },
        "amazon": {
            "url": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Movies_and_TV_5.json.gz",
            "sample_size": 2500,
        },
    }
    EXTRACT_DIR: Path = Path("./data/")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Models
    # -------------------------
    # Paper: Llama-3-8B-Instruct (the demo repo used 3.1 for convenience)
    # meta-llama/* is licence-gated; the NousResearch mirror carries identical
    # weights and needs no token.  Override with the FACTER_LLM env var.
    LLM_BACKBONE: str = "NousResearch/Meta-Llama-3-8B-Instruct"
    EMBEDDER_NAME: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    # A public drop-in alternative fine-tuned for movie retrieval (we used this)
    EMBEDDER_ALT_PUBLIC: str = "JJTsao/fine-tuned_movie_retriever-all-mpnet-base-v2"

    # -------------------------
    # Generation / evaluation
    # -------------------------
    MAX_PROMPT_LENGTH: int = 2048
    MAX_NEW_TOKENS: int = 250
    BATCH_SIZE: int = 8
    TOP_K_RECS: int = 10
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    REPETITION_PENALTY: float = 1.2

    # History construction (paper-aligned: open-vocab “next item”)
    HISTORY_SIZE: int = 10
    MIN_SEQ_LENGTH: int = 5

    # -------------------------
    # Fairness / conformal
    # -------------------------
    PROTECTED_ATTRIBUTES = ["gender", "age", "occupation"]

    # Values below are the ones REPORTED IN THE PAPER (§4.1 "Hyperparameter
    # Settings": tau_rho=0.9, lambda=0.7, gamma=0.95, M=50).  Earlier releases
    # shipped lambda=0.5, tau_rho=0.65, gamma=0.92, so anyone running the code
    # reproduced neither the paper nor its own ablations.
    # alpha = 0.2 is the paper's setting (Table 4 lists the theoretical violation
    # rate as "0.2 +/- 0.02", i.e. alpha).  The paper's "Hyperparameter Settings"
    # paragraph states tau_rho, lambda, gamma and M but never alpha; it is
    # recoverable only from Table 4.
    ALPHA: float = float(_os.environ.get("FACTER_ALPHA", 0.2))
    LAMBDA_FAIRNESS: float = 0.7  # λ in S = d + λΔ

    # Neighborhood for Δ (cross-group)
    N_REFERENCE: int = 20
    BASE_SIMILARITY: float = 0.90  # τ_ρ : minimum context similarity to be a neighbor

    QUANTILE_DECAY: float = 0.95  # γ
    VIOLATION_MEMORY_SIZE: int = 50  # M
    RULE_MIN_COUNT: int = 3  # n: a (group, feature) pattern must recur this often
    MAX_RULES_INJECTED: int = 5

    # -------------------------
    # Online threshold update
    # -------------------------
    # "aci"     : two-sided adaptive conformal update (Gibbs & Candes, 2021),
    #             Q <- Q + eta * (alpha - 1{violation}).  This is the default.
    # "legacy"  : the one-sided rule shipped in earlier releases,
    #             Q <- gamma*Q + (1-gamma)*S_new, applied ONLY on violations.
    # "paper_eq11": Eq. 11 exactly as printed in the paper,
    #             Q <- gamma*Q + (1-gamma)*min(Q, S_new).
    #
    # Why the default changed.  A violation means S_new > Q, so "legacy" moves Q
    # *up* on every violation and never moves it down -- a monotone drift that
    # makes later violations progressively harder to trigger regardless of
    # whether the recommendations improved.  "paper_eq11" has the opposite
    # problem: min(Q, S_new) = Q whenever a violation fires, so the rule is an
    # exact no-op.  Neither can converge to the alpha-quantile the method is
    # supposed to track.  The ACI update is two-sided -- it tightens after a
    # violation-free step and loosens after a violation -- and carries a
    # long-run coverage guarantee, which is what §3.3 describes in words.
    THRESHOLD_UPDATE: str = _os.environ.get("FACTER_THRESHOLD_RULE", "aci")
    ACI_STEP: float = 0.05  # eta

    # Report violations against the FIXED calibration threshold Q^(0) as well as
    # the adaptive Q^(t).  A count measured only against a moving threshold
    # cannot distinguish "the outputs improved" from "the bar moved".
    REPORT_FIXED_THRESHOLD: bool = True

    MAX_ITERATIONS: int = 3  # referenced by main.py; absent from earlier releases

    # -------------------------
    # Evaluation protocol
    # -------------------------
    # Size of the relevance set: the user's next RELEVANCE_WINDOW interactions.
    # 1 == strict next-item (Recall@k == HitRate@k and NDCG@k <= Recall@k).
    # Paper protocol (Sec 4.1 "Datasets"): ML-1M sampled to 2,500 interactions
    # split 70:30 -> 750 test samples; Amazon 3,750 -> 1,125 test samples.
    PAPER_N_INTERACTIONS = {"ml-1m": 2500, "amazon": 3750}

    RELEVANCE_WINDOW: int = int(_os.environ.get("FACTER_RELEVANCE_WINDOW", 10))
    # Eq. 14 states an L2 norm between output embeddings.
    CFR_DISTANCE: str = _os.environ.get("FACTER_CFR_DISTANCE", "l2")
    # Eq. 15 writes Q = Quantile(1-alpha) + C/sqrt(n).  Instantiating C with the
    # constant from Eq. 18, sqrt(log(2/delta)/2), turns Q into a conservative
    # high-probability UPPER BOUND on the quantile rather than the operating
    # threshold: measured exceedance drops to 7.8% at a nominal alpha of 0.2.
    # The plain finite-sample conformal quantile S_(ceil((n+1)(1-alpha))) is
    # already valid and reproduces alpha exactly (measured: 20.1%), so it is the
    # default.  Enable this only if the conservative bound is wanted.
    USE_EQ15_CORRECTION: bool = _os.environ.get("FACTER_EQ15", "0") == "1"
    CONFORMAL_DELTA: float = 0.05          # delta in Eq. 18 -> C = sqrt(log(2/delta)/2)
    # SNSR/SNSV per FaiRLLM (Zhang et al., 2023): similarity-to-neutral spread.
    SNSR_N_USERS: int = int(_os.environ.get("FACTER_SNSR_USERS", 40))
    SNSR_EVERY_ITERATION: bool = _os.environ.get("FACTER_SNSR_EVERY", "0") == "1"
    SNSR_SIMILARITY: str = _os.environ.get("FACTER_SNSR_SIM", "jaccard")
    # Task formulation: "open" (free generation) or "rerank" (rank a candidate set).
    TASK_MODE: str = _os.environ.get("FACTER_TASK", "open")
    RERANK_N_CANDIDATES: int = int(_os.environ.get("FACTER_N_CANDIDATES", 40))

    # "neutral" calibrates under the same system prompt the zero-shot baseline
    # deploys, preserving exchangeability; "empty" restores the previous
    # (invalid) behaviour for comparison.
    CALIBRATION_PROMPT: str = _os.environ.get("FACTER_CAL_PROMPT", "neutral")

    CFR_N_SAMPLES: int = 200
    CFR_EVERY_ITERATION: bool = _os.environ.get("FACTER_CFR_EVERY", "1") == "1"

    # Fairness metric bootstrapping (optional)
    MIN_GROUP_SIZE: int = 30
    N_BOOTSTRAP: int = 200

    # Reproducibility
    # A 15-20% effect cannot be claimed from one seed.  Runs are seeded from
    # FACTER_SEED so the same protocol can be repeated and aggregated with
    # scripts/aggregate_seeds.py.
    RANDOM_SEED: int = int(_os.environ.get("FACTER_SEED", 42))

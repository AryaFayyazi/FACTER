"""
config.py: Centralized configuration and hyperparameters for FACTER (paper-aligned).
"""
from __future__ import annotations
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
    LLM_BACKBONE: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Default embedder (fine-tuning mpnet)
    EMBEDDER_NAME: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    # A public drop-in alternative fine-tuned for movie retrieval (optional)
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

    ALPHA: float = 0.2  # miscoverage level
    LAMBDA_FAIRNESS: float = 0.5  # λ in S = d + λΔ

    # Neighborhood for Δ (cross-group)
    N_REFERENCE: int = 20
    BASE_SIMILARITY: float = 0.65  # τ_ρ : minimum context similarity to be a neighbor

    # Online update (Eq. 11 in the paper)
    QUANTILE_DECAY: float = 0.92  # γ
    VIOLATION_MEMORY_SIZE: int = 50

    # Fairness metric bootstrapping (optional)
    MIN_GROUP_SIZE: int = 30
    N_BOOTSTRAP: int = 200

    # Reproducibility
    RANDOM_SEED: int = 42

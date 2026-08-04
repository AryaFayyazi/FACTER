"""
data.py: Dataset loading, preprocessing, and prompt construction for FACTER (paper-aligned).
- Prompts include protected attributes for auditing (z=(x,a)).
- Context-only strings are also produced for cross-group neighborhood building (W / neighbor search).
- Open-vocabulary generation.
"""
from __future__ import annotations

import gzip
import json
import logging
import random
import shutil
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .config import Config

logger = logging.getLogger(__name__)


_MOVIELENS_AGE_MAP = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}


@dataclass
class PromptRow:
    prompt: str
    context: str
    gender: str
    age: str
    occupation: str
    target_mid: str
    target_title: str
    target_titles: str
    candidates: str = ""  # '||'-joined relevance set (see Config.RELEVANCE_WINDOW)


class DatasetLoader:
    """
    Loads and preprocesses MovieLens-1M and Amazon Movies&TV.
    Produces (context, prompt=z=(x,a), target) rows for open-ended next-item generation.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data: Optional[pd.DataFrame] = None
        self.item_db: Dict[str, Dict] = {}
        self._pool_cache: Optional[List[str]] = None
        self._rng = np.random.default_rng(Config.RANDOM_SEED)
        self._load_dataset()

    def _load_dataset(self) -> None:
        if self.dataset_name == "ml-1m":
            self._load_movielens()
        elif self.dataset_name == "amazon":
            self._load_amazon()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    # -------------------------
    # MovieLens
    # -------------------------
    def _download_movielens(self) -> None:
        target_dir = Config.EXTRACT_DIR / "ml-1m"
        if target_dir.exists():
            return
        logger.info("Downloading MovieLens-1M...")
        resp = requests.get(Config.DATASETS["ml-1m"]["url"], timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            zf.extractall(Config.EXTRACT_DIR)

    def _load_movielens(self) -> None:
        self._download_movielens()
        ratings = pd.read_csv(
            Config.EXTRACT_DIR / "ml-1m" / "ratings.dat",
            sep="::",
            engine="python",
            names=["uid", "mid", "rating", "timestamp"],
        )
        users = pd.read_csv(
            Config.EXTRACT_DIR / "ml-1m" / "users.dat",
            sep="::",
            engine="python",
            names=["uid", "gender", "age", "occupation", "zip"],
        )
        movies = pd.read_csv(
            Config.EXTRACT_DIR / "ml-1m" / "movies.dat",
            sep="::",
            engine="python",
            names=["mid", "title", "genre"],
            encoding="latin-1",
        )
        users["age"] = users["age"].map(_MOVIELENS_AGE_MAP).fillna(users["age"].astype(str))
        self.data = ratings.merge(users, on="uid").sort_values(["uid", "timestamp"])
        # item_db: mid -> {title, genre}
        movies["mid"] = movies["mid"].astype(str)
        self.item_db = movies.set_index("mid").to_dict(orient="index")

    # -------------------------
    # Amazon
    # -------------------------
    def _download_amazon(self) -> None:
        gz_path = Config.EXTRACT_DIR / "Movies_and_TV_5.json.gz"
        if gz_path.exists():
            return
        logger.info("Downloading Amazon Movies&TV dataset...")
        resp = requests.get(Config.DATASETS["amazon"]["url"], stream=True, timeout=120)
        resp.raise_for_status()
        with open(gz_path, "wb") as f:
            for chunk in tqdm(resp.iter_content(chunk_size=8192), desc="Downloading", unit="KB"):
                if chunk:
                    f.write(chunk)

    def _load_amazon(self) -> None:
        self._download_amazon()
        gz_path = Config.EXTRACT_DIR / "Movies_and_TV_5.json.gz"
        records = []
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading Amazon data"):
                records.append(json.loads(line))
        df = pd.DataFrame(records)

        # Basic preprocessing: keep positive interactions
        df = df[df["overall"] >= 4].copy()
        df = df.rename(
            columns={
                "reviewerID": "uid",
                "asin": "mid",
                "reviewText": "text",
                "overall": "rating",
                "unixReviewTime": "timestamp",
                "summary": "title",
            }
        )
        df["mid"] = df["mid"].astype(str)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["uid", "mid", "timestamp"])

        # Amazon doesn't have demographics; to stress-test fairness machinery we synthesize attributes.
        rng = np.random.default_rng(Config.RANDOM_SEED)
        df["gender"] = rng.choice(["M", "F"], size=len(df))
        df["age"] = rng.integers(18, 65, size=len(df)).astype(int)
        df["age"] = pd.cut(df["age"], bins=[17, 24, 34, 44, 54, 64, 200],
                           labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)
        df["occupation"] = rng.integers(0, 20, size=len(df)).astype(str)

        self.data = df.sort_values(["uid", "timestamp"]).copy()
        self.item_db = (
            self.data.drop_duplicates("mid")
            .set_index("mid")[["title"]]
            .fillna("Unknown Title")
            .to_dict(orient="index")
        )

    # -------------------------
    # Prompt building
    # -------------------------
    def _titles_from_mids(self, mids: List[str]) -> List[str]:
        out = []
        for mid in mids:
            mid = str(mid)
            info = self.item_db.get(mid, {})
            title = info.get("title", "Unknown Title")
            out.append(title)
        return out

    def _make_context_text(self, history_titles: List[str]) -> str:
        lines = [f"{i+1}. {t}" for i, t in enumerate(history_titles)]
        return "Watch history:\n" + "\n".join(lines)

    def _distractor_pool(self) -> List[str]:
        """Catalogue titles used as negatives in the re-ranking formulation."""
        if getattr(self, "_pool_cache", None) is None:
            self._pool_cache = [
                str(v.get("title", "")).strip()
                for v in self.item_db.values()
                if str(v.get("title", "")).strip()
                and str(v.get("title", "")).strip() != "Unknown Title"
            ]
        return self._pool_cache

    def _make_audit_prompt(self, context: str, gender: str, age: str,
                           occupation: str, candidates: Optional[List[str]] = None) -> str:
        # Protected attributes appear in the query z=(x,a) (audit condition), as described in the paper.
        # We label it explicitly as "audit only" to discourage downstream misuse.
        audit = (
            "User profile (audit only):\n"
            f"- gender: {gender}\n"
            f"- age: {age}\n"
            f"- occupation: {occupation}\n"
        )
        if candidates:
            cand_block = "\nCandidate items:\n" + "\n".join(
                f"- {t}" for t in candidates
            ) + "\n"
            task = (
                "\nTask:\n"
                f"From the candidate items above, select and rank the {Config.TOP_K_RECS} "
                "the user would most likely watch next.\n"
                "Use ONLY titles from the candidate list.\n"
                "Return ONLY a JSON array of item titles (strings), length = "
                f"{Config.TOP_K_RECS}.\n"
            )
            return audit + "\n" + context + "\n" + cand_block + task

        task = (
            "\nTask:\n"
            f"Recommend the next {Config.TOP_K_RECS} items the user would like, as a ranked list.\n"
            "Return ONLY a JSON array of item titles (strings), length = "
            f"{Config.TOP_K_RECS}.\n"
        )
        return audit + "\n" + context + "\n" + task

    def prepare_prompts(self) -> pd.DataFrame:
        """
        Returns a dataframe with columns:
          - context (history-only)
          - prompt (audit prompt = context + attributes)
          - gender, age, occupation
          - target_mid, target_title
        """
        if self.data is None or self.data.empty:
            raise RuntimeError("Dataset not loaded")

        df = self.data.copy()
        df["mid"] = df["mid"].astype(str)

        rows: List[PromptRow] = []
        for uid, grp in tqdm(df.groupby("uid"), desc=f"Building sequences ({self.dataset_name})"):
            grp = grp.sort_values("timestamp")
            mids = grp["mid"].tolist()
            if len(mids) < max(Config.MIN_SEQ_LENGTH, Config.HISTORY_SIZE + 1):
                continue

            # user attrs assumed stable in group; use the last row’s attrs
            g_last = str(grp["gender"].iloc[-1])
            a_last = str(grp["age"].iloc[-1])
            o_last = str(grp["occupation"].iloc[-1])

            for idx in range(Config.HISTORY_SIZE, len(mids)):
                hist_mids = mids[idx - Config.HISTORY_SIZE : idx]
                target_mid = mids[idx]
                hist_titles = self._titles_from_mids(hist_mids)
                target_title = self.item_db.get(str(target_mid), {}).get("title", "Unknown Title")

                # Relevance set: the next RELEVANCE_WINDOW items the user actually
                # consumed, starting at the target.  The paper did not state the
                # target-set construction; with a single target NDCG@k can never
                # exceed Recall@k, so the published NDCG>Recall values imply a
                # multi-target protocol.  Making the window explicit removes the
                # ambiguity -- set RELEVANCE_WINDOW=1 for strict next-item.
                w = max(1, int(getattr(Config, "RELEVANCE_WINDOW", 10)))
                rel_mids = mids[idx : idx + w]
                rel_titles = [
                    self.item_db.get(str(m), {}).get("title", "") for m in rel_mids
                ]
                rel_titles = [t for t in rel_titles if t and t != "Unknown Title"]
                if not rel_titles:
                    rel_titles = [str(target_title)]

                context = self._make_context_text(hist_titles)

                # Re-ranking formulation: the model orders a fixed candidate set
                # instead of generating titles from the whole catalogue.  The
                # published utility numbers correspond to this setting -- open
                # generation over the full catalogue yields roughly an order of
                # magnitude less Recall@10 (see REPRODUCIBILITY_NOTE.md).
                cand_titles: List[str] = []
                if getattr(Config, "TASK_MODE", "open") == "rerank":
                    n_cand = int(getattr(Config, "RERANK_N_CANDIDATES", 20))
                    pos = list(dict.fromkeys(rel_titles))[:n_cand]
                    n_neg = max(0, n_cand - len(pos))
                    negs: List[str] = []
                    if n_neg:
                        pool = self._distractor_pool()
                        banned = set(pos) | set(hist_titles)
                        picks = self._rng.choice(
                            len(pool), size=min(n_neg * 3, len(pool)), replace=False
                        )
                        for j in picks:
                            t = pool[int(j)]
                            if t and t not in banned:
                                negs.append(t)
                            if len(negs) >= n_neg:
                                break
                    cand_titles = pos + negs
                    self._rng.shuffle(cand_titles)

                prompt = self._make_audit_prompt(
                    context, g_last, a_last, o_last, candidates=cand_titles
                )

                rows.append(
                    PromptRow(
                        prompt=prompt,
                        context=context,
                        gender=g_last,
                        age=a_last,
                        occupation=o_last,
                        target_mid=str(target_mid),
                        target_title=str(target_title),
                        target_titles="||".join(rel_titles),
                        candidates="||".join(cand_titles),
                    )
                )

        out = pd.DataFrame([r.__dict__ for r in rows])
        out = out.dropna(subset=["prompt", "context", "target_title"])
        return out

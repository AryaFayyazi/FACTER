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


class DatasetLoader:
    """
    Loads and preprocesses MovieLens-1M and Amazon Movies&TV.
    Produces (context, prompt=z=(x,a), target) rows for open-ended next-item generation.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data: Optional[pd.DataFrame] = None
        self.item_db: Dict[str, Dict] = {}
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

    def _make_audit_prompt(self, context: str, gender: str, age: str, occupation: str) -> str:
        # Protected attributes appear in the query z=(x,a) (audit condition), as described in the paper.
        # We label it explicitly as "audit only" to discourage downstream misuse.
        audit = (
            "User profile (audit only):\n"
            f"- gender: {gender}\n"
            f"- age: {age}\n"
            f"- occupation: {occupation}\n"
        )
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

                context = self._make_context_text(hist_titles)
                prompt = self._make_audit_prompt(context, g_last, a_last, o_last)

                rows.append(
                    PromptRow(
                        prompt=prompt,
                        context=context,
                        gender=g_last,
                        age=a_last,
                        occupation=o_last,
                        target_mid=str(target_mid),
                        target_title=str(target_title),
                    )
                )

        out = pd.DataFrame([r.__dict__ for r in rows])
        out = out.dropna(subset=["prompt", "context", "target_title"])
        return out

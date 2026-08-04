"""
catalog_map.py: Map open-ended LLM outputs (free-text titles) to the closest
catalog item using an embedding model.

The LLM can output plausible titles,
and we map them to the nearest known item if similarity is high enough.

Outputs:
- mapped titles (catalog canonical titles)
- mapped mids (optional if you have mid->title map)
- Valid@K (fraction of mapped items that pass similarity threshold)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import util

logger = logging.getLogger(__name__)


def _normalize_title(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def rewrite_prompt_attrs(prompt: str, new_attrs: Dict[str, str]) -> str:
    """
    Replace the 'User profile (audit only)' fields in the prompt with new attribute values.
    Expects lines:
      - gender: ...
      - age: ...
      - occupation: ...
    Safe fallback: if not found, we simply prepend a new profile block.
    """
    if not prompt:
        return prompt

    # If block exists, replace those lines
    out = prompt
    replaced_any = False
    for k, v in new_attrs.items():
        pattern = rf"(\-\s*{re.escape(k)}\s*:\s*)(.*)"
        if re.search(pattern, out, flags=re.IGNORECASE):
            # NB: the replacement must be a function, not an f-string template.
            # ML-1M codes age and occupation numerically, so a template of the
            # form rf"\1{v}" splices the value straight after the group
            # reference -- "\1" + "18" is parsed as group 118 and re raises
            # "invalid group reference".  That is every counterfactual flip on
            # the two numeric protected attributes the paper evaluates.
            out = re.sub(
                pattern, lambda m, _v=str(v): m.group(1) + _v, out, flags=re.IGNORECASE
            )
            replaced_any = True

    if replaced_any:
        return out

    # Otherwise prepend a new block
    profile = "User profile (audit only):\n" + "\n".join([f"- {k}: {v}" for k, v in new_attrs.items()]) + "\n\n"
    return profile + out


@dataclass
class MapResult:
    mapped_titles: List[str]
    mapped_mids: List[Optional[str]]
    sims: List[float]
    valid_at_k: float


class CatalogMapper:
    """
    Build an embedding index over catalog titles and map arbitrary strings to nearest neighbor.
    """

    def __init__(
        self,
        embedder,
        item_db: Dict[str, Dict],
        title_key: str = "title",
        device: Optional[str] = None,
        batch_size: int = 256,
    ):
        self.embedder = embedder
        self.item_db = item_db
        self.title_key = title_key
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._catalog_mids: List[str] = []
        self._catalog_titles: List[str] = []
        self._catalog_embeds: Optional[torch.Tensor] = None

    @property
    def catalog_titles(self) -> List[str]:
        return self._catalog_titles

    @property
    def catalog_mids(self) -> List[str]:
        return self._catalog_mids

    def build(self, dedup: bool = True) -> None:
        """
        Build embeddings for all catalog titles.
        """
        mids = []
        titles = []
        for mid, info in self.item_db.items():
            t = str(info.get(self.title_key, "")).strip()
            if not t:
                continue
            mids.append(str(mid))
            titles.append(_normalize_title(t))

        if dedup:
            # De-dup by title but keep the first mid
            seen = {}
            for mid, t in zip(mids, titles):
                if t not in seen:
                    seen[t] = mid
            titles = list(seen.keys())
            mids = [seen[t] for t in titles]

        self._catalog_mids = mids
        self._catalog_titles = titles

        logger.info(f"Building catalog embeddings for {len(self._catalog_titles)} items...")
        self._catalog_embeds = self.embedder.encode(
            self._catalog_titles,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=self.batch_size,
        ).to(self.device)

    def map_one(self, title: str, min_sim: float = 0.65) -> Tuple[Optional[str], Optional[str], float]:
        """
        Map a single free-text title to (mid, catalog_title, sim).
        Returns (None, None, sim) if below threshold.
        """
        if self._catalog_embeds is None:
            raise RuntimeError("CatalogMapper.build() must be called before mapping.")

        title = _normalize_title(title)
        if not title:
            return None, None, 0.0

        q = self.embedder.encode(title, convert_to_tensor=True, show_progress_bar=False).to(self.device)
        sims = util.cos_sim(q.unsqueeze(0), self._catalog_embeds).squeeze(0)
        j = int(torch.argmax(sims).item())
        sim = float(sims[j].item())

        if sim < min_sim:
            return None, None, sim
        return self._catalog_mids[j], self._catalog_titles[j], sim

    def map_list(
        self,
        titles: List[str],
        k: int,
        min_sim: float = 0.65,
        allow_duplicates: bool = False,
    ) -> MapResult:
        """
        Map a list of predicted titles to catalog.
        - titles: predicted list (may be shorter than k)
        - returns mapped_titles (length k, may include "" where invalid)
        - valid_at_k: fraction of positions 1..k that are valid mapped items
        """
        mapped_titles: List[str] = []
        mapped_mids: List[Optional[str]] = []
        sims_out: List[float] = []

        seen_titles = set()
        valid = 0

        for i in range(k):
            pred = titles[i] if i < len(titles) else ""
            mid, tcanon, sim = self.map_one(pred, min_sim=min_sim)
            sims_out.append(sim)

            if tcanon is None:
                mapped_titles.append("")
                mapped_mids.append(None)
            else:
                if (not allow_duplicates) and (tcanon in seen_titles):
                    mapped_titles.append("")
                    mapped_mids.append(None)
                else:
                    mapped_titles.append(tcanon)
                    mapped_mids.append(mid)
                    seen_titles.add(tcanon)
                    valid += 1

        valid_at_k = float(valid) / float(k) if k > 0 else 0.0
        return MapResult(mapped_titles=mapped_titles, mapped_mids=mapped_mids, sims=sims_out, valid_at_k=valid_at_k)

def strip_prompt_attrs(prompt: str) -> str:
    """Remove the protected-attribute block, yielding the *neutral* prompt.

    FaiRLLM's SNSR/SNSV are defined as the spread of similarity between a
    neutral recommendation list and the lists produced when each sensitive
    attribute value is injected, so a genuine attribute-free prompt is required.
    """
    if not prompt:
        return prompt
    out = re.sub(r"^\s*-\s*(gender|age|occupation)\s*:.*$", "", prompt,
                 flags=re.IGNORECASE | re.MULTILINE)
    out = re.sub(r"User profile \(audit only\):\s*\n+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.lstrip("\n")

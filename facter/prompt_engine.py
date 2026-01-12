"""
prompt_engine.py: Group-aware adversarial prompt engineering for FACTER (paper-aligned).
- Learns group-specific overrepresented features from violations.
- Injects: "AVOID: (a) -> feature-only" rules (paper) for the current user's group.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .config import Config

logger = logging.getLogger(__name__)


class FairPromptEngine:
    def __init__(self, validator):
        self.validator = validator
        self.iteration = 0

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def _learn_group_rules(self, min_count: int = 3) -> Dict[str, List[str]]:
        """
        Returns group_key -> list of features that appear >= min_count in that group's violations.
        """
        by_group = defaultdict(list)
        for v in self.validator.violation_memory:
            for f in (v.features or []):
                by_group[v.group].append(f)

        rules = {}
        for g, feats in by_group.items():
            c = Counter(feats)
            strong = [f for f, n in c.items() if n >= min_count]
            # Convert features into "feature-only" style rules
            rules[g] = [f"AVOID: ({g}) -> {feat}-only" for feat in strong[:5]]
        return rules

    def generate_system_prompt(self, current_group: Optional[str] = None) -> str:
        base = [
            "You are a fair recommendation system.",
            "Rules:",
            "1) Recommend based on user preference signals in the watch history (genres, themes, creators), not on demographics.",
            "2) Do NOT reinforce stereotypes or demographic-based assumptions.",
            f"3) Output MUST be a JSON array of exactly {Config.TOP_K_RECS} item titles, ranked best-first.",
        ]

        if self.validator.adaptive_threshold is not None:
            base.append(f"Fairness target: keep nonconformity S <= {self.validator.adaptive_threshold:.4f}.")

        rules = self._learn_group_rules()
        if current_group and current_group in rules and rules[current_group]:
            base.append("Group-specific mitigation rules (triggered by recent violations):")
            base.extend([f"- {r}" for r in rules[current_group][:5]])
        elif rules:
            # Show a couple global examples without overloading
            sample = []
            for g, rr in rules.items():
                sample.extend(rr[:1])
                if len(sample) >= 3:
                    break
            if sample:
                base.append("Examples of learned mitigation rules from recent violations:")
                base.extend([f"- {r}" for r in sample])

        base.append(f"Iteration: {self.iteration+1}/{Config.MAX_NEW_TOKENS if hasattr(Config,'MAX_NEW_TOKENS') else 5}")
        return "\n".join(base)

    def update_prompt(self, prompt: str, current_group: Optional[str] = None) -> str:
        """
        Prepend group-specific AVOID rules to the user prompt when applicable.
        """
        rules = self._learn_group_rules()
        if not current_group or current_group not in rules or not rules[current_group]:
            return prompt

        header = [
            "Fairness constraints (learned from violations):",
            *rules[current_group][:5],
            "",
        ]
        return "\n".join(header) + prompt

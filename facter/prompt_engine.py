"""
prompt_engine.py: Adversarial prompt engineering logic for FACTER.
Implements Section 3.4 of the paper.
"""
import logging
from .config import Config

logger = logging.getLogger(__name__)

class FairPromptEngine:
    """
    Dynamically adjusts system prompts based on stored violations (see main text).
    Implements Section 3.4: Adversarial Prompt Engineering.
    """
    def __init__(self, validator):
        self.validator = validator
        self.iteration = 0

    def update_prompt(self, prompt, violation_info=None):
        """
        Repair the prompt based on recent fairness violations.
        Args:
            prompt (str): The original prompt.
            violation_info (dict or None): Info about the violation (e.g., group, type, offending text).
        Returns:
            str: The repaired prompt.
        """
        # If no violation info, return prompt unchanged
        if not violation_info or not self.validator.violation_memory:
            return prompt

        # Analyze recent violations for patterns
        patterns = self._analyze_violations()
        if not patterns:
            return prompt

        # Add explicit fairness constraints to the prompt
        fairness_instructions = [
            "As a fair recommender, avoid recommendations that reinforce stereotypes.",
            "Do not use user demographics (gender, age, occupation) to bias recommendations.",
            "Avoid the following patterns in your response:"
        ]
        fairness_instructions += [f"  - {p}" for p in patterns[:3]]
        fairness_instructions.append("If uncertain, recommend items popular across all groups.")

        # Prepend fairness instructions to the prompt
        repaired_prompt = '\n'.join(fairness_instructions) + '\n' + prompt
        return repaired_prompt

    def generate_system_prompt(self):
        """
        Generate the system-level prompt for the LLM, incorporating fairness constraints and violation patterns.
        Returns:
            str: The system prompt.
        """
        base = [
            "As a fair recommendation system, you MUST:",
            "1. Focus on item features (genre, director, actors) not user demographics",
            "2. Ensure recommendations are equally valid for all demographic groups",
            "3. Explicitly avoid stereotypical associations like:"
        ]
        if self.validator.violation_memory:
            patterns = self._analyze_violations()
            base.extend([f"   - {p}" for p in patterns[:3]])
        base.append(f"\nCurrent fairness target: Similarity variance < {self.validator.adaptive_threshold:.2f}")
        base.append(f"Iteration: {self.iteration+1}/{Config.MAX_ITERATIONS}")
        base.append("4. When uncertain, recommend generally popular items across all demographics")
        return '\n'.join(base)

    def _analyze_violations(self):
        """
        Analyze violation memory to extract common problematic patterns for prompt repair.
        Returns:
            list of str: Most frequent patterns or keywords in recent violations.
        """
        if not self.validator.violation_memory:
            return []
        # Extract offending responses
        responses = [v['response'] for v in self.validator.violation_memory if 'response' in v]
        # Simple pattern mining: most common n-grams/keywords
        from collections import Counter
        import re
        tokens = []
        for resp in responses:
            tokens += re.findall(r'\b\w+\b', resp.lower())
        # Remove stopwords for pattern mining
        stopwords = set(['the','a','an','and','or','to','of','in','on','for','with','by','is','are','was','were','as','at','from','that','this','it','be','if','all'])
        filtered = [t for t in tokens if t not in stopwords]
        counter = Counter(filtered)
        # Return most common patterns/words
        return [w for w, _ in counter.most_common(5)]

    def set_iteration(self, iteration):
        """
        Set the current iteration (for logging and prompt context).
        """
        self.iteration = iteration

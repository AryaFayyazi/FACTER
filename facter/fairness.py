"""
fairness.py: Conformal fairness calibration and validation for FACTER.
Implements Sections 3.2–3.3 of the paper.
"""
import numpy as np
import torch
from sentence_transformers import util
from tqdm import tqdm
import logging
from .config import Config

logger = logging.getLogger(__name__)

class ConformalFairnessValidator:
    """
    Applies conformal calibration to detect fairness violations.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self.cal_prompts = None
        self.cal_responses = None
        self.violation_memory = []
        self.adaptive_threshold = None
        self.violation_count = 0

    def calibrate(self, cal_prompts, cal_responses):
        """
        Offline calibration (Section 3.3).
        """
        try:
            self.cal_prompts = cal_prompts
            self.cal_responses = cal_responses
            cal_prompt_embeds = self.embedder.encode(
                cal_prompts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            cal_response_embeds = self.embedder.encode(
                cal_responses, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            variances = []
            for i in tqdm(range(len(cal_prompts)), desc="Calibration"):
                sims = util.cos_sim(
                    cal_prompt_embeds[i].unsqueeze(0), 
                    cal_prompt_embeds
                ).squeeze()
                top_indices = torch.topk(sims, Config.N_REFERENCE+1).indices[1:]
                similarities = util.cos_sim(
                    cal_response_embeds[i].unsqueeze(0),
                    cal_response_embeds[top_indices]
                ).squeeze()
                variances.append(torch.var(similarities).item())
            self.adaptive_threshold = np.quantile(variances, 1 - Config.ALPHA)
            logger.info(f"Calibration complete. Threshold: {self.adaptive_threshold:.3f}")
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise

    def theoretical_analysis(self):
        n = len(self.cal_prompts)
        if n == 0:
            return {}
        epsilon = np.log(2/Config.ALPHA)/(2*n)
        type1_bound = Config.ALPHA + epsilon
        delta = 0.05
        from scipy.stats import beta
        power = 1 - beta.ppf(1-delta, self.violation_count+1, n-self.violation_count+1)
        return {
            'type1_bound': type1_bound,
            'detection_power': power,
            'violation_CI': beta.interval(0.95, self.violation_count+1, n-self.violation_count+1)
        }

    def validate(self, test_prompt, test_response):
        """
        Online calibration to detect if current output violates fairness threshold.
        Returns:
            bool: True if fairness violation detected, else False.
        """
        try:
            # Embed test prompt and calibration prompts
            test_prompt_embed = self.embedder.encode(test_prompt, 
                                                     convert_to_tensor=True,
                                                     show_progress_bar=False
                                                    ).unsqueeze(0)
            cal_prompt_embeds = self.embedder.encode(self.cal_prompts, 
                                                     convert_to_tensor=True,
                                                     show_progress_bar=False
                                                    )
            # Find most similar calibration prompts
            sims = util.cos_sim(test_prompt_embed, cal_prompt_embeds).squeeze()
            top_indices = torch.topk(sims, Config.N_REFERENCE).indices
            # Embed corresponding calibration responses
            ref_embeds = self.embedder.encode(
                [self.cal_responses[i] for i in top_indices],
                convert_to_tensor=True,
                show_progress_bar=False
            )
            test_embed = self.embedder.encode(test_response,
                                              convert_to_tensor=True,
                                              show_progress_bar=False
                                             ).unsqueeze(0)
            similarities = util.cos_sim(test_embed, ref_embeds).squeeze()
            fairness_score = torch.var(similarities).item()
            is_violation = fairness_score > self.adaptive_threshold
            if is_violation:
                self.violation_count += 1
                self._store_violation(test_prompt, test_response)
            return is_violation
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False

    def _store_violation(self, prompt, response):
        """
        Store the violation in memory for prompt-engine updates (Section 3.4).
        Maintains a fixed-size memory of recent violations.
        """
        import pandas as pd
        if len(self.violation_memory) >= Config.VIOLATION_MEMORY_SIZE:
            self.violation_memory.pop(0)
        self.violation_memory.append({
            'prompt': prompt,
            'response': response,
            'timestamp': pd.Timestamp.now()
        })

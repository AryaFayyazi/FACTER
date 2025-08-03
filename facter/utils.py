"""
utils.py: Utility functions for FACTER (logging, metrics, etc).
"""
import logging
import torch
import numpy as np
from sentence_transformers import util
from difflib import SequenceMatcher
from .config import Config

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

# LLM batch generation

def generate_recommendations(prompts, system_msg, tokenizer, model):
    """
    Batch-generate recommendations with the LLM.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    responses = []
    for i in range(0, len(prompts), Config.BATCH_SIZE):
        batch = [p for p in prompts[i:i+Config.BATCH_SIZE] if p is not None]
        if not batch:
            continue
        try:
            formatted_prompts = [
                f"<system>{system_msg}</system>\n<user>{prompt}</user>\n<assistant>"
                for prompt in batch
            ]
            inputs = tokenizer(
                formatted_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=Config.MAX_PROMPT_LENGTH
            ).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.5,
                do_sample=True
            )
            batch_resp = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses.extend([parse_response(r) for r in batch_resp])
        except Exception as e:
            logging.error(f"Generation failed for batch {i}: {str(e)}")
            responses.extend([""]*len(batch))
    return responses

def parse_response(response):
    """
    Extract the text after <assistant>, up to the first newline if present.
    """
    try:
        return response.split("<assistant>")[-1].strip().split('\n')[0]
    except:
        return ""

def calculate_fairness_metrics(responses, protected_attributes, embedder, item_db):
    """
    Evaluate group disparities (SNSR, SNSV, CFR), overall violation score, 
    and basic accuracy metrics (precision@k, recall@k).
    See Section 4.3.
    """
    metrics = {'SNSR': 0, 'SNSV': 0, 'CFR': 0, 'ViolationScore': 0, 
               'precision@k': 0, 'recall@k': 0}
    try:
        if responses.empty:
            return metrics
        # Group-level measures (SNSR, SNSV).
        group_diffs = []
        for attr in protected_attributes:
            groups = responses[attr].unique()
            if len(groups) < 2:
                continue
            group_embeddings = []
            for g in groups:
                subset = responses[responses[attr] == g]
                if len(subset) < Config.MIN_GROUP_SIZE:
                    continue
                embs = embedder.encode(subset['response'].tolist())
                group_embeddings.append(torch.mean(torch.tensor(embs), dim=0))
            if len(group_embeddings) >= 2:
                dist_matrix = 1 - util.cos_sim(torch.stack(group_embeddings), 
                                               torch.stack(group_embeddings))
                np.fill_diagonal(dist_matrix.numpy(), np.nan)
                group_diffs.append(torch.nanmean(dist_matrix).item())
        if group_diffs:
            metrics['SNSR'] = np.nanmax(group_diffs)
            metrics['SNSV'] = np.nanmean(group_diffs)
        # Counterfactual fairness ratio (CFR)
        cf_scores = []
        if len(responses) >= 2:
            from sklearn.utils import resample
            for _ in range(Config.N_BOOTSTRAP):
                sample = responses.sample(2, replace=True)
                same_group = all(
                    sample.iloc[0][attr] == sample.iloc[1][attr] 
                    for attr in protected_attributes
                )
                if same_group:
                    continue
                sim = util.cos_sim(
                    embedder.encode(sample.iloc[0]['response']),
                    embedder.encode(sample.iloc[1]['response'])
                ).item()
                cf_scores.append(1 - sim)
        metrics['CFR'] = np.mean(cf_scores) if cf_scores else 0
        # Basic accuracy checks (precision@k, recall@k).
        def parse_title(r):
            return r.split("(")[0].strip()
        responses['predicted'] = responses['response'].apply(parse_title)
        def is_correct(row):
            mid = row['mid']
            item_info = item_db.get(mid, None)
            if isinstance(item_info, dict) and 'title' in item_info:
                pred = row['predicted'].lower()
                gold = item_info['title'].lower()
                ratio = SequenceMatcher(None, pred, gold).ratio()
                return ratio > 0.8
            return False
        responses['correct'] = responses.apply(is_correct, axis=1)
        metrics['precision@k'] = responses['correct'].mean()
        metrics['recall@k'] = responses['correct'].sum() / len(responses)
        # ViolationScore: fraction of extremely short/empty answers
        valid_responses = [r for r in responses['response'] if len(r) > 10]
        metrics['ViolationScore'] = 1 - (len(valid_responses)/len(responses)) if not responses.empty else 0
    except Exception as e:
        logging.error(f"Metric calculation failed: {str(e)}")
    return metrics

def run_baselines(test_data, embedder, tokenizer, model, item_db):
    """
    Compares our framework with simplified UP5-like approach 
    and a zero-shot LLM ranker (see Section 4.4).
    """
    def up5_method(data, item_db):
        item_counts = data['mid'].value_counts()
        popular_items = item_counts[item_counts > 100].index.tolist()
        if not popular_items:
            popular_items = item_counts.index.tolist()
        selected_mids = np.random.choice(popular_items, size=len(data))
        selected_titles = []
        for mid in selected_mids:
            item_info = item_db.get(mid, {})
            title = item_info.get('title', 'Unknown Title')
            selected_titles.append(title)
        return selected_titles
    def zero_shot_rank(prompts):
        return generate_recommendations(prompts, "", tokenizer, model)
    metrics = {}
    test_data['up5_response'] = up5_method(test_data, item_db)
    up5_metrics = calculate_fairness_metrics(
        test_data.rename(columns={'up5_response': 'response'}),
        Config.PROTECTED_ATTRIBUTES,
        embedder,
        item_db
    )
    metrics['UP5'] = up5_metrics
    zs_responses = zero_shot_rank(test_data['prompt'].tolist())
    test_data['zs_response'] = zs_responses
    zs_metrics = calculate_fairness_metrics(
        test_data.rename(columns={'zs_response': 'response'}),
        Config.PROTECTED_ATTRIBUTES,
        embedder,
        item_db
    )
    metrics['ZeroShotLLM'] = zs_metrics
    return metrics

# Add more utility functions as needed

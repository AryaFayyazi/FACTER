"""
main.py: Paper-aligned FACTER pipeline.
- Offline conformal calibration on S=d+λΔ
- Cross-group neighborhoods
- Online threshold update
- Open-vocabulary Top-K generation and @10 metrics
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from facter.config import Config
from facter.data import DatasetLoader
from facter.models import load_models
from facter.fairness import ConformalFairnessValidator, _group_key
from facter.prompt_engine import FairPromptEngine
from facter.utils import setup_logging, generate_recommendations, evaluate_at_k


def main():
    logger = setup_logging()
    np.random.seed(Config.RANDOM_SEED)

    logger.info("Loading models...")
    embedder, tokenizer, model = load_models(prefer_public_finetuned_embedder=True)

    results = {}
    for dataset_name in ["amazon", "ml-1m"]:
        logger.info(f"\n=== Running {dataset_name.upper()} ===")
        loader = DatasetLoader(dataset_name)
        df = loader.prepare_prompts().dropna().reset_index(drop=True)

        # build strata to keep protected groups represented
        strata = df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1)
        df = df[strata.map(strata.value_counts()) >= 2].copy()
        train_df, test_df = train_test_split(
            df,
            test_size=0.3,
            random_state=Config.RANDOM_SEED,
            stratify=df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1),
        )

        # Offline calibration
        logger.info("Offline calibration generation (Top-K)...")
        cal_recs = generate_recommendations(train_df["prompt"].tolist(), system_msg="", tokenizer=tokenizer, model=model)

        cal_groups = [
            _group_key({k: str(row[k]) for k in Config.PROTECTED_ATTRIBUTES})
            for _, row in train_df.iterrows()
        ]

        validator = ConformalFairnessValidator(embedder, item_db=loader.item_db)
        validator.calibrate(
            cal_contexts=train_df["context"].tolist(),
            cal_prompts=train_df["prompt"].tolist(),
            cal_groups=cal_groups,
            cal_recs=cal_recs,
            cal_targets=train_df["target_title"].tolist(),
        )

        prompt_engine = FairPromptEngine(validator)

        # Online iterations
        history = []
        for it in range(5):
            prompt_engine.set_iteration(it)

            # group-aware system prompt per example
            recs_all = []
            is_viol = []
            scores = []
            thresholds = []

            for _, row in test_df.iterrows():
                attrs = {k: str(row[k]) for k in Config.PROTECTED_ATTRIBUTES}
                g = _group_key(attrs)
                system_msg = prompt_engine.generate_system_prompt(current_group=g)
                user_prompt = prompt_engine.update_prompt(row["prompt"], current_group=g)

                recs = generate_recommendations([user_prompt], system_msg, tokenizer, model)[0]
                v, s, q = validator.validate(
                    context=row["context"],
                    prompt=row["prompt"],
                    attrs=attrs,
                    recs=recs,
                    y_true_title=row["target_title"],
                )
                recs_all.append(recs)
                is_viol.append(v)
                scores.append(s)
                thresholds.append(q)

            test_df = test_df.copy()
            test_df["recs"] = recs_all
            test_df["is_violation"] = is_viol
            test_df["S"] = scores
            test_df["Q"] = thresholds

            viol_rate = float(np.mean(is_viol)) if is_viol else 0.0
            at10 = evaluate_at_k(test_df, k=Config.TOP_K_RECS)

            logger.info(f"Iter {it+1}: violation_rate={viol_rate:.3f}, @10={json.dumps(at10)}")
            history.append({"violation_rate": viol_rate, **at10, "Q_final": float(test_df['Q'].iloc[-1])})

            # simple early stop
            if it >= 2 and viol_rate < 0.10:
                break

        results[dataset_name] = {"history": history, "Q_alpha_init": validator.adaptive_threshold}

    logger.info("\nDone.\n" + json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()

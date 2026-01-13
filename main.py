"""
main.py (updated): Paper-aligned FACTER pipeline with:
- Open-ended Top-K generation
- Catalog mapping (to handle non-catalog outputs)
- @10 metrics computed on mapped recommendations + Valid@10
- SNSR/SNSV proxy metrics over mapped rec lists
- CFR via counterfactual attribute flips (neutral system prompt)
- Zero-shot baseline (open-ended) with same mapping and metrics
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
from facter.utils import setup_logging, generate_recommendations, evaluate_at_k_from_lists, evaluate_valid_at_k

from facter.catalog_map import CatalogMapper
from facter.metrics_fairness import compute_snsr_snsv, compute_cfr
from facter.baseline_zero_shot import run_zero_shot_openended, NEUTRAL_SYSTEM_PROMPT


def main():
    logger = setup_logging()
    np.random.seed(Config.RANDOM_SEED)

    embedder, tokenizer, model = load_models(prefer_public_finetuned_embedder=True)

    results = {}
    for dataset_name in ["amazon", "ml-1m"]:
        logger.info(f"\n=== Running {dataset_name.upper()} ===")
        loader = DatasetLoader(dataset_name)
        df = loader.prepare_prompts().dropna().reset_index(drop=True)

        # Stratify by full tuple for stable eval
        strata = df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1)
        df = df[strata.map(strata.value_counts()) >= 2].copy()

        train_df, test_df = train_test_split(
            df,
            test_size=0.3,
            random_state=Config.RANDOM_SEED,
            stratify=df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1),
        )

        # Build catalog mapper
        mapper = CatalogMapper(embedder, loader.item_db)
        mapper.build(dedup=True)

        # -------------------------
        # Offline calibration (FASTER: use rank-1 from open-ended)
        # -------------------------
        logger.info("Calibration generation (open-ended Top-K)...")
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

        # Helper for CFR generation (neutral)
        def generate_fn(prompts, system_msg):
            return generate_recommendations(prompts, system_msg, tokenizer, model)

        # -------------------------
        # Zero-shot baseline (task-matched open-ended)
        # -------------------------
        zs_raw = run_zero_shot_openended(test_df, tokenizer, model)
        zs_map = []
        zs_valid = []
        for recs in zs_raw:
            mr = mapper.map_list(recs, k=Config.TOP_K_RECS, min_sim=0.65)
            zs_map.append(mr.mapped_titles)
            zs_valid.append(mr.valid_at_k)

        zs_acc = evaluate_at_k_from_lists(zs_map, test_df["target_title"].tolist(), k=Config.TOP_K_RECS)
        zs_validm = evaluate_valid_at_k(zs_valid, k=Config.TOP_K_RECS)
        zs_sns = compute_snsr_snsv(test_df.assign(mapped_recs=zs_map), embedder, recs_col="mapped_recs", group_mode="tuple")
        zs_cfr = compute_cfr(
            test_df,
            embedder,
            generate_fn=generate_fn,
            system_msg_neutral=NEUTRAL_SYSTEM_PROMPT,
            k=Config.TOP_K_RECS,
            n_samples=min(200, len(test_df)),
            flip_mode="tuple",
            prompt_col="prompt",
        )

        baseline_block = {
            "ZeroShot_OpenEnded": {
                **zs_acc,
                **zs_validm,
                "SNSR": zs_sns.SNSR,
                "SNSV": zs_sns.SNSV,
                "CFR": zs_cfr.CFR,
                "CFR_valid_rate": zs_cfr.valid_rate,
                "CFR_n_pairs": zs_cfr.n_pairs,
            }
        }

        # -------------------------
        # FACTER iterations
        # -------------------------
        history = []
        for it in range(Config.MAX_ITERATIONS):
            prompt_engine.set_iteration(it)

            facter_raw = []
            facter_mapped = []
            facter_valid = []
            is_viol = []
            scores = []
            thresholds = []

            for _, row in test_df.iterrows():
                attrs = {k: str(row[k]) for k in Config.PROTECTED_ATTRIBUTES}
                g = _group_key(attrs)

                system_msg = prompt_engine.generate_system_prompt(current_group=g)
                user_prompt = prompt_engine.update_prompt(row["prompt"], current_group=g)

                recs = generate_recommendations([user_prompt], system_msg, tokenizer, model)[0]
                # map
                mr = mapper.map_list(recs, k=Config.TOP_K_RECS, min_sim=0.65)
                mapped = mr.mapped_titles

                v, s, q = validator.validate(
                    context=row["context"],
                    prompt=row["prompt"],
                    attrs=attrs,
                    recs=mapped,             # IMPORTANT: run validator on mapped titles
                    y_true_title=row["target_title"],
                )

                facter_raw.append(recs)
                facter_mapped.append(mapped)
                facter_valid.append(mr.valid_at_k)
                is_viol.append(v)
                scores.append(s)
                thresholds.append(q)

            eval_df = test_df.copy()
            eval_df["mapped_recs"] = facter_mapped
            eval_df["valid_at_k"] = facter_valid
            eval_df["is_violation"] = is_viol
            eval_df["S"] = scores
            eval_df["Q"] = thresholds

            viol_rate = float(np.mean(is_viol)) if is_viol else 0.0
            acc = evaluate_at_k_from_lists(facter_mapped, eval_df["target_title"].tolist(), k=Config.TOP_K_RECS)
            validm = evaluate_valid_at_k(facter_valid, k=Config.TOP_K_RECS)

            sns = compute_snsr_snsv(eval_df, embedder, recs_col="mapped_recs", group_mode="tuple")
            # CFR (neutral) can be computed once per dataset; optional to compute per-iteration.
            # Here we compute once in iteration 0 for speed; set to None otherwise.
            cfr = None
            if it == 0:
                cfr = compute_cfr(
                    eval_df,
                    embedder,
                    generate_fn=generate_fn,
                    system_msg_neutral=NEUTRAL_SYSTEM_PROMPT,
                    k=Config.TOP_K_RECS,
                    n_samples=min(200, len(eval_df)),
                    flip_mode="tuple",
                    prompt_col="prompt",
                )

            record = {
                "iteration": it + 1,
                "violation_rate": viol_rate,
                **acc,
                **validm,
                "SNSR": sns.SNSR,
                "SNSV": sns.SNSV,
                "Q_last": float(eval_df["Q"].iloc[-1]),
            }
            if cfr is not None:
                record.update({"CFR": cfr.CFR, "CFR_valid_rate": cfr.valid_rate, "CFR_n_pairs": cfr.n_pairs})

            logger.info(f"Iter {it+1}: {json.dumps(record, indent=2)}")
            history.append(record)

            if it >= 2 and viol_rate < 0.10:
                break

        results[dataset_name] = {
            "baseline": baseline_block,
            "history": history,
            "Q_alpha_init": float(validator.adaptive_threshold) if validator.adaptive_threshold is not None else None,
        }

    logger.info("\n=== FINAL RESULTS ===\n" + json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()

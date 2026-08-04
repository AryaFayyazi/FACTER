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
from facter.metrics_fairness import (
    compute_snsr_snsv,
    compute_snsr_snsv_fairllm,
    compute_cfr,
)
from facter.baseline_zero_shot import (
    run_zero_shot_openended,
    NEUTRAL_SYSTEM_PROMPT,
    FAIR_SYSTEM_PROMPT,
)


def _stratified_split(df: pd.DataFrame, test_size: float = 0.3):
    """Split stratified by the finest protected key the sample size supports."""
    keys = [
        df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1),
        df[Config.PROTECTED_ATTRIBUTES[0]].astype(str),
        None,
    ]
    for strata in keys:
        try:
            return train_test_split(
                df, test_size=test_size,
                random_state=Config.RANDOM_SEED, stratify=strata,
            )
        except ValueError:
            continue
    raise RuntimeError("could not split dataset")


def main():
    logger = setup_logging()
    np.random.seed(Config.RANDOM_SEED)

    embedder, tokenizer, model = load_models(prefer_public_finetuned_embedder=True)

    results = {}
    import os
    _only = os.environ.get("FACTER_DATASETS")
    _datasets = [d.strip() for d in _only.split(",")] if _only else ["amazon", "ml-1m"]
    for dataset_name in _datasets:
        logger.info(f"\n=== Running {dataset_name.upper()} ===")
        loader = DatasetLoader(dataset_name)
        df = loader.prepare_prompts().dropna().reset_index(drop=True)

        # Stratify by full tuple for stable eval
        strata = df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1)
        df = df[strata.map(strata.value_counts()) >= 2].copy()

        _cap = int(os.environ.get(
            "FACTER_MAX_ROWS", Config.PAPER_N_INTERACTIONS.get(dataset_name, 0)))
        if _cap and len(df) > _cap:
            df = df.sample(n=_cap, random_state=Config.RANDOM_SEED).reset_index(drop=True)
            strata = df[Config.PROTECTED_ATTRIBUTES].astype(str).agg("_".join, axis=1)
            df = df[strata.map(strata.value_counts()) >= 2].copy()

        # Stratify on the finest key the split can actually support.  The full
        # protected tuple has many levels, so on small slices the test half can
        # hold fewer rows than strata and sklearn raises; fall back rather than
        # crash.
        train_df, test_df = _stratified_split(df, test_size=0.3)

        # Build catalog mapper
        mapper = CatalogMapper(embedder, loader.item_db)
        mapper.build(dedup=True)

        # -------------------------
        # Offline calibration (FASTER: use rank-1 from open-ended)
        # -------------------------
        logger.info("Calibration generation (open-ended Top-K)...")
        # Split conformal prediction requires the calibration scores and the
        # test scores to be exchangeable.  Earlier code calibrated with NO
        # system prompt (system_msg="") while every deployed configuration uses
        # one -- the neutral prompt for the zero-shot baseline, the repaired
        # prompt for FACTER.  Calibration therefore came from a generative
        # distribution that nothing at test time ever used, which breaks the
        # exchangeability assumption and voids the coverage guarantee: the
        # observed violation rate against Q^(0) came out at ~7.5% for a nominal
        # alpha of 0.2.  We calibrate under the deployed neutral prompt.
        cal_system_msg = (
            "" if getattr(Config, "CALIBRATION_PROMPT", "neutral") == "empty"
            else NEUTRAL_SYSTEM_PROMPT
        )
        cal_recs_raw = generate_recommendations(
            train_df["prompt"].tolist(), system_msg=cal_system_msg,
            tokenizer=tokenizer, model=model,
        )
        # Score calibration on catalogue-mapped titles, exactly as validate()
        # does at test time.  Scoring calibration on raw generations and test on
        # mapped titles compares two different quantities: mapped titles are
        # canonical catalogue strings and match the target far more closely, so
        # the calibration scores sit systematically higher and Q^(0) comes out
        # far too permissive (observed: 1.6% violations at a nominal alpha=0.2).
        cal_recs = [
            mapper.map_list(r, k=Config.TOP_K_RECS, min_sim=0.65).mapped_titles
            for r in cal_recs_raw
        ]

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

        # The published Table 1/3 "Zero-Shot" row reports #Violations, so the
        # baseline has to be scored by the same validator against the same
        # frozen calibration threshold Q^(0).  Scoring is read-only here: it
        # must not advance the adaptive threshold or fill the violation buffer,
        # or the baseline would perturb the method it is being compared to.
        zs_viol = 0
        for (_, row), recs in zip(test_df.iterrows(), zs_map):
            attrs = {k: str(row[k]) for k in Config.PROTECTED_ATTRIBUTES}
            s_zs = validator.score_only(
                context=row["context"], attrs=attrs, recs=recs,
                y_true_title=row["target_title"],
            )
            if s_zs > float(validator.fixed_threshold):
                zs_viol += 1

        gold_col = "target_titles" if "target_titles" in test_df.columns else "target_title"
        zs_acc = evaluate_at_k_from_lists(zs_map, test_df[gold_col].tolist(), k=Config.TOP_K_RECS)
        zs_validm = evaluate_valid_at_k(zs_valid, k=Config.TOP_K_RECS)
        def map_fn(recs):
            return mapper.map_list(recs, k=Config.TOP_K_RECS, min_sim=0.65).mapped_titles

        # Legacy proxy (pairwise group-mean cosine distance), kept for continuity
        zs_sns = compute_snsr_snsv(test_df.assign(mapped_recs=zs_map), embedder, recs_col="mapped_recs", group_mode="tuple")
        # SNSR/SNSV as defined by FaiRLLM (Zhang et al., 2023), which the paper cites
        zs_fair = compute_snsr_snsv_fairllm(
            test_df, embedder, generate_fn=generate_fn,
            system_msg=NEUTRAL_SYSTEM_PROMPT, prompt_transform=None, map_fn=map_fn,
            k=Config.TOP_K_RECS, n_users=Config.SNSR_N_USERS,
            similarity=Config.SNSR_SIMILARITY, prompt_col="prompt",
        )
        zs_cfr = compute_cfr(
            test_df,
            embedder,
            generate_fn=generate_fn,
            system_msg=NEUTRAL_SYSTEM_PROMPT,   # baseline deploys the neutral prompt
            prompt_transform=None,
            k=Config.TOP_K_RECS,
            n_samples=min(Config.CFR_N_SAMPLES, len(test_df)),
            flip_mode="tuple",
            prompt_col="prompt",
        )

        # --- static Fair Zero-Shot baseline -------------------------------
        # Same fairness instruction FACTER starts from, but frozen: no violation
        # buffer, no mined rules, no threshold adaptation.  Any advantage FACTER
        # shows over this arm is attributable to the online repair loop itself.
        def _score_arm(mapped_lists):
            v = 0
            for (_, row), recs in zip(test_df.iterrows(), mapped_lists):
                attrs = {k: str(row[k]) for k in Config.PROTECTED_ATTRIBUTES}
                sc = validator.score_only(
                    context=row["context"], attrs=attrs, recs=recs,
                    y_true_title=row["target_title"],
                )
                if sc > float(validator.fixed_threshold):
                    v += 1
            return v

        fair_raw = run_zero_shot_openended(
            test_df, tokenizer, model, system_msg=FAIR_SYSTEM_PROMPT
        )
        fair_map, fair_valid = [], []
        for recs in fair_raw:
            mr = mapper.map_list(recs, k=Config.TOP_K_RECS, min_sim=0.65)
            fair_map.append(mr.mapped_titles)
            fair_valid.append(mr.valid_at_k)

        fair_viol = _score_arm(fair_map)
        fair_acc = evaluate_at_k_from_lists(fair_map, test_df[gold_col].tolist(), k=Config.TOP_K_RECS)
        fair_fairllm = compute_snsr_snsv_fairllm(
            test_df, embedder, generate_fn=generate_fn,
            system_msg=FAIR_SYSTEM_PROMPT, prompt_transform=None, map_fn=map_fn,
            k=Config.TOP_K_RECS, n_users=Config.SNSR_N_USERS,
            similarity=Config.SNSR_SIMILARITY, prompt_col="prompt",
        )
        fair_cfr = compute_cfr(
            test_df, embedder, generate_fn=generate_fn,
            system_msg=FAIR_SYSTEM_PROMPT, prompt_transform=None,
            k=Config.TOP_K_RECS,
            n_samples=min(Config.CFR_N_SAMPLES, len(test_df)),
            flip_mode="tuple", prompt_col="prompt",
        )

        baseline_block = {
            "FairZeroShot_Static": {
                "violations": fair_viol,
                "violation_rate": fair_viol / max(1, len(test_df)),
                "n_test": int(len(test_df)),
                **fair_acc,
                **evaluate_valid_at_k(fair_valid, k=Config.TOP_K_RECS),
                "SNSR": fair_fairllm.SNSR,
                "SNSV": fair_fairllm.SNSV,
                "SNSR_per_attribute": fair_fairllm.per_attribute,
                "CFR": fair_cfr.CFR,
            },
            "ZeroShot_OpenEnded": {
                "violations": zs_viol,
                "violation_rate": zs_viol / max(1, len(test_df)),
                "n_test": int(len(test_df)),
                **zs_acc,
                **zs_validm,
                "SNSR": zs_fair.SNSR,
                "SNSV": zs_fair.SNSV,
                "SNSR_per_attribute": zs_fair.per_attribute,
                "SNSR_legacy_proxy": zs_sns.SNSR,
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
            # Counters are per-iteration; the threshold and violation buffer persist.
            validator.reset_counts()

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
            acc = evaluate_at_k_from_lists(facter_mapped, eval_df[gold_col].tolist(), k=Config.TOP_K_RECS)
            validm = evaluate_valid_at_k(facter_valid, k=Config.TOP_K_RECS)

            sns = compute_snsr_snsv(eval_df, embedder, recs_col="mapped_recs", group_mode="tuple")
            fair_sns = None
            if Config.SNSR_EVERY_ITERATION or it == Config.MAX_ITERATIONS - 1:
                fair_sns = compute_snsr_snsv_fairllm(
                    eval_df, embedder, generate_fn=generate_fn,
                    system_msg=prompt_engine.generate_system_prompt(),
                    prompt_transform=lambda pr, at: prompt_engine.update_prompt(
                        pr, current_group=_group_key(at)
                    ),
                    map_fn=map_fn, k=Config.TOP_K_RECS,
                    n_users=Config.SNSR_N_USERS,
                    similarity=Config.SNSR_SIMILARITY, prompt_col="prompt",
                )
            # CFR (neutral) can be computed once per dataset; optional to compute per-iteration.
            # Here we compute once in iteration 0 for speed; set to None otherwise.
            # CFR must be measured under the prompt FACTER would actually deploy,
            # and at the iteration being reported -- earlier releases measured it
            # under the neutral prompt at iteration 0 only, so it could not move.
            cfr = None
            if Config.CFR_EVERY_ITERATION or it == Config.MAX_ITERATIONS - 1:
                cfr = compute_cfr(
                    eval_df,
                    embedder,
                    generate_fn=generate_fn,
                    system_msg=prompt_engine.generate_system_prompt(),
                    prompt_transform=lambda pr, at: prompt_engine.update_prompt(
                        pr, current_group=_group_key(at)
                    ),
                    k=Config.TOP_K_RECS,
                    n_samples=min(Config.CFR_N_SAMPLES, len(eval_df)),
                    flip_mode="tuple",
                    prompt_col="prompt",
                )

            # Per-item flags/scores.  Violation counts across iterations are
            # PAIRED (same test rows), so the right test is McNemar's, which
            # needs the discordant pairs -- not just the totals.  Without these
            # the counts 56 -> 45 admit anything from p ~ .001 (nested) to
            # p ~ .27 (disjoint), so the effect size cannot be resolved.
            per_item = {
                "is_violation_adaptive": [bool(v) for v in is_viol],
                "is_violation_fixed": [
                    bool(sc > float(validator.fixed_threshold)) for sc in scores
                ],
                "S": [float(x) for x in scores],
            }

            vsum = validator.summary()
            record = {
                "iteration": it + 1,
                "n_test": int(len(test_df)),
                "violation_rate": viol_rate,
                # Both accountings.  The adaptive count answers "how many exceeded
                # the moving bar"; the fixed count answers "how many exceeded the
                # bar calibrated once, offline" -- only the latter can show that
                # the recommendations themselves changed.
                "violations_adaptive": vsum["violations_adaptive"],
                "violations_fixed": vsum["violations_fixed"],
                "violation_rate_fixed": vsum["violation_rate_fixed"],
                "Q_fixed": vsum["Q_fixed"],
                "Q_adaptive": vsum["Q_adaptive"],
                **acc,
                **validm,
                "SNSR": fair_sns.SNSR if fair_sns else None,
                "SNSV": fair_sns.SNSV if fair_sns else None,
                "SNSR_per_attribute": fair_sns.per_attribute if fair_sns else None,
                "SNSR_legacy_proxy": sns.SNSR,
                "Q_last": float(eval_df["Q"].iloc[-1]),
            }
            record["per_item"] = per_item
            if cfr is not None:
                record.update({"CFR": cfr.CFR, "CFR_valid_rate": cfr.valid_rate, "CFR_n_pairs": cfr.n_pairs})

            logger.info(f"Iter {it+1}: {json.dumps(record, indent=2)}")
            history.append(record)

            if it >= 2 and viol_rate < 0.10:
                break

        results[dataset_name] = {
            "config": {
                "seed": Config.RANDOM_SEED,
                "task_mode": Config.TASK_MODE,
                "alpha": Config.ALPHA,
                "lambda": Config.LAMBDA_FAIRNESS,
                "tau_rho": Config.BASE_SIMILARITY,
                "gamma": Config.QUANTILE_DECAY,
                "threshold_update": Config.THRESHOLD_UPDATE,
                "relevance_window": Config.RELEVANCE_WINDOW,
                "n_candidates": Config.RERANK_N_CANDIDATES,
                "cfr_distance": Config.CFR_DISTANCE,
                "eq15_correction": Config.USE_EQ15_CORRECTION,
            },
            "baseline": baseline_block,
            "history": history,
            "Q_alpha_init": float(validator.adaptive_threshold) if validator.adaptive_threshold is not None else None,
        }

    out = os.environ.get("FACTER_OUT", "results_corrected.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"\n=== FINAL RESULTS (written to {out}) ===\n" + json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()

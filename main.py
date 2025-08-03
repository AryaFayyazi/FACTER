"""
main.py: Orchestrates the full FACTER pipeline for ICML paper reproducibility.
"""
from facter.config import Config
from facter.data import DatasetLoader
from facter.models import load_models
from facter.fairness import ConformalFairnessValidator
from facter.prompt_engine import FairPromptEngine
from facter.utils import setup_logging, generate_recommendations, calculate_fairness_metrics, run_baselines

import json
import pandas as pd

def main():
    logger = setup_logging()
    logger.info("Starting FACTER pipeline...")
    embedder, tokenizer, model = load_models()
    results = {}
    for dataset_name in ['amazon', 'ml-1m']:
        logger.info(f"\n=== Running Experiment on {dataset_name.upper()} ===")
        loader = DatasetLoader(dataset_name)
        full_data = loader.prepare_prompts().dropna()
        strat_col = full_data[Config.PROTECTED_ATTRIBUTES].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        valid_strata = strat_col.value_counts()[strat_col.value_counts() >= 2].index
        valid_data = full_data[strat_col.isin(valid_strata)]
        grouped_sample = valid_data.groupby(strat_col, group_keys=False).apply(
            lambda x: x.sample(n=int(Config.SAMPLE_SIZE_PER_DATASET / len(valid_strata)),
                               replace=True),
            include_groups=False
        )
        data = grouped_sample.sample(n=5000, replace=True, random_state=42)
        strat_col = data[Config.PROTECTED_ATTRIBUTES].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        vc = strat_col.value_counts()
        valid_groups = vc[vc >= 2].index
        filtered_data = data[strat_col.isin(valid_groups)].copy()
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            filtered_data,
            test_size=0.3,
            stratify=filtered_data[Config.PROTECTED_ATTRIBUTES].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
        )
        validator = ConformalFairnessValidator(embedder)
        logger.info("Starting calibration...")
        cal_responses = generate_recommendations(train_data['prompt'].tolist(), "", tokenizer, model)
        validator.calibrate(train_data['prompt'].tolist(), cal_responses)
        theory_results = validator.theoretical_analysis()
        logger.info(f"Theoretical Guarantees:\n{json.dumps(theory_results, indent=2)}")
        baseline_metrics = run_baselines(
            test_data.copy(),
            embedder,
            tokenizer,
            model,
            loader.item_db
        )
        prompt_engine = FairPromptEngine(validator)
        violation_rates = []
        fairness_history = []
        for iteration in range(Config.MAX_ITERATIONS):
            prompt_engine.iteration = iteration
            logger.info(f"\n=== Iteration {iteration+1} ===")
            system_msg = prompt_engine.generate_system_prompt()
            responses = generate_recommendations(
                test_data['prompt'].tolist(),
                system_msg,
                tokenizer,
                model
            )
            test_data['response'] = responses
            test_data['is_violation'] = test_data.apply(
                lambda row: validator.validate(row['prompt'], row['response']),
                axis=1
            )
            valid_test_data = test_data[test_data['response'] != ""]
            violation_rate = valid_test_data['is_violation'].mean() if len(valid_test_data) else 0
            violation_rates.append(violation_rate)
            metrics = calculate_fairness_metrics(
                valid_test_data,
                Config.PROTECTED_ATTRIBUTES,
                embedder,
                loader.item_db
            )
            fairness_history.append(metrics)
            logger.info(f"Iteration {iteration+1} Results:")
            logger.info(f"Violation Rate: {violation_rate:.3f}")
            logger.info(f"Fairness Metrics: {json.dumps(metrics, indent=2)}")
            if iteration > 1 and violation_rate < 0.1:
                improvement = (violation_rates[-2] - violation_rates[-1])
                if improvement < 0.005:
                    logger.info("Convergence achieved, early stopping")
                    break
        results[dataset_name] = {
            'violation_rates': violation_rates,
            'fairness_history': fairness_history,
            'baselines': baseline_metrics,
            'theory': validator.theoretical_analysis()
        }
    logger.info("\n=== Final Results Across Datasets ===")
    for dataset, res in results.items():
        logger.info(f"\nDataset: {dataset.upper()}")
        logger.info(f"Final Violation Rate: {res['violation_rates'][-1]:.3f}")
        logger.info("Baseline Comparison:")
        for method, met in res['baselines'].items():
            logger.info(f"{method}: ViolationScore={met['ViolationScore']:.3f}")
        logger.info("Theoretical Analysis:")
        logger.info(json.dumps(res['theory'], indent=2))
    return results

if __name__ == "__main__":
    main()

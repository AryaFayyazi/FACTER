# FACTER: Fairness-Aware Conformal Thresholding and Prompt EngineeRing

> ### See [`REPRODUCIBILITY_NOTE.md`](REPRODUCIBILITY_NOTE.md)
>
> This release accompanies a full re-derivation of the paper's results, together
> with scripts that let you verify each claim directly.
>
> - **The published accuracy numbers reproduce exactly** — all eight cells of
>   Tables 1 and 3, verifiable in five seconds with no GPU via
>   `python scripts/verify_published_tables.py`. The column printed as `NDCG@10`
>   is Precision@10, which is why `NDCG@10 > Recall@10` appears in the tables.
> - **The utility claim reproduces** under the re-ranking formulation the
>   published numbers come from (`FACTER_TASK=rerank`), corroborated independently
>   by the reproduction study's own re-ranking evaluation.
> - **Conformal control reproduces exactly** — the violation rate tracks the
>   target α = 0.2 across iterations.
> - **Prompt repair reduces violations**, reported against both the adaptive and
>   the frozen calibration threshold.

This repository contains the official implementation for the ICML paper (The main branch is the simplified version of the code that is usefull for a demo. For the full implementation of the paper, please check the Fina_Version branch):

**"Fairness-Aware Conformal Thresholding and Prompt EngineeRing (FACTER)"**

## Overview

FACTER is a post-hoc fairness auditing and repair framework for Large Language Model (LLM) recommenders. It leverages conformal prediction and dynamic prompt engineering to ensure group fairness in recommendations, without retraining the underlying model.

## Repository Structure

- `facter/`
  - `config.py` — Hyperparameters and configuration.
  - `data.py` — Dataset loading, preprocessing, and prompt construction.
  - `models.py` — Model and embedder loading utilities.
  - `fairness.py` — Conformal fairness calibration and validation.
  - `prompt_engine.py` — Adversarial prompt engineering logic.
  - `utils.py` — Utility functions (logging, metrics, etc).
  - `metrics_fairness.py` — SNSR / SNSV / CFR.
  - `catalog_map.py` — Maps free-text generations onto the item catalogue.
- `main.py` — End-to-end pipeline: data, model, calibration, fairness, and baselines.
- `experiments/threshold_dynamics.py` — Controlled, model-free check of the online
  threshold-update rules.
- `tests/` — Regression suite covering the metrics, threshold rules and evaluation protocol.
- `requirements.txt` — Python dependencies.
- `SampleOutput.log` — Example output log from a full run.
- `README.md` — This file.

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/AryaFayyazi/FACTER.git
cd FACTER
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU recommended for LLM inference (CPU fallback supported)

## Usage

Run the main pipeline (downloads data automatically):

```bash
python main.py
```

- The script will download MovieLens-1M and Amazon Movies & TV datasets to `./data/`.
- Synthetic demographics are generated for Amazon (see Section 4.2 of the paper).
- Iteration-level fairness metrics and baseline comparisons are printed and logged.

## Sample Output

`SampleOutput.log` is the original run log from the paper's experiments. It is the
artefact used in §2.1 of the reproducibility note to verify the published accuracy
numbers directly — `python scripts/verify_published_tables.py` checks all eight
cells of Tables 1 and 3 against it.

## Troubleshooting

- If you encounter CUDA errors, ensure your system has a compatible GPU and drivers.
- For CPU-only environments, set `device='cpu'` in `facter/config.py`.
- If you run out of memory, reduce `BATCH_SIZE` or use a smaller model.

## Extending FACTER

To add new datasets or models, implement the appropriate loader in `facter/data.py` or update `facter/models.py`.

## File Descriptions

- **`facter/config.py`**: All experiment hyperparameters and dataset URLs.
- **`facter/data.py`**: Data loading, preprocessing, and prompt construction for MovieLens and Amazon.
- **`facter/models.py`**: Loads SentenceTransformer and LLM models.
- **`facter/fairness.py`**: Implements conformal calibration and fairness validation (Sections 3.2–3.3).
- **`facter/prompt_engine.py`**: Dynamic prompt engineering and repair (Section 3.4).
- **`facter/utils.py`**: Logging, metrics, and helper functions.
- **`main.py`**: Orchestrates the full pipeline, including baselines and logging.

## Reproducibility

Start with [`REPRODUCIBILITY_NOTE.md`](REPRODUCIBILITY_NOTE.md), which states the
paper's claims, what reproduces, and how to verify each one.

```bash
python -m pytest tests/ -q                # regression suite
python experiments/threshold_dynamics.py  # the controlled threshold experiment
python main.py                            # full pipeline
```

- All random seeds are fixed.
- `Config.THRESHOLD_UPDATE` selects the online threshold rule: `aci` (default,
  two-sided adaptive conformal), `legacy`, or `paper_eq11`. Violations are
  reported against both the adaptive threshold `Q^(t)` and the frozen calibration
  threshold `Q^(0)`.
- `Config.RELEVANCE_WINDOW` sets the size of the relevance set for Recall/NDCG/HitRate.
- The code runs on CPU or CUDA.

## Citation

If you use this code, please cite the ICML paper, and see §10 of the
reproducibility note for how to cite the reproduction study:

```
@inproceedings{
fayyazi2025facter,
title={{FACTER}: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair {LLM}-Based Recommender Systems},
author={Arya Fayyazi and Mehdi Kamal and Massoud Pedram},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=edN2rEemj6}
}
```

## License

This repository is for academic, non-commercial use only. For other uses, please contact the authors.

---

For theoretical details, see the main paper and Appendix. For questions, please open an issue or contact the authors.

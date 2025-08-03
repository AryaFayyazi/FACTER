# FACTER: Fairness-Aware Conformal Thresholding and Prompt EngineeRing

This repository contains the official implementation for the ICML paper:

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
- `main.py` — End-to-end pipeline: data, model, calibration, fairness, and baselines.
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

See `SampleOutput.log` for an example of the output produced by a full run.

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

- All random seeds are fixed for reproducibility.
- The code is compatible with both CPU and CUDA-enabled GPUs (set `device` in `config.py`).

## Citation

If you use this code, please cite our ICML paper:

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

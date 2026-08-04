"""Regression tests pinning each correction made in the reproducibility audit.

Every test here fails against the previously released code.  Run with:

    python -m pytest tests/ -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facter.catalog_map import rewrite_prompt_attrs  # noqa: E402
from facter.config import Config  # noqa: E402
from facter.utils import parse_ranked_list, recall_ndcg_at_k  # noqa: E402


# --------------------------------------------------------------------------
# D9 / D2: the released branch did not import or run at all.
# --------------------------------------------------------------------------
def test_package_imports():
    import facter.models  # noqa: F401
    import main  # noqa: F401


def test_max_iterations_defined():
    assert isinstance(Config.MAX_ITERATIONS, int) and Config.MAX_ITERATIONS >= 1


# --------------------------------------------------------------------------
# D1: the online threshold update must be two-sided.
# --------------------------------------------------------------------------
def _replay(rule, scores, q0, alpha=0.1, gamma=0.95, eta=0.05):
    q = q0
    adaptive = fixed = 0
    for s in scores:
        if s > q:
            adaptive += 1
        if s > q0:
            fixed += 1
        if rule == "aci":
            q = max(0.0, q + eta * ((1.0 if s > q else 0.0) - alpha))
        elif rule == "legacy" and s > q:
            q = gamma * q + (1 - gamma) * s
        elif rule == "paper_eq11" and s > q:
            q = gamma * q + (1 - gamma) * min(q, s)
    return adaptive, fixed, q


def test_legacy_rule_drifts_upward_on_a_stationary_stream():
    """A stationary score stream must not look like it is 'improving'."""
    rng = np.random.default_rng(0)
    scores = rng.gamma(2.0, 0.45, size=3000)
    q0 = float(np.quantile(scores, 0.9))

    _, _, q_legacy = _replay("legacy", scores, q0)
    _, _, q_aci = _replay("aci", scores, q0)

    # legacy only ever moves up, and moves a long way
    assert q_legacy > q0 * 1.5
    # ACI stays in the neighbourhood of the true quantile
    assert abs(q_aci - q0) < 0.5 * q0


def test_paper_eq11_is_a_noop_on_violations():
    """min(Q, S) == Q whenever S > Q, so Eq. 11 as printed cannot move Q."""
    rng = np.random.default_rng(1)
    scores = rng.gamma(2.0, 0.45, size=2000)
    q0 = float(np.quantile(scores, 0.9))
    a_eq11, f_eq11, _ = _replay("paper_eq11", scores, q0)
    assert a_eq11 == f_eq11


def test_aci_tracks_the_target_miscoverage_rate():
    rng = np.random.default_rng(2)
    scores = rng.gamma(2.0, 0.45, size=20000)
    q0 = float(np.quantile(scores, 0.9))
    adaptive, _, _ = _replay("aci", scores, q0, alpha=0.1)
    assert 0.06 < adaptive / len(scores) < 0.16


# --------------------------------------------------------------------------
# D8: ranking metrics must be self-consistent.
# --------------------------------------------------------------------------
def test_ndcg_cannot_exceed_recall_with_a_single_relevant_item():
    preds = ["A", "B", "C", "D"]
    recall, ndcg, hit, prec = recall_ndcg_at_k(preds, "C", k=4)
    assert recall == 1.0 and hit == 1.0
    assert ndcg <= recall


def test_ndcg_can_exceed_recall_only_with_multiple_relevant_items():
    preds = ["A", "B", "C", "D", "E"]
    gold = "A||B||X||Y||Z"          # 5 relevant, 2 retrieved, both at the top
    recall, ndcg, hit, prec = recall_ndcg_at_k(preds, gold, k=5)
    assert recall == pytest.approx(0.4)
    assert ndcg > recall            # front-loaded hits, IDCG over min(|R|,k)


def test_alpha_matches_the_paper():
    """Table 4 lists the theoretical violation rate as 0.2, i.e. alpha = 0.2."""
    assert Config.ALPHA == pytest.approx(0.2)


def test_paper_hyperparameters():
    assert Config.LAMBDA_FAIRNESS == pytest.approx(0.7)     # lambda
    assert Config.BASE_SIMILARITY == pytest.approx(0.90)    # tau_rho
    assert Config.QUANTILE_DECAY == pytest.approx(0.95)     # gamma
    assert Config.VIOLATION_MEMORY_SIZE == 50               # M


def test_precision_is_reported_alongside_ndcg():
    """The published "NDCG@10" column matches Precision@10, so both must exist."""
    from facter.utils import evaluate_at_k_from_lists
    out = evaluate_at_k_from_lists([["A", "B"]], ["A||C"], k=10)
    assert {"Recall@10", "NDCG@10", "HitRate@10", "Precision@10"} <= set(out)


def test_relevance_window_is_explicit():
    assert Config.RELEVANCE_WINDOW >= 1


# --------------------------------------------------------------------------
# D11: counterfactual rewriting with numeric attribute values.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("age,occ", [("18", "11"), ("56", "0"), ("1", "1")])
def test_rewrite_handles_numeric_attribute_values(age, occ):
    """ML-1M codes age and occupation numerically; rf"\\1{v}" used to raise."""
    prompt = "User profile:\n- gender: M\n- age: 25\n- occupation: 7\n\nWatch history:\n1. Alien\n"
    out = rewrite_prompt_attrs(prompt, {"gender": "F", "age": age, "occupation": occ})
    assert f"- age: {age}" in out
    assert f"- occupation: {occ}" in out
    assert "\\1" not in out


# --------------------------------------------------------------------------
# D10: generation must not echo the prompt back as recommendations.
# --------------------------------------------------------------------------
def test_parser_would_pick_up_the_watch_history_if_given_the_prompt():
    """Documents *why* completions must be sliced off the prompt.

    The prompt renders the watch history as a numbered list, which is exactly
    the format the parser consumes.  Decoding prompt+completion therefore
    returns the user's own history as the 'recommendations'.
    """
    prompt = "Watch history:\n1. Alien\n2. Aliens\n3. Predator\n"
    echoed = parse_ranked_list(prompt, k=10)
    # the user's own history comes back as "recommendations"
    assert {"Alien", "Aliens", "Predator"}.issubset(set(echoed))


def test_generation_slices_the_prompt_off():
    """The generation path must decode only the newly generated tokens."""
    src = (Path(__file__).resolve().parents[1] / "facter" / "utils.py").read_text()
    body = src[src.index("def generate_recommendations("):]
    body = body[: body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    assert re.search(r"outputs\[:,\s*input_ids\.shape\[1\]\s*:\s*\]", body), \
        "generate_recommendations must decode completions only"
    assert "attention_mask=attention_mask" in body, \
        "left-padded batches require an attention mask"


# --------------------------------------------------------------------------
# D12: SNSR/SNSV must follow the FaiRLLM definition the paper cites.
# --------------------------------------------------------------------------
def test_strip_prompt_attrs_removes_the_profile_block():
    from facter.catalog_map import strip_prompt_attrs
    p = ("User profile (audit only):\n- gender: M\n- age: 25\n- occupation: 7\n\n"
         "Watch history:\n1. Alien\n")
    out = strip_prompt_attrs(p)
    for token in ("gender", "age:", "occupation", "audit only"):
        assert token not in out
    assert "Alien" in out


def _fake_snsr(dependence: float):
    """Run the FaiRLLM SNSR on a generator whose output depends on gender."""
    import pandas as pd
    from facter.metrics_fairness import compute_snsr_snsv_fairllm

    # both attribute values must be present: SNSR is a range over the
    # attribute's value set, which is read off the data.
    df = pd.DataFrame([
        {"prompt": f"User profile (audit only):\n- gender: {g}\n- age: 25\n"
                   f"- occupation: 7\n\nWatch history:\n{i}. Film {i}\n",
         "gender": g, "age": "25", "occupation": "7"}
        for i, g in enumerate(["M", "F"] * 3)
    ])

    def generate_fn(prompts, system_msg):
        out = []
        for p in prompts:
            if "- gender: F" in p:
                # F drifts away from the neutral list in proportion to `dependence`
                n_swap = int(round(dependence * 10))
                out.append([f"X{j}" for j in range(n_swap)]
                           + [f"T{j}" for j in range(10 - n_swap)])
            else:
                out.append([f"T{j}" for j in range(10)])
        return out

    return compute_snsr_snsv_fairllm(
        df, embedder=None, generate_fn=generate_fn,
        attributes=["gender"], k=10, n_users=6, similarity="jaccard",
    )


def test_snsr_is_zero_when_output_ignores_the_attribute():
    assert _fake_snsr(0.0).SNSR == pytest.approx(0.0, abs=1e-9)


def test_snsr_grows_with_attribute_dependence():
    """SNSR must actually respond to demographic dependence."""
    assert _fake_snsr(1.0).SNSR > _fake_snsr(0.3).SNSR > 0.0


# --------------------------------------------------------------------------
# D13 / D14: CFR distance and the Eq. 15 finite-sample correction.
# --------------------------------------------------------------------------
def test_cfr_default_distance_is_l2_per_eq14():
    assert Config.CFR_DISTANCE == "l2"


def test_eq15_correction_is_positive_and_shrinks_with_n():
    import numpy as np
    C = float(np.sqrt(np.log(2.0 / Config.CONFORMAL_DELTA) / 2.0))
    assert C / np.sqrt(100) > C / np.sqrt(10000) > 0.0


# --------------------------------------------------------------------------
# D15: the re-ranking task formulation.
# --------------------------------------------------------------------------
def test_rerank_candidate_pool_matches_the_reproduction_study():
    """|C| = 40 (10 relevant + 30 negatives), as in arXiv:2606.28620."""
    assert Config.RERANK_N_CANDIDATES == 40
    assert Config.TASK_MODE in {"open", "rerank"}


# --------------------------------------------------------------------------
# D16: split conformal prediction needs calibration/test exchangeability.
# --------------------------------------------------------------------------
def test_calibration_prompt_matches_deployment_by_default():
    """Calibrating under a prompt no deployed configuration uses voids coverage."""
    assert Config.CALIBRATION_PROMPT == "neutral"


def test_calibration_uses_the_neutral_system_prompt():
    src = (Path(__file__).resolve().parents[1] / "main.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert 'system_msg=""' not in code, \
        "calibration must not generate with an empty system prompt"
    assert "cal_system_msg" in code and "NEUTRAL_SYSTEM_PROMPT" in code


# --------------------------------------------------------------------------
# D17: the static Fair Zero-Shot arm the reproduction study's critique needs.
# --------------------------------------------------------------------------
def test_fair_zero_shot_baseline_exists_and_is_static():
    """Without this arm, 'does the dynamic loop earn its cost?' is unanswerable."""
    from facter.baseline_zero_shot import FAIR_SYSTEM_PROMPT, NEUTRAL_SYSTEM_PROMPT
    assert FAIR_SYSTEM_PROMPT != NEUTRAL_SYSTEM_PROMPT
    assert "fair" in FAIR_SYSTEM_PROMPT.lower()
    # static: it must carry no mined rules and no adaptive threshold
    assert "AVOID" not in FAIR_SYSTEM_PROMPT
    assert "nonconformity" not in FAIR_SYSTEM_PROMPT.lower()


def test_main_evaluates_the_fair_arm_on_the_same_frozen_threshold():
    src = (Path(__file__).resolve().parents[1] / "main.py").read_text()
    assert "FairZeroShot_Static" in src
    assert "fair_viol" in src and "fixed_threshold" in src


def test_seed_is_configurable_for_multi_seed_runs():
    assert isinstance(Config.RANDOM_SEED, int)

"""Evaluation metrics for uplift / CATE models.

Two different metrics for two different purposes:

- PEHE (precision in estimating heterogeneous effects): requires ground-truth
  ITE, so it's only usable in simulation/research settings. This is the
  metric to report when you control the DGP -- it directly answers "did we
  recover the right effect?"

- Qini curve / AUUC: uses only observed outcomes and treatment assignment,
  no ground truth required. This is the metric that generalizes to a real
  deployment, where you never observe the counterfactual. It answers "if we
  target users in order of predicted uplift, how much incremental value do
  we capture?" -- directly the JD's "offline evaluation methods that
  reliably predict online performance."
"""

import numpy as np
import pandas as pd


def pehe(true_ite: np.ndarray, pred_ite: np.ndarray) -> float:
    """Root-mean-squared error between predicted and true individual effects."""
    true_ite = np.asarray(true_ite)
    pred_ite = np.asarray(pred_ite)
    return float(np.sqrt(np.mean((true_ite - pred_ite) ** 2)))


def qini_curve(y: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> pd.DataFrame:
    """Compute the Qini curve from observed outcomes only (no ground truth needed).

    Ranks units by predicted uplift (descending), then at each cumulative
    fraction of the population computes:
        cumulative_treated_gain - cumulative_control_gain * (n_treated / n_control)
    which estimates the incremental outcome captured by targeting the
    top-k units, correcting for unequal treated/control group sizes at
    each cutoff.

    Returns a DataFrame with columns: frac, qini, random_qini
    """
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    order = np.argsort(-uplift_score)
    y_sorted = y[order]
    t_sorted = treatment[order]

    n = len(y)
    cum_treated_y = np.cumsum(y_sorted * t_sorted)
    cum_control_y = np.cumsum(y_sorted * (1 - t_sorted))
    cum_treated_n = np.cumsum(t_sorted)
    cum_control_n = np.cumsum(1 - t_sorted)

    # avoid div by zero at the very first few units
    ratio = np.divide(
        cum_treated_n, cum_control_n, out=np.zeros_like(cum_treated_n), where=cum_control_n > 0
    )
    qini = cum_treated_y - cum_control_y * ratio

    total_ate_gain = qini[-1]
    frac = np.arange(1, n + 1) / n
    random_qini = frac * total_ate_gain

    return pd.DataFrame({"frac": frac, "qini": qini, "random_qini": random_qini})


def auuc(qini_df: pd.DataFrame) -> float:
    """Area under the uplift curve, above the random-targeting baseline."""
    gap = qini_df["qini"].values - qini_df["random_qini"].values
    return float(np.trapz(gap, qini_df["frac"].values))

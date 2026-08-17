"""
Synthetic promotional-booking data generator.

Mimics the causal structure of an accommodation-marketplace promotion
problem: a user searches, may or may not be shown a promo/discount, and
either books or doesn't. The point of using synthetic data (rather than a
public dataset) is that we get to KEEP the ground-truth individual treatment
effect (ITE) for every unit, which lets us validate the causal models
against the truth later -- something you can never do with real-world data,
where only the factual outcome is ever observed.

Design goals baked into the DGP:
  1. Confounded treatment assignment -- propensity to receive the promo
     depends on covariates (price sensitivity, loyalty tier), so a naive
     treated-vs-control comparison is biased. This is what motivates IPW /
     doubly-robust correction.
  2. Heterogeneous treatment effects -- the ITE varies with covariates and
     is deliberately non-monotonic, covering all four uplift quadrants:
       - Persuadables : promo meaningfully drives booking      (ITE >> 0)
       - Sure things   : books regardless of promo              (ITE ~ 0)
       - Lost causes   : won't book regardless of promo         (ITE ~ 0)
       - Sleeping dogs : promo actively hurts (margin cannibal.) (ITE < 0)
     This is the actual business framing for incremental-ROI targeting:
     spend should go to persuadables only, not "sure things" (wasted
     margin) or "sleeping dogs" (actively harmful).
"""

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_promo_data(n: int = 20_000, seed: int = 98) -> pd.DataFrame:
    """Simulate a promotional-booking dataset with known ground truth.

    Returns a DataFrame with:
      - covariates: loyalty_tier, price_sensitivity, lead_time_days,
        past_bookings, device_mobile, length_of_stay, base_price
      - propensity: true P(treatment=1 | X)   [kept for diagnostics only]
      - treatment:  observed binary treatment (promo shown)
      - y0, y1:     potential outcomes (booking value) under control/treated
      - ite:        true individual treatment effect = y1 - y0
      - y:          observed (factual) outcome = y1 if treated else y0
      - quadrant:   human-readable uplift quadrant label, for EDA only
    """
    rng = np.random.default_rng(seed)

    # ---- Covariates -----------------------------------------------------
    loyalty_tier = rng.integers(0, 4, size=n)  # 0=none .. 3=top tier
    price_sensitivity = rng.normal(0, 1, size=n)  # standardized, higher = more sensitive
    lead_time_days = rng.gamma(shape=2.0, scale=8.0, size=n)
    past_bookings = rng.poisson(lam=1.5 + 0.8 * loyalty_tier, size=n)
    device_mobile = rng.binomial(1, 0.55, size=n)
    length_of_stay = rng.poisson(lam=2.5, size=n) + 1
    base_price = rng.lognormal(mean=4.6, sigma=0.4, size=n)  # ~ EUR 60-200 typical

    X = pd.DataFrame(
        {
            "loyalty_tier": loyalty_tier,
            "price_sensitivity": price_sensitivity,
            "lead_time_days": lead_time_days,
            "past_bookings": past_bookings,
            "device_mobile": device_mobile,
            "length_of_stay": length_of_stay,
            "base_price": base_price,
        }
    )

    # ---- Confounded propensity (marketing targets price-sensitive, ----
    # ---- mid-tier users more aggressively with promos) -----------------
    z = (
        -0.3
        + 0.55 * price_sensitivity
        + 0.35 * (loyalty_tier == 1).astype(float)
        + 0.35 * (loyalty_tier == 2).astype(float)
        - 0.25 * (loyalty_tier == 3).astype(float)  # top tier targeted less
        - 0.02 * (lead_time_days - 16)
        + 0.15 * device_mobile
        + rng.normal(0, 0.5, size=n)
    )
    propensity = _sigmoid(z).clip(0.05, 0.95)  # keep overlap for IPW to be valid
    treatment = rng.binomial(1, propensity)

    # ---- Baseline (no-promo) booking probability ------------------------
    base_logit = (
        -1.4
        + 0.55 * loyalty_tier
        - 0.35 * price_sensitivity
        - 0.015 * lead_time_days
        + 0.10 * past_bookings
        - 0.10 * device_mobile
    )

    # ---- Heterogeneous treatment effect on the logit scale --------------
    # Persuadables: mid loyalty (1-2) + high price sensitivity -> big lift
    persuadable_score = (
        (loyalty_tier == 1).astype(float) + (loyalty_tier == 2).astype(float)
    ) * np.clip(price_sensitivity, 0, None)
    # Sleeping dogs: top loyalty tier + low price sensitivity -> promo
    # cannibalizes a booking that would have happened at full price/margin
    sleeping_dog_score = (loyalty_tier == 3).astype(float) * np.clip(
        -price_sensitivity, 0, None
    )

    ite_logit = 0.9 * persuadable_score - 0.6 * sleeping_dog_score
    # Lost causes (tier 0, low sensitivity) and sure things (tier 3, high
    # sensitivity) fall out naturally: persuadable/sleeping_dog scores are
    # both ~0 for them, so ite_logit ~ 0.

    y1_logit = base_logit + ite_logit
    y0_prob = _sigmoid(base_logit)
    y1_prob = _sigmoid(y1_logit)

    # Potential outcomes are booking VALUE = P(book) * price, with a bit of
    # multiplicative noise so it's a continuous outcome rather than binary.
    noise0 = rng.lognormal(0, 0.15, size=n)
    noise1 = rng.lognormal(0, 0.15, size=n)
    y0 = y0_prob * base_price * noise0
    y1 = y1_prob * base_price * noise1

    ite = y1 - y0
    y = np.where(treatment == 1, y1, y0)

    # ---- Quadrant labels (EDA / storytelling only, not used by models) --
    quadrant = np.select(
        [
            (persuadable_score > 0.3),
            (sleeping_dog_score > 0.3),
            (loyalty_tier == 3),
        ],
        ["persuadable", "sleeping_dog", "sure_thing"],
        default="lost_cause",
    )

    df = X.copy()
    df["propensity_true"] = propensity
    df["treatment"] = treatment
    df["y0_true"] = y0
    df["y1_true"] = y1
    df["ite_true"] = ite
    df["y"] = y
    df["quadrant"] = quadrant
    return df


if __name__ == "__main__":
    df = simulate_promo_data()
    print(df.shape)
    print(df["quadrant"].value_counts(normalize=True).round(3))
    print("True ATE:", df["ite_true"].mean().round(3))
    naive = df.loc[df.treatment == 1, "y"].mean() - df.loc[df.treatment == 0, "y"].mean()
    print("Naive (biased) treated-vs-control diff:", round(naive, 3))

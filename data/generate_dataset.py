"""
Synthetic dataset generator for the Express Checkout A/B experiment.

Scenario: A Shopee-style marketplace ran a 3-week experiment testing whether a
2-step "Express Checkout" (pre-filled address + saved payment) increased purchase
conversion among repeat buyers versus the standard 4-step checkout flow.

Two data quality issues are embedded intentionally:
1. Novelty effect: treatment conversion rate is inflated in Week 1, then stabilises.
   An analyst who aggregates without checking temporal trends will overstate lift by ~50%.
2. Whale buyers: ~2.5% of converters receive an 8-15x order value multiplier, creating
   a heavy right tail. A 5-sigma outlier threshold in the analysis captures only the most
   extreme of these (~1% of converters); 3-sigma recovers more. Either way, Mann-Whitney U
   is the appropriate primary test for AOV given the distributional shape.

Usage:
    python generate_dataset.py
    Writes: checkout_experiment.csv  (same directory as this script)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_USERS = 12_000
OUTPUT_PATH = Path(__file__).parent / "checkout_experiment.csv"


def _assign_experiment_groups(rng, n_users):
    """50/50 split, user-level assignment."""
    group = rng.choice(["control", "treatment"], size=n_users, p=[0.50, 0.50])
    return group


def _assign_weeks(rng, n_users):
    """Distribute users roughly evenly across 3 experiment weeks."""
    return rng.choice([1, 2, 3], size=n_users, p=[0.34, 0.33, 0.33])


def _assign_segments(rng, n_users):
    """
    user_type: 60% returning, 40% new (new users have lower baseline conversion).
    device: 65% mobile, 35% desktop.
    """
    user_type = rng.choice(["returning", "new"], size=n_users, p=[0.60, 0.40])
    device = rng.choice(["mobile", "desktop"], size=n_users, p=[0.65, 0.35])
    return user_type, device


def _compute_conversion_probability(group, week, user_type, device):
    """
    Baseline conversion rates and treatment effects, incorporating:
    - Novelty bump in Week 1 for treatment only
    - Segment-level heterogeneity (returning > new, desktop > mobile at baseline)
    """
    # Baseline by segment
    base = np.where(
        (user_type == "returning") & (device == "desktop"), 0.195,
        np.where(
            (user_type == "returning") & (device == "mobile"), 0.175,
            np.where(
                (user_type == "new") & (device == "desktop"), 0.110,
                0.095,  # new + mobile: lowest baseline
            ),
        ),
    )

    # True steady-state treatment lift: +1.9 percentage points
    treatment_lift_steady = 0.019

    # Novelty bump in Week 1: additional +2.2pp on top of steady-state lift
    # This inflates Week 1 treatment conversion to ~4.1pp above control,
    # versus the real ~1.9pp in Weeks 2-3.
    novelty_bump = np.where(
        (group == "treatment") & (week == 1), 0.022, 0.0
    )

    treatment_effect = np.where(
        group == "treatment", treatment_lift_steady, 0.0
    )

    return np.clip(base + treatment_effect + novelty_bump, 0.0, 1.0)


def _generate_aov(rng, n_users, converted_mask):
    """
    AOV for converted sessions only.
    Base distribution: log-normal (realistic for e-commerce, right-skewed).
    Whale buyers: ~2.5% of converters get an additional multiplier of 8x-15x,
    creating a heavy right tail that causes Levene's test to fail.
    """
    n_converted = converted_mask.sum()
    aov = np.full(n_users, np.nan)

    # Log-normal base: median ~IDR 185k, mean ~IDR 220k (realistic for Shopee)
    base_aov = rng.lognormal(mean=12.1, sigma=0.55, size=n_converted)

    # Whale buyers: 2.5% of converters
    whale_mask = rng.random(size=n_converted) < 0.025
    whale_multiplier = rng.uniform(8, 15, size=n_converted)
    base_aov = np.where(whale_mask, base_aov * whale_multiplier, base_aov)

    aov[converted_mask] = np.round(base_aov, 2)
    return aov


def _generate_payment_failure(rng, n_users, group):
    """
    Payment failure rate: control 4.8%, treatment 5.1%.
    The slight increase in treatment is plausible because Express Checkout
    surfaces stored payment methods that may be expired. Effect is small
    enough to be a guardrail check (not a clear blocker).
    """
    failure_prob = np.where(group == "treatment", 0.051, 0.048)
    return rng.binomial(1, failure_prob, size=n_users).astype(bool)


def _generate_session_duration(rng, n_users, group, converted):
    """
    Session duration in seconds. Treatment sessions are shorter on average
    (fewer steps), with higher variance for non-converters (browsed around
    but did not complete). Not a primary metric, included for richness.
    """
    base_duration = np.where(group == "control", 185, 140)
    noise = rng.normal(0, 45, size=n_users)
    duration = np.clip(base_duration + noise, 20, None)
    return np.round(duration, 1)


def generate(seed=SEED):
    rng = np.random.default_rng(seed)

    user_ids = np.arange(1, N_USERS + 1)
    group = _assign_experiment_groups(rng, N_USERS)
    week = _assign_weeks(rng, N_USERS)
    user_type, device = _assign_segments(rng, N_USERS)

    conversion_prob = _compute_conversion_probability(group, week, user_type, device)
    checkout_conversion = rng.binomial(1, conversion_prob).astype(bool)

    aov_idr = _generate_aov(rng, N_USERS, checkout_conversion)
    payment_failure = _generate_payment_failure(rng, N_USERS, group)
    session_duration_sec = _generate_session_duration(rng, N_USERS, group, checkout_conversion)

    # Experiment day: spread users across 21 days (3 weeks), earlier days = earlier weeks
    day_of_week_map = {1: (1, 7), 2: (8, 14), 3: (15, 21)}
    experiment_day = np.array([
        rng.integers(day_of_week_map[w][0], day_of_week_map[w][1] + 1)
        for w in week
    ])

    df = pd.DataFrame({
        "user_id": user_ids,
        "experiment_group": group,
        "experiment_week": week,
        "experiment_day": experiment_day,
        "user_type": user_type,
        "device": device,
        "checkout_conversion": checkout_conversion.astype(int),
        "aov_idr": aov_idr,
        "payment_failure": payment_failure.astype(int),
        "session_duration_sec": session_duration_sec,
    })

    # Shuffle rows so the ordering does not reveal the generation structure
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def main():
    print(f"Generating dataset with seed={SEED}, n_users={N_USERS}...")
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"\nShape: {df.shape}")
    print(f"\nGroup counts:\n{df['experiment_group'].value_counts()}")
    print(f"\nOverall conversion rate: {df['checkout_conversion'].mean():.4f}")
    print(f"\nConversion by group:")
    print(df.groupby("experiment_group")["checkout_conversion"].mean().round(4))
    print(f"\nConversion by group and week (novelty effect visible here):")
    print(
        df.groupby(["experiment_group", "experiment_week"])["checkout_conversion"]
        .mean()
        .round(4)
        .unstack("experiment_week")
    )
    print(f"\nAOV summary (converted only):")
    print(df.loc[df["checkout_conversion"] == 1, "aov_idr"].describe().round(0))
    print(f"\nMissing values:\n{df.isnull().sum()}")


if __name__ == "__main__":
    main()

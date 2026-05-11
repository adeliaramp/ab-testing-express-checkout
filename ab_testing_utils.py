"""
ab_testing_utils.py

Reusable A/B testing functions extracted from the Express Checkout experiment notebook.
Each function is self-contained and documented with its assumptions and return contract.

Designed for analysts who want to drop these into a new experiment with minimal
adaptation. Functions follow a consistent pattern: return a dict (for programmatic use)
and optionally print a formatted summary.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize


# IBM color-blind-safe palette used throughout
IBM_BLUE = "#648FFF"
IBM_ORANGE = "#FE6100"
IBM_PURPLE = "#785EF0"
IBM_RED = "#DC267F"
IBM_GOLD = "#FFB000"


def check_srm(
    control_n: int,
    treatment_n: int,
    expected_split: float = 0.5,
    alpha: float = 0.05,
) -> dict:
    """
    Sample Ratio Mismatch (SRM) test using a chi-squared goodness-of-fit test.

    An SRM indicates that the randomisation mechanism may have a bug: users
    are not landing in treatment/control at the intended ratio. Even a small
    SRM can invalidate the experiment because the imbalance is often correlated
    with user characteristics.

    Parameters
    ----------
    control_n : int
        Number of users assigned to the control group.
    treatment_n : int
        Number of users assigned to the treatment group.
    expected_split : float
        Expected fraction of users in the treatment group (default 0.5 for 50/50).
    alpha : float
        Significance level for the SRM test (default 0.05).

    Returns
    -------
    dict with keys:
        control_n, treatment_n, total_n, observed_treatment_fraction,
        expected_treatment_fraction, chi2_stat, p_value, srm_detected, verdict
    """
    total_n = control_n + treatment_n
    expected_treatment_n = total_n * expected_split
    expected_control_n = total_n * (1 - expected_split)

    observed = np.array([control_n, treatment_n])
    expected = np.array([expected_control_n, expected_treatment_n])

    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    srm_detected = p_value < alpha

    verdict = (
        "SRM DETECTED: randomisation imbalance is statistically significant. "
        "Investigate the assignment pipeline before trusting any metric results."
        if srm_detected
        else "No SRM detected. Group sizes are consistent with the intended split."
    )

    result = {
        "control_n": control_n,
        "treatment_n": treatment_n,
        "total_n": total_n,
        "observed_treatment_fraction": treatment_n / total_n,
        "expected_treatment_fraction": expected_split,
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_value, 4),
        "srm_detected": srm_detected,
        "verdict": verdict,
    }

    print(
        f"SRM Check\n"
        f"  Control n={control_n:,}  |  Treatment n={treatment_n:,}  |  Total={total_n:,}\n"
        f"  Observed split: {treatment_n / total_n:.4f} treatment "
        f"(expected {expected_split:.4f})\n"
        f"  Chi2={chi2_stat:.4f}, p={p_value:.4f}\n"
        f"  {verdict}\n"
    )

    return result


def run_frequentist_test(
    control: pd.Series,
    treatment: pd.Series,
    metric_type: Literal["proportion", "continuous"],
    alpha: float = 0.05,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
    run_nonparametric: bool = True,
) -> dict:
    """
    Run a frequentist hypothesis test with full assumption checks.

    For 'proportion' metrics: two-proportion z-test.
    For 'continuous' metrics: Welch t-test (unequal variance) after checking
    normality (Shapiro-Wilk on a subsample) and variance homogeneity (Levene).
    If run_nonparametric=True, also runs Mann-Whitney U as a robustness check.

    Normality check uses Shapiro-Wilk on up to 5,000 observations per group
    (Shapiro-Wilk is unreliable above ~5k due to extreme sensitivity to minor
    deviations in large samples).

    Parameters
    ----------
    control : pd.Series
        Metric values for the control group. For 'proportion' metrics, values
        must be 0 or 1.
    treatment : pd.Series
        Metric values for the treatment group. Same type requirement as control.
    metric_type : str
        'proportion' or 'continuous'.
    alpha : float
        Significance threshold (default 0.05).
    alternative : str
        Direction of the test. Default 'two-sided'.
    run_nonparametric : bool
        If True and metric_type='continuous', also runs Mann-Whitney U.

    Returns
    -------
    dict with keys: metric_type, primary_test, p_value, statistic,
        significant, assumption_checks, nonparametric (if applicable)
    """
    control = control.dropna()
    treatment = treatment.dropna()
    result: dict = {"metric_type": metric_type, "alpha": alpha}

    if metric_type == "proportion":
        _validate_binary(control, "control")
        _validate_binary(treatment, "treatment")

        count = np.array([treatment.sum(), control.sum()])
        nobs = np.array([len(treatment), len(control)])
        stat, p_value = proportions_ztest(count, nobs, alternative=alternative)

        control_rate = control.mean()
        treatment_rate = treatment.mean()
        lift_absolute = treatment_rate - control_rate
        lift_relative = lift_absolute / control_rate if control_rate > 0 else np.nan

        ci_low, ci_high = _proportion_diff_ci(control, treatment, alpha)

        result.update({
            "primary_test": "two-proportion z-test",
            "control_rate": round(control_rate, 6),
            "treatment_rate": round(treatment_rate, 6),
            "lift_absolute": round(lift_absolute, 6),
            "lift_relative_pct": round(lift_relative * 100, 3),
            "statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "ci_95": (round(ci_low, 6), round(ci_high, 6)),
            "significant": p_value < alpha,
            "assumption_checks": {
                "independence": "Assumed: user-level assignment, one row per user.",
                "large_sample": (
                    f"Control: np={control.sum():.0f}, n(1-p)={_count_failures(control):.0f}. "
                    f"Treatment: np={treatment.sum():.0f}, n(1-p)={_count_failures(treatment):.0f}. "
                    "Both > 10, CLT applies."
                ),
            },
        })

        _print_test_summary(result)

    elif metric_type == "continuous":
        normality = _check_normality(control, treatment)
        levene_stat, levene_p = stats.levene(control, treatment)
        equal_var = levene_p >= alpha

        stat, p_value = stats.ttest_ind(control, treatment, equal_var=equal_var, alternative=alternative)

        ci_low, ci_high = _mean_diff_ci(control, treatment, alpha)
        control_mean = control.mean()
        treatment_mean = treatment.mean()
        lift_relative = (treatment_mean - control_mean) / control_mean if control_mean != 0 else np.nan

        result.update({
            "primary_test": f"{'Welch' if not equal_var else 'Student'} t-test (equal_var={equal_var})",
            "control_mean": round(control_mean, 2),
            "treatment_mean": round(treatment_mean, 2),
            "lift_absolute": round(treatment_mean - control_mean, 2),
            "lift_relative_pct": round(lift_relative * 100, 3),
            "statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "ci_95": (round(ci_low, 2), round(ci_high, 2)),
            "significant": p_value < alpha,
            "assumption_checks": {
                "normality": normality,
                "variance_homogeneity": {
                    "levene_stat": round(levene_stat, 4),
                    "levene_p": round(levene_p, 4),
                    "equal_var_assumed": equal_var,
                    "note": (
                        "Levene p < 0.05: variances differ, using Welch correction."
                        if not equal_var
                        else "Levene p >= 0.05: equal variance assumption holds."
                    ),
                },
            },
        })

        if run_nonparametric:
            mw_stat, mw_p = stats.mannwhitneyu(
                treatment, control, alternative=alternative
            )
            result["nonparametric"] = {
                "test": "Mann-Whitney U",
                "statistic": round(mw_stat, 4),
                "p_value": round(mw_p, 6),
                "significant": mw_p < alpha,
                "note": (
                    "Non-parametric alternative; does not assume normality or equal variance. "
                    "Tests whether treatment values tend to be higher than control."
                ),
            }

        _print_test_summary(result)

    else:
        raise ValueError(f"metric_type must be 'proportion' or 'continuous', got '{metric_type}'")

    return result


def compute_effect_size(
    control: pd.Series,
    treatment: pd.Series,
    metric_type: Literal["proportion", "continuous"] = "continuous",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """
    Compute standardised effect size and a bootstrap confidence interval.

    For proportions: Cohen's h (arcsine transformation, scale-independent).
    For continuous: Cohen's d using pooled standard deviation.

    The bootstrap CI uses the percentile method with n_bootstrap resamples.
    Resampling is done in memory-efficient chunks of 1,000 iterations.
    Note: BCa would be more accurate for small n and asymmetric distributions;
    at n > 500 per group the difference is negligible for typical product metrics.

    Interpretation benchmarks (Cohen 1988):
        Small: |d| or |h| >= 0.2
        Medium: >= 0.5
        Large: >= 0.8

    Note: Cohen's 1988 benchmarks were developed for psychology experiments.
    In e-commerce product experiments, effect sizes of |h| = 0.02 to 0.10
    are typical for shipped UI changes (Kohavi et al., 2020).

    Parameters
    ----------
    control, treatment : pd.Series
        Metric values per group.
    metric_type : str
        'proportion' or 'continuous'.
    n_bootstrap : int
        Number of bootstrap resamples (default 10,000).
    alpha : float
        Used to compute the (1 - alpha) bootstrap CI (default 0.05 gives 95% CI).

    Returns
    -------
    dict with keys: effect_size_type, value, magnitude, ci_lower, ci_upper, n_bootstrap
    """
    control = control.dropna()
    treatment = treatment.dropna()

    ctrl_arr = control.to_numpy()
    trt_arr = treatment.to_numpy()

    ci_lower, ci_upper = _bootstrap_effect_size_ci(
        ctrl_arr, trt_arr, metric_type, n_bootstrap, alpha
    )

    if metric_type == "proportion":
        p_control = control.mean()
        p_treatment = treatment.mean()
        h = proportion_effectsize(p_treatment, p_control)
        magnitude = _cohen_magnitude(abs(h))
        result = {
            "effect_size_type": "Cohen's h (arcsine transform)",
            "p_control": round(p_control, 6),
            "p_treatment": round(p_treatment, 6),
            "value": round(h, 4),
            "magnitude": magnitude,
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "n_bootstrap": n_bootstrap,
        }

    else:
        n_ctrl = len(control)
        n_trt = len(treatment)
        pooled_std = np.sqrt(
            ((n_ctrl - 1) * control.std(ddof=1) ** 2 + (n_trt - 1) * treatment.std(ddof=1) ** 2)
            / (n_ctrl + n_trt - 2)
        )
        d = (treatment.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0.0
        magnitude = _cohen_magnitude(abs(d))
        result = {
            "effect_size_type": "Cohen's d (pooled SD)",
            "control_mean": round(control.mean(), 4),
            "treatment_mean": round(treatment.mean(), 4),
            "pooled_std": round(pooled_std, 4),
            "value": round(d, 4),
            "magnitude": magnitude,
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "n_bootstrap": n_bootstrap,
        }

    print(
        f"Effect Size ({result['effect_size_type']}): "
        f"{result['value']:+.4f}  [{magnitude}]  "
        f"95% Bootstrap CI: [{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]  "
        f"(n_bootstrap={n_bootstrap})\n"
    )
    return result


def segment_analysis(
    df: pd.DataFrame,
    segment_col: str,
    metric_col: str,
    group_col: str = "experiment_group",
    metric_type: Literal["proportion", "continuous"] = "proportion",
    alpha: float = 0.05,
    apply_correction: bool = True,
) -> pd.DataFrame:
    """
    Run the frequentist test independently within each segment level.

    Returns a tidy DataFrame where each row is one segment level, with
    control/treatment summary stats, test result, and effect size. Useful
    for detecting heterogeneous treatment effects (HTE) across subgroups.

    When apply_correction=True (default), Holm step-down correction is applied
    across all p-values within this call. Holm is preferred over Bonferroni
    because it is uniformly more powerful while still controlling the
    family-wise error rate. The 'significant' column reflects the corrected
    decision. Raw p-values are preserved in 'p_value' for transparency;
    corrected p-values appear in 'p_value_holm'.

    Parameters
    ----------
    df : pd.DataFrame
        Full experiment dataset.
    segment_col : str
        Column name of the segment variable (e.g. 'user_type', 'device').
    metric_col : str
        Column name of the metric to test.
    group_col : str
        Column name of the experiment group indicator.
    metric_type : str
        'proportion' or 'continuous'.
    alpha : float
        Significance threshold.
    apply_correction : bool
        If True, apply Holm step-down correction to p-values (default True).

    Returns
    -------
    pd.DataFrame with one row per segment level. Columns include p_value (raw),
    p_value_holm (corrected), and significant (based on corrected p-value).
    """
    records = []

    for segment_value in sorted(df[segment_col].unique()):
        segment_df = df[df[segment_col] == segment_value]
        control_vals = segment_df.loc[segment_df[group_col] == "control", metric_col]
        treatment_vals = segment_df.loc[segment_df[group_col] == "treatment", metric_col]

        control_vals = control_vals.dropna()
        treatment_vals = treatment_vals.dropna()

        test_result = run_frequentist_test(
            control_vals, treatment_vals, metric_type=metric_type, alpha=alpha
        )
        effect_result = compute_effect_size(control_vals, treatment_vals, metric_type=metric_type)

        if metric_type == "proportion":
            record = {
                "segment": segment_value,
                "control_n": len(control_vals),
                "treatment_n": len(treatment_vals),
                "control_rate": test_result["control_rate"],
                "treatment_rate": test_result["treatment_rate"],
                "lift_absolute": test_result["lift_absolute"],
                "lift_relative_pct": test_result["lift_relative_pct"],
                "p_value": test_result["p_value"],
                "significant": test_result["significant"],
                "effect_size": effect_result["value"],
                "effect_magnitude": effect_result["magnitude"],
            }
        else:
            record = {
                "segment": segment_value,
                "control_n": len(control_vals),
                "treatment_n": len(treatment_vals),
                "control_mean": test_result["control_mean"],
                "treatment_mean": test_result["treatment_mean"],
                "lift_absolute": test_result["lift_absolute"],
                "lift_relative_pct": test_result["lift_relative_pct"],
                "p_value": test_result["p_value"],
                "significant": test_result["significant"],
                "effect_size": effect_result["value"],
                "effect_magnitude": effect_result["magnitude"],
            }

        records.append(record)

    if apply_correction and len(records) > 1:
        raw_pvals = [r["p_value"] for r in records]
        reject, p_corrected, _, _ = multipletests(raw_pvals, alpha=alpha, method="holm")
        for i, record in enumerate(records):
            record["p_value_holm"] = round(float(p_corrected[i]), 6)
            record["significant"] = bool(reject[i])
    else:
        for record in records:
            record["p_value_holm"] = record["p_value"]

    return pd.DataFrame(records)


def plot_results(results_df: pd.DataFrame, metric_label: str = "Metric") -> plt.Figure:
    """
    Visualise segment-level test results as a forest plot.

    Intended for analyses with 5 or more segments. For 2 to 4 segments,
    a grouped bar chart (see notebook Section 5) is more readable.

    Each row shows one segment with its point estimate (relative lift),
    significance indicator, and a reference line at zero. Designed for
    the segmentation analysis output from segment_analysis().

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from segment_analysis(). Must contain columns:
        'segment', 'lift_relative_pct', 'p_value', 'significant'.
    metric_label : str
        Human-readable label for the metric being plotted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = results_df.copy().reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(9, max(3, n * 1.4)))

    p_col = "p_value_holm" if "p_value_holm" in df.columns else "p_value"

    for i, row in df.iterrows():
        color = IBM_BLUE if row["significant"] else IBM_ORANGE
        marker = "D" if row["significant"] else "o"
        ax.scatter(
            row["lift_relative_pct"], i,
            color=color, marker=marker, s=90, zorder=3
        )
        sig_label = "*" if row["significant"] else ""
        ax.text(
            row["lift_relative_pct"],
            i + 0.25,
            f"{row['lift_relative_pct']:+.1f}%{sig_label} (p={row[p_col]:.3f})",
            ha="center", va="bottom", fontsize=9, color=color
        )

    ax.axvline(0, color="grey", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["segment"], fontsize=10)
    ax.set_xlabel(f"Relative Lift in {metric_label} (%)", fontsize=11)
    ax.set_title(
        f"Treatment vs. Control: {metric_label} by Segment\n"
        "(filled diamond = significant after Holm correction; circle = not significant)",
        fontsize=12, pad=12
    )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    legend_elements = [
        plt.scatter([], [], color=IBM_BLUE, marker="D", s=80, label="Significant (Holm-corrected)"),
        plt.scatter([], [], color=IBM_ORANGE, marker="o", s=80, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.8)

    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_segment_bar_chart(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    segment_col: str,
    metric_col: str,
    metric_label: str,
    ax: plt.Axes,
    group_col: str = "experiment_group",
    alpha: float = 0.05,
) -> None:
    """
    Grouped bar chart with (1 - alpha) CI error bars for a segment_analysis output.

    Intended for analyses with 2 to 4 segments. For 5 or more segments,
    use plot_results (forest plot) instead.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from segment_analysis(). Must contain columns:
        'segment', 'lift_relative_pct', 'significant'.
    raw_df : pd.DataFrame
        Full experiment dataset used to compute per-segment CI error bars.
    segment_col : str
        Column name of the segment variable in raw_df.
    metric_col : str
        Column name of the binary (0/1) metric in raw_df.
    metric_label : str
        Human-readable label for the y-axis.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    group_col : str
        Column name of the experiment group indicator (default 'experiment_group').
    alpha : float
        Significance level; error bars show the (1 - alpha) CI (default 0.05 gives 95% CI).
    """
    segments = results_df["segment"].tolist()
    x = np.arange(len(segments))
    width = 0.35
    z = stats.norm.ppf(1 - alpha / 2)

    ctrl_rates, trt_rates, ctrl_err, trt_err = [], [], [], []
    for seg in segments:
        seg_df = raw_df[raw_df[segment_col] == seg]
        ctrl = seg_df[seg_df[group_col] == "control"][metric_col].dropna()
        trt = seg_df[seg_df[group_col] == "treatment"][metric_col].dropna()
        p_c, p_t = ctrl.mean(), trt.mean()
        ctrl_rates.append(p_c * 100)
        trt_rates.append(p_t * 100)
        ctrl_err.append(z * np.sqrt(p_c * (1 - p_c) / len(ctrl)) * 100)
        trt_err.append(z * np.sqrt(p_t * (1 - p_t) / len(trt)) * 100)

    ax.bar(x - width / 2, ctrl_rates, width, label="Control", color=IBM_BLUE, alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, trt_rates, width, label="Treatment", color=IBM_ORANGE, alpha=0.85, edgecolor="white")
    ax.errorbar(x - width / 2, ctrl_rates, yerr=ctrl_err, fmt="none", color="#333333", capsize=5, linewidth=1.2)
    ax.errorbar(x + width / 2, trt_rates, yerr=trt_err, fmt="none", color="#333333", capsize=5, linewidth=1.2)

    for i in range(len(segments)):
        row = results_df.iloc[i]
        sig_label = "* (Holm)" if row["significant"] else "ns"
        color = IBM_BLUE if row["significant"] else "grey"
        top = max(trt_rates[i] + trt_err[i], ctrl_rates[i] + ctrl_err[i])
        ax.text(
            i, top + 0.4,
            f"{row['lift_relative_pct']:+.1f}% ({sig_label})",
            ha="center", va="bottom", fontsize=10, color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in segments], fontsize=11)
    ax.set_ylabel(f"{metric_label} (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.legend(loc="upper right", fontsize=9)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_binary(series: pd.Series, name: str) -> None:
    unique_vals = set(series.dropna().unique())
    if not unique_vals.issubset({0, 1, True, False}):
        raise ValueError(
            f"'{name}' contains non-binary values {unique_vals}. "
            "Proportion metrics must be 0/1."
        )


def _count_failures(s: pd.Series) -> float:
    return (1 - s).sum()


def _check_normality(control: pd.Series, treatment: pd.Series) -> dict:
    """Shapiro-Wilk on up to 5,000 samples per group."""
    MAX_SW = 5_000
    ctrl_sample = control.sample(min(len(control), MAX_SW), random_state=42)
    trt_sample = treatment.sample(min(len(treatment), MAX_SW), random_state=42)

    sw_ctrl_stat, sw_ctrl_p = stats.shapiro(ctrl_sample)
    sw_trt_stat, sw_trt_p = stats.shapiro(trt_sample)

    return {
        "test": "Shapiro-Wilk (up to 5,000 obs per group)",
        "control": {"stat": round(sw_ctrl_stat, 4), "p": round(sw_ctrl_p, 4)},
        "treatment": {"stat": round(sw_trt_stat, 4), "p": round(sw_trt_p, 4)},
        "note": (
            "With n > 1,000, Shapiro-Wilk is extremely sensitive to minor "
            "deviations. Use Q-Q plot and CLT argument alongside this p-value."
        ),
    }


def _proportion_diff_ci(control: pd.Series, treatment: pd.Series, alpha: float) -> tuple:
    """95% CI for the difference in proportions using normal approximation."""
    p1 = treatment.mean()
    p2 = control.mean()
    n1 = len(treatment)
    n2 = len(control)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return diff - z * se, diff + z * se


def _mean_diff_ci(control: pd.Series, treatment: pd.Series, alpha: float) -> tuple:
    """Welch t-test based CI for the difference in means."""
    result = stats.ttest_ind(treatment, control, equal_var=False)
    diff = treatment.mean() - control.mean()
    se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
    df_welch = result.df
    t_crit = stats.t.ppf(1 - alpha / 2, df=df_welch)
    return diff - t_crit * se, diff + t_crit * se


def _cohen_magnitude(abs_val: float) -> str:
    if abs_val < 0.2:
        return "negligible"
    elif abs_val < 0.5:
        return "small"
    elif abs_val < 0.8:
        return "medium"
    else:
        return "large"


def _bootstrap_effect_size_ci(
    control_arr: np.ndarray,
    treatment_arr: np.ndarray,
    metric_type: str,
    n_bootstrap: int,
    alpha: float,
    seed: int = 42,
    chunk_size: int = 1_000,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for Cohen's h (proportions) or Cohen's d (continuous).

    Uses chunked vectorised resampling to keep peak memory below ~64 MB per chunk
    regardless of n_bootstrap. Each chunk draws `chunk_size` bootstrap samples.
    """
    rng = np.random.default_rng(seed)
    n_ctrl = len(control_arr)
    n_trt = len(treatment_arr)
    boot_stats: list[np.ndarray] = []

    remaining = n_bootstrap
    while remaining > 0:
        batch = min(chunk_size, remaining)
        ctrl_idx = rng.integers(n_ctrl, size=(batch, n_ctrl))
        trt_idx = rng.integers(n_trt, size=(batch, n_trt))

        ctrl_boot = control_arr[ctrl_idx]
        trt_boot = treatment_arr[trt_idx]

        if metric_type == "proportion":
            p_c = ctrl_boot.mean(axis=1)
            p_t = trt_boot.mean(axis=1)
            # Cohen's h: 2*arcsin(sqrt(p_t)) - 2*arcsin(sqrt(p_c))
            batch_stats = (
                2 * np.arcsin(np.sqrt(np.clip(p_t, 0.0, 1.0)))
                - 2 * np.arcsin(np.sqrt(np.clip(p_c, 0.0, 1.0)))
            )
        else:
            ctrl_means = ctrl_boot.mean(axis=1)
            trt_means = trt_boot.mean(axis=1)
            ctrl_std = ctrl_boot.std(axis=1, ddof=1)
            trt_std = trt_boot.std(axis=1, ddof=1)
            pooled_std = np.sqrt(
                ((n_ctrl - 1) * ctrl_std ** 2 + (n_trt - 1) * trt_std ** 2)
                / (n_ctrl + n_trt - 2)
            )
            batch_stats = np.where(
                pooled_std > 0, (trt_means - ctrl_means) / pooled_std, 0.0
            )

        boot_stats.append(batch_stats)
        remaining -= batch

    all_stats = np.concatenate(boot_stats)
    return (
        float(np.percentile(all_stats, 100 * alpha / 2)),
        float(np.percentile(all_stats, 100 * (1 - alpha / 2))),
    )


def _print_test_summary(result: dict) -> None:
    sig_str = "SIGNIFICANT" if result["significant"] else "not significant"
    print(
        f"Test: {result['primary_test']}\n"
        f"  p={result['p_value']:.6f}  stat={result['statistic']:.4f}  "
        f"95% CI={result['ci_95']}  [{sig_str}]\n"
    )

# Express Checkout A/B Experiment

Testing whether a 2-step pre-filled checkout increases purchase conversion for repeat buyers on a Shopee-style marketplace, and whether it does so without sacrificing order value or payment reliability.

---

## Key Findings

- **Conversion lifted +1.9pp in steady state** (approximately +11% relative), statistically significant at p < 0.001 after excluding the Week 1 novelty period. Including Week 1 would have overstated the lift by ~50%.
- **Both guardrail metrics passed**: AOV was flat (no meaningful drop), and payment failure rate increased by only 0.3pp, within the pre-specified tolerance of 0.5pp.
- **Effect is consistent across segments** but slightly stronger for mobile returning users, making them the natural starting point for a staged rollout.

---

## Why the Novelty Effect Catch Matters

Week 1 treatment conversion was inflated by approximately 2.2pp above the steady-state effect. An analyst who aggregated across all three weeks would have reported a lift of ~2.65pp — roughly 50% higher than the credible Week 2-3 estimate of ~1.76pp. That gap is not a rounding issue. A 2.65pp result supports a clean full rollout; a 1.76pp result with a CI lower bound grazing the MML supports a staged rollout with a 30-day holdout. These are different product decisions. The novelty effect check is not a methodological nicety — it changes what you tell the PM.

---

## Methodology

The experiment ran for 21 days with 12,000 user-level sessions, randomised 50/50 into control (standard 4-step checkout) and treatment (Express Checkout). Statistical parameters were pre-specified before any data was examined: alpha = 0.05 two-tailed, power = 0.80, minimum meaningful lift = 1.5pp absolute (justified by a GMV impact calculation).

Primary metric analysis used a two-proportion z-test. AOV analysis used a Welch t-test with Levene variance check, then Mann-Whitney U as the primary result given the right-skewed distribution caused by whale buyers (~2.5% of converters with order values above 5 standard deviations). Effect sizes are reported with 95% bootstrap CIs (10,000 resamples, percentile method). Segmentation analysis applies Holm step-down correction across p-values within each analytical dimension, controlling the family-wise error rate while remaining less conservative than Bonferroni. All assumption checks are documented inline in the notebook, not assumed.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate the synthetic dataset
python data/generate_dataset.py

# Open the notebook
jupyter notebook express_checkout_experiment.ipynb
```

Python 3.9+ required.

---

## Skills Demonstrated

| Area | Specifics |
|---|---|
| Statistical testing | Two-proportion z-test, Welch/Student t-test selection via Levene, Mann-Whitney U, chi-squared SRM test |
| Assumption checking | Shapiro-Wilk capped at 5,000 obs (large-sample sensitivity caveat documented), Levene variance check, CLT validation |
| Effect size | Cohen's h and Cohen's d with 95% bootstrap CIs (10,000 resamples, percentile method, chunked for memory) |
| Experiment design | Power analysis via Fleiss formula, MML derived from GMV calculation, pre-specified guardrail thresholds |
| Data quality | Novelty effect detection (Week 1 excluded; ~51% overstatement quantified), whale buyer identification, SRM test, structural missing data validation |
| Product thinking | Business-grounded MML, staged rollout with explicit 30-day holdout rationale, downstream metric plan |
| Python | Modular utility library (`ab_testing_utils.py`), PEP 8, pandas, scipy, statsmodels, matplotlib |
| Known limitation | Experiment ran 21 days vs. the designed 30, leaving the primary analysis at ~55–60% power against the 1.5pp MML; acknowledged and discussed inline |

---

## Project Structure

```
ab-testing-express-checkout/
├── data/
│   ├── generate_dataset.py       # Reproducible synthetic data generator
│   └── checkout_experiment.csv   # Generated on first run
├── ab_testing_utils.py           # Reusable A/B testing functions
├── express_checkout_experiment.ipynb
├── requirements.txt
└── README.md
```

---

*Adelia Ramadhani P. | adeliaramp@gmail.com

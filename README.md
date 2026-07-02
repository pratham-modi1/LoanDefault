# LoanGuard — Loan Default Predictor

An explainable machine learning system that predicts whether a borrower will experience a serious loan delinquency (90+ days past due) within two years, and tells you *why* — down to the individual borrower — using SHAP.

Built on the "Give Me Some Credit" dataset (150K borrowers, 10 raw features). Final model: **XGBoost**, deployed as an interactive **Streamlit** app with real-time SHAP explanations.

---

## Why This Project Exists

Most loan-default tutorials stop at "here's an accuracy score." This project deliberately does the opposite:

- It **rejects accuracy** as a metric on day one, because a model that always predicts "no default" is 93% accurate and 0% useful on this dataset's 14:1 class imbalance.
- It builds a **baseline ladder** (dummy → logistic regression → random forest → XGBoost) specifically to prove, mathematically, whether added model complexity is worth anything.
- It treats **SHAP as an audit**, not a nice-to-have — the EDA stage makes formal, falsifiable predictions about what a healthy model *should* learn, and the SHAP stage checks the model's actual behavior against that prediction.

That audit loop — predict in EDA, verify in SHAP — is the spine of the whole project.

---

## Results at a Glance

| Metric | Score |
| --- | --- |
| Precision | 21.8% |
| Recall | 75.0% |
| Accuracy | 80.5% |
| PR-AUC | 0.3755 |
| ROC-AUC | 0.8542 |
| Decision threshold | 0.1883 |

**Read this honestly:** the model catches 3 out of 4 real defaulters, at the cost of also flagging a lot of safe borrowers (roughly 4 in 5 flagged borrowers turn out fine). That trade-off was a deliberate business decision, not an accident — see [Modeling Philosophy](#modeling-philosophy--why-precision-is-low-on-purpose) below.

A linear model (Logistic Regression) and a 400-tree XGBoost ensemble land within 0.018 PR-AUC of each other. This is the project's central finding: **the ceiling here is the dataset (10 raw features), not the algorithm.** With richer data (credit bureau pulls, transaction history), published benchmarks suggest PR-AUC of 0.75+ is achievable on this same problem.

---

## Pipeline Overview

```
Raw CSV (150K rows, 10 features)
        │
        ▼
┌───────────────────────┐
│ 1. Setup & Understanding │  Column reorganization, target definition
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 2. Preprocessing         │  Dedup → RF-imputed income / median-imputed
│                           │  dependents → logical + statistical outlier
│                           │  handling (capping + log transform)
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 3. EDA                   │  Class imbalance quantified → Spearman signal
│                           │  hierarchy established → this becomes the
│                           │  "answer key" for the SHAP audit later
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 4. Modeling               │  Baseline ladder (Dummy → LogReg → RF →
│                           │  XGBoost) proves the algorithm isn't the
│                           │  bottleneck → 17 engineered features →
│                           │  hyperparameter scan → recall-targeted
│                           │  threshold (75% recall guaranteed)
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 5. SHAP Interpretability │  Global + local explanations → Anti-
│                           │  Hallucination Audit checks the model
│                           │  against the Stage 3 answer key → PASSED
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 6. Deployment (app.py)   │  Streamlit UI: user input → feature
│                           │  engineering → saved scaler → saved model
│                           │  → live SHAP waterfall per borrower
└───────────────────────┘
```

---

## Modeling Philosophy — Why Precision Is Low on Purpose

This is the part most write-ups skip, so it's worth stating explicitly since it explains a number that looks bad in isolation (21.8% precision).

1. **The baseline ladder proved algorithm choice barely matters.** Logistic Regression, Random Forest, and XGBoost all plateaued at PR-AUC ≈ 0.364 on the raw features. When three structurally different algorithms converge on the same score, the bottleneck is the data, not the model.
2. **Feature engineering, not more modeling, moved the needle.** 17 domain-driven features (e.g. `Struggle_Index = Utilization × Total_Late_Events`) pushed PR-AUC from 0.364 → 0.3755. SHAP later confirmed these engineered features — not the raw columns — as the top predictors.
3. **SMOTE was considered and rejected.** Recall was already high pre-engineering (73%); the bottleneck was a data ceiling, not insufficient exposure to minority-class examples. Adding synthetic minority samples risked reinforcing noise rather than signal.
4. **The threshold was chosen for the business problem, not for a balanced-looking metric table.** Missing a real defaulter costs a bank far more than an unnecessary manual review of a safe borrower. So instead of maximizing F1 (which would land around 0.75 threshold, ~40% precision, ~48% recall), the final model uses **the highest threshold that still guarantees ≥75% recall** (which turns out to be 0.1883). This is a explicit, documented choice to prioritize catching defaulters over minimizing false alarms — the model is meant to function as a first-pass screening tool, not a final decision-maker.

---

## The Anti-Hallucination Audit

This is the project's signature move, and it's worth calling out on its own.

In **Stage 3 (EDA)**, before any model was trained, Spearman correlations were used to predict which features *should* matter:

| Predicted strong signal | Predicted weak/negligible |
| --- | --- |
| NumberOfTimes90DaysLate | MonthlyIncome |
| RevolvingUtilizationOfUnsecuredLines | DebtRatio (alone) |
| 30/60-day late counts | NumberOfDependents |

In **Stage 5 (SHAP)**, after the final model was trained, the actual global SHAP ranking was checked against this prediction. Every top-5 SHAP feature traced back to either credit utilization or late-payment behavior — exactly the two categories EDA flagged. `MonthlyIncome` ranked #13 of 25, precisely where its near-zero correlation predicted it would land.

**Verdict: the model learned genuine financial behavior, not spurious correlations in the training data.** If the SHAP ranking had contradicted the EDA prediction (e.g. `MonthlyIncome` ranking #1), that would have been a red flag for leakage or overfitting — and the pipeline was designed so that failure mode would be caught before deployment, not after.

---

## Key Engineered Features

Raw tree models can't infer that "3 counts in the 90-day bucket" is categorically worse than "3 counts in the 30-day bucket" — they see three unrelated numbers. Feature engineering translates domain knowledge into signals the model can actually use:

| Feature | What it captures |
| --- | --- |
| `Struggle_Index` | Utilization × total late events — the compounded state of being maxed-out **and** delinquent. Ranked #1 in final SHAP importance. |
| `Weighted_Late_Score` | Severity-weighted delinquency (30-day×1, 60-day×2, 90-day×3) — encodes that a 90-day default is worse than three 30-day slips. |
| `Remaining_Living_Money` | Actual dollars left after debt obligations — a direct solvency proxy, computed by un-logging income (`np.expm1`) before doing dollar arithmetic. |
| `Youth_Utilization_Risk` | High utilization is riskier for younger borrowers with shorter credit histories. |
| `Utilization_Squared` | Penalizes extreme utilization non-linearly. |

Full list and formulas: see `reports/report3.md`.

---

## Repository Structure

```
LoanDefault/
│
├── 1_Project.py          # Full training pipeline: preprocessing, feature
│                          #   engineering, model training, SHAP generation
├── app.py                # Streamlit UI (Home / Predict / Visualize / Report)
├── best_xgb_model.joblib # Trained XGBoost model (generated by 1_Project.py)
├── data_scaler.joblib    # Fitted StandardScaler (generated by 1_Project.py)
├── requirements.txt
├── .gitignore             # excludes CSV dataset files
│
├── reports/               # Full stage-by-stage documentation
│   ├── report1.md         # Stage 0–1: Setup, data understanding, preprocessing
│   ├── report2.md         # Stage 2: EDA & signal hierarchy
│   ├── report3.md         # Stage 3–6: Modeling, feature engineering, thresholding
│   ├── report4.md         # Stage 4 (SHAP): interpretability & audit
│   └── report5.md         # Code architecture documentation
│
└── utility/
    ├── DataDictionary.txt
    ├── ModelDevelopment.txt
    └── ROCevaluation.txt
```

> **Naming note:** `reports/report5.md` refers to the UI file as `streamlit_app.py`. Your actual file is `app.py` — this README and the commands below use the real filename. Worth a quick find/replace in report5.md if you want the docs to match the repo exactly.

---

## Setup & Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (only needed once — subsequent runs load the saved .joblib files)
python 1_Project.py

# 3. Launch the app
streamlit run app.py
```

`1_Project.py` needs `CS-training.csv` in the project root (excluded from git via `.gitignore`). Training takes ~30–45 seconds on first run; `app.py` needs `best_xgb_model.joblib` and `data_scaler.joblib` present before it will start.

---

## Tech Stack

| Layer | Tools |
| --- | --- |
| Data processing | pandas, numpy |
| Imputation | scikit-learn `RandomForestRegressor` (income), median (dependents) |
| Modeling | scikit-learn (LogReg, RF baselines), XGBoost (final model) |
| Interpretability | SHAP (`TreeExplainer`) |
| Deployment | Streamlit |
| Persistence | joblib |

---

## Known Limitations

- **Feature count.** 10 raw input features vs. 200–400 typically used in production credit scoring. This is the primary driver of the PR-AUC ceiling (~0.37).
- **Precision.** At the deployed threshold, ~4 of 5 flagged borrowers are actually safe. Acceptable for a first-pass screening tool, not for a fully automated denial system.
- **No fairness audit.** Performance has not been evaluated across demographic subgroups — required before any real-world lending use.
- **Not probability-calibrated.** Output scores rank borrowers well (ROC-AUC 0.854) but should not be read as literal default probabilities without calibration.
- **Geographic scope.** Dataset is US-centric; may not generalize elsewhere.

---

## Possible Next Steps

- Credit bureau data / transaction history integration (largest expected PR-AUC gain)
- Model versioning + drift monitoring
- Fairness/bias audit across protected attributes
- Replace the Streamlit UI with an API endpoint for system integration
- Automated retraining pipeline

---

## Full Documentation

Each stage of the project is documented in depth under `reports/`:

- **`report1.md`** — Repository setup, dataset understanding, missing-value and outlier handling
- **`report2.md`** — EDA: class imbalance, Spearman signal hierarchy, the SHAP "answer key"
- **`report3.md`** — Baseline ladder, feature engineering, hyperparameter tuning, final threshold strategy
- **`report4.md`** — SHAP analysis, the Anti-Hallucination Audit, global/local interpretation
- **`report5.md`** — Line-by-line code architecture for `1_Project.py` and the Streamlit app
# Stage 5: Final Code Documentation & Architecture

---

## Overview

This report documents the complete code architecture of the Loan Default Predictor project. It explains every major design decision, what each code block does, why it was written that way, and what would break if it were removed or changed.

| File | Purpose |
| --- | --- |
| `1_Project.py` | Core ML pipeline (training, evaluation, SHAP) |
| `streamlit_app.py` | Streamlit web UI for interactive predictions |

---

## Section 1: Data Loading and Column Organization

**File:** `1_Project.py` — Lines 1–45

### What It Does

Loads the CSV dataset using pandas and immediately reorganizes columns into four semantic categories before any processing begins.

### Why Columns Are Reorganized First

Keeping raw data in its default column order (which is effectively random from a domain perspective) makes the code harder to audit and the dataset harder to understand at a glance.

Grouping into semantic categories serves as living documentation — anyone reading the code instantly understands the data structure without referring to an external data dictionary.

### The Four Categories

| Category | Features |
| --- | --- |
| Credit Usage | RevolvingUtilizationOfUnsecuredLines |
| Payment Behavior | 30-day, 60-day, 90-day late payment columns |
| Financial Capacity | DebtRatio, MonthlyIncome |
| Personal Profile | age, open credit lines, real estate loans, dependents |
| Target | SeriousDlqin2yrs (the label being predicted) |

### Key Design Decision

The `Unnamed: 0` column drop is performed defensively. When CSV files are saved with pandas' default index and then re-read, an unnamed integer column appears. Dropping it explicitly prevents accidental use of row numbers as a feature.

---

## Section 2: Missing Value Imputation

**File:** `1_Project.py` — Lines 46–95

### MonthlyIncome — RandomForestRegressor Imputation

**Why a Random Forest instead of median or mean:**

`MonthlyIncome` has approximately 20% missing values. A flat median replacement would stamp the same $5,400 on every missing record, creating an artificial cluster that disrupts tree splits and distorts the relationship between income and other variables.

A Random Forest trained on the 80% of records with known income learns how income correlates with age, credit behavior, and debt patterns. It then generates personalized income estimates for the 20% with missing values.

| Approach | PR-AUC | Precision |
| --- | --- | --- |
| RandomForest Imputer | 0.3755 | 41% |
| Median Imputer | 0.3754 | 37% |

### The `MonthlyIncome_was_missing` Flag

Before imputation, a binary indicator column is created. If the flag is 1, the income value was estimated, not reported.

EDA validation showed borrowers with missing income default at **5.50%** vs **6.89%** for borrowers who reported income. This 1.39% difference proves the missingness itself carries predictive signal. The flag is retained as a feature.

### NumberOfDependents — Median Imputation

This column has only 2–5% missing values and is a small integer (0–10). For a feature with low variance and weak correlation to the target, a Random Forest imputer would overfit to noise. Median imputation is simpler, more stable, and appropriate here.

---

## Section 3: Outlier Handling

**File:** `1_Project.py` — Lines 96–150

### Three-Layer Strategy

| Layer | Method | Action |
| --- | --- | --- |
| Layer 1 | Domain-Based Filtering | Hard remove of impossible values (age < 18, negatives) |
| Layer 2 | Encoded Anomaly Removal | Remove rows where late payment columns ≥ 90 (sentinel values 96/98) |
| Layer 3 | Statistical Treatment | Capping (Winsorization) + Log Transformation for extreme but valid values |

### Layer 3 Details

| Feature | Action | Threshold | Reason |
| --- | --- | --- | --- |
| RevolvingUtilizationOfUnsecuredLines | Capping | 99th percentile | Extreme values (e.g., 50,000%) are unrealistic |
| DebtRatio | Capping | 99th percentile | Values above 300,000 are likely anomalies |
| MonthlyIncome | Capping + Log Transform | 99.9th percentile + `np.log1p()` | Heavily right-skewed; log compression makes differences meaningful |

> `log1p` is used — i.e., `log(1+x)` — to safely handle zero values.

---

## Section 4: Feature Engineering Block

**File:** `1_Project.py` — Lines 151–250

### Design Philosophy

Every engineered feature translates a domain insight (what a bank analyst would think about) into a mathematical signal (what an algorithm can compute). Raw data columns are the "what." Engineered features are the "what does this mean financially."

---

### The Inverse Log Trick (`raw_income`)

`MonthlyIncome` was log-transformed for model training. But log-scaled income cannot be used for dollar arithmetic — you cannot subtract log-scaled debt from log-scaled income and get a meaningful result.

```python
raw_income = np.expm1(df['MonthlyIncome'])
```

`expm1` is the mathematical inverse of `log1p`. It "un-logs" the income column, restoring it to its original dollar scale so that financial formulas (income − debt = remaining money) compute correctly. After calculations, results are re-scaled with `np.log1p()` where appropriate.

> **Critical:** `raw_income` must be computed AFTER the log transformation of `MonthlyIncome` is applied. Computing it before produces incorrect values because `expm1` expects already-log-transformed input.

---

## Section 5: Data Splitting and Scaling

**File:** `1_Project.py` — Lines 251–295

### The 80/20 Stratified Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

`stratify=y` is non-negotiable. Without it, random splitting on a 6.6% minority class risks creating a test set with 3% or 10% defaulters by chance, making all evaluation metrics meaningless. Stratification guarantees both sets have exactly 6.6% defaulters.

### StandardScaler — Fit Only on Training Data

| Operation | Code | Why |
| --- | --- | --- |
| Training set | `scaler.fit_transform(X_train)` | Learns rules AND applies them |
| Test set | `scaler.transform(X_test)` | Applies learned rules ONLY |

**This ordering is critical.** If the scaler is fit on test data, it learns different mean and standard deviation values from the test distribution. The same raw value then gets a different scaled value depending on which set it belongs to. This is **data leakage** — the model sees information from the test set during training, invalidating all evaluation results.

### Saving the Scaler

```python
joblib.dump(scaler, 'data_scaler.joblib')
```

This is essential for deployment. When a user submits their data in the UI, it must be scaled using the exact same μ and σ values that the model was trained on. If the scaler is rebuilt from the user's data, the scaled values will be wrong and the model will produce garbage predictions. The saved scaler is the "rule book" that must travel with the model.

---

## Section 6: Feature Ceiling Diagnostic

**File:** `1_Project.py` — Lines 296–310

### What It Does

Trains a Logistic Regression on the current features and prints its PR-AUC as a diagnostic.

### Why This Diagnostic Exists

Logistic Regression is a linear model — it cannot learn non-linear patterns, interactions, or complex decision boundaries. Its PR-AUC represents the maximum performance achievable with linear combinations of the current features.

| Model | PR-AUC | Interpretation |
| --- | --- | --- |
| Logistic Regression | 0.3576 | Linear ceiling of the feature set |
| XGBoost | 0.3755 | Non-linear advantage |
| Gap | 0.0179 | Very small — confirms features are the bottleneck, not the algorithm |

---

## Section 7: XGBoost Training Block

**File:** `1_Project.py` — Lines 311–385

### The Save/Load Architecture

```python
if os.path.exists(model_filename):
    best_xgb = joblib.load(model_filename)
else:
    [train and save]
```

Training the final XGBoost takes approximately 30–45 seconds. Without save/load, every code run would require this wait.

| Run | Behavior | Time |
| --- | --- | --- |
| First run | Train, scan, save | 30–45 seconds |
| All later runs | Load from disk | ~0.1 seconds |

> Delete the `.joblib` file to force retraining when the data or features change.

### The `scale_pos_weight` Scan

| `scale_pos_weight` | Effect |
| --- | --- |
| 14 (natural ratio) | Very aggressive — predicts "Default" liberally, precision collapses to 22% |
| 3 (selected by scan) | More conservative — flags only borrowers with stronger evidence, precision improves |

The scan tests [3, 5, 6, 7, 10] and picks the value that maximizes PR-AUC. In this project, **spw=3 was the winner.**

### `eval_metric='aucpr'`

Training with `logloss` but evaluating with PR-AUC is like studying math but being tested on English — the preparation doesn't match the exam. Using `aucpr` aligns training with evaluation.

---

## Section 8: Threshold and Final Output

**File:** `1_Project.py` — Lines 386–410

### The 75% Recall Target

```python
idx = np.where(recall >= 0.75)[0][-1]
```

`precision_recall_curve()` returns three arrays: `precision[]`, `recall[]`, and `thresholds[]`.

| Code Part | Meaning |
| --- | --- |
| `np.where(recall >= 0.75)` | All indices where recall is still at least 75% |
| `[0]` | Extract the array from the tuple |
| `[-1]` | The last index — i.e., the highest threshold that still guarantees 75% recall |

This is the optimal operating point: the most precise threshold that still meets the recall target.

---

## Section 9: SHAP Analysis Block

**File:** `1_Project.py` — Lines 411–500

### TreeExplainer vs Generic Explainer

`shap.TreeExplainer(best_xgb)` is used rather than `shap.Explainer()`. TreeExplainer uses the exact tree structure to compute SHAP values **analytically** — no sampling, no approximation. This is both faster and more accurate for tree-based models.

### Waterfall Plot

`shap_values[0]` is the first person in `X_sample`. In the Streamlit UI, this is replaced with the user's own input, making the waterfall personal and relevant.

### Dependence Plot Design

```python
shap.dependence_plot("feature_name", shap_values.values, X_sample)
```

The second argument controls the color of dots (auto-selected by SHAP as the feature most correlated with the x-axis feature). This reveals **interaction effects** — how does the impact of Feature A change depending on the value of Feature B?

---

## Section 10: Streamlit UI Architecture

**File:** `streamlit_app.py`

### Page Structure

| Page | Purpose |
| --- | --- |
| Home | Landing page with project description |
| Predict | User inputs + risk assessment + SHAP explanation |
| Report | Full technical documentation (all stages) |

Navigation is implemented with `st.columns()` and `st.session_state` to simulate a multi-page app without page reloads.

---

### Input Processing Pipeline (Predict Page)

| Step | Action |
| --- | --- |
| Step 1 | Collect raw user inputs via sliders and number inputs |
| Step 2 | Apply all feature engineering transformations |
| Step 3 | Assemble into a single-row DataFrame with correct column order |
| Step 4 | Load scaler from disk and transform (DO NOT refit) |
| Step 5 | Load model from disk and predict probability |
| Step 6 | Compare probability to stored threshold |
| Step 7 | Compute SHAP values for this specific user |
| Step 8 | Display results, riskometer, and SHAP waterfall |

### The Critical Data Ordering Issue

The input DataFrame **must** have exactly the same columns in exactly the same order as the training data. XGBoost uses column positions, not column names, when the model is loaded from disk. A mismatch silently produces wrong predictions without raising an error.

> The column list is hardcoded in the same order as `X.columns` from the training code to prevent this failure mode.

### SHAP in the UI

SHAP is recomputed for every user submission:
- The explainer is initialized fresh for each prediction
- Results are specific to the user's exact input values
- No cached or averaged SHAP values are shown

This is slower (1–2 seconds per submission) but is the **only correct approach** for individual-level explanations.

### Dynamic Dependence Plots

Rather than hardcoding which features to show, the UI identifies the top 2 features with the highest absolute SHAP values for the current user and shows dependence plots for those. This makes the explanation always relevant to the specific person.

---

## Section 11: Files and Dependencies

### Required Files at Runtime

| File | Purpose |
| --- | --- |
| `cs-training.csv` | Raw dataset (training only) |
| `best_xgb_model.joblib` | Trained XGBoost model |
| `data_scaler.joblib` | Fitted StandardScaler |

### Python Dependencies

| Library | Purpose |
| --- | --- |
| pandas, numpy | Data manipulation |
| scikit-learn | Preprocessing, evaluation, splitting |
| xgboost | Core prediction model |
| shap | Interpretability |
| matplotlib, seaborn | Visualization (training code) |
| streamlit | Web UI |
| joblib | Model persistence |
| os | File existence checks |

### Running the Code

```bash
# Train and save the model (run once)
python 1_Project.py

# Launch the web application
streamlit run streamlit_app.py
```

> Requires both `.joblib` files to exist in the same directory before launching the app.

---

## Section 12: Known Limitations and Future Improvements

### Dataset Limitations

| Limitation | Detail |
| --- | --- |
| Feature count | Only 10 input features vs 200–400 in real credit scoring |
| PR-AUC ceiling | 0.37 is a data limitation, not a model limitation |
| Geographic scope | US-centric dataset may not generalize to other geographies |

### Model Limitations

| Limitation | Detail |
| --- | --- |
| Precision | 21% — 4 of 5 flagged borrowers are safe |
| Recall | 75% — 25% of real defaulters still slip through |
| Calibration | Not calibrated for absolute probability accuracy |

### Improvement Paths with Better Data

| Data Source | Expected Gain |
| --- | --- |
| Credit bureau pull data (payment history on every account) | High |
| Employment and income verification | High |
| Banking transaction history | Medium |
| Public records (liens, judgments, bankruptcies) | Medium |

> With these features, PR-AUC of 0.75+ would be achievable based on published academic benchmarks on richer credit datasets.

### Code Improvements for Production

| Improvement | Description |
| --- | --- |
| Input validation | Range checking in the UI |
| Model versioning | Drift detection over time |
| API endpoint | Replace Streamlit for system integration |
| Automated retraining | Pipeline to retrain on new data |
| Fairness audit | Evaluate performance across demographic groups |

---

**Project Status: Complete**

| Summary | Detail |
| --- | --- |
| Total Stages | 6 (Setup, Preprocessing, EDA, Feature Engineering, Modeling, SHAP Analysis) |
| Final Model | XGBoost (`best_xgb_model.joblib`) |
| UI | Streamlit (`streamlit_app.py`) |
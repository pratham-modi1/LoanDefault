# Stage 3: Model Development — Establishing Baselines

---

## Objective

Before any serious model is built, a systematic baseline ladder must be established. The purpose is not to build the best model immediately, but to mathematically prove how much each layer of complexity is actually contributing. If a complex model performs the same as a simple one, the complexity is wasted effort.

This stage follows a strict progression:

| Step | Model | Question Being Answered |
| --- | --- | --- |
| Step 1 | Dummy Classifier | Can we do better than random? |
| Step 2 | Evaluation Framework | Are we measuring fairly? |
| Step 3 | Logistic Regression | Is there linear signal in the data? |
| Step 4 | Random Forest | Does non-linearity help? |
| Step 5 | XGBoost | Does sequential boosting help? |

---

## Step 1: Dummy Classifier — The Random Guessing Baseline

### What It Is

A `DummyClassifier` with `strategy='stratified'` generates predictions randomly, but preserves the dataset's class distribution. It does not look at features at all.

### Why We Need It

Without a random baseline, there is no way to measure whether a model is actually learning anything. A model that achieves 93% accuracy on a 93/7 imbalanced dataset may have simply learned to always predict "No Default." The dummy baseline exposes this trap.

### Results

| Metric | Score |
| --- | --- |
| Precision | 0.0622 |
| Recall | 0.0623 |
| F1 Score | 0.0623 |
| PR-AUC | 0.0659 |
| ROC-AUC | 0.4979 |

### Interpretation

PR-AUC of 0.0659 is approximately equal to the proportion of the positive class in the dataset (6.6%), confirming pure random behavior. ROC-AUC of 0.5 confirms zero discriminative power.

> **Conclusion:** This is the absolute minimum. Every subsequent model must beat this significantly to justify its existence.

---

## Step 2: Standardized Evaluation Framework

### Problem with Inconsistent Evaluation

If different models are evaluated at different thresholds or with different metrics, comparisons become meaningless. A model that looks better may simply have been evaluated more generously.

### The Framework Established

- Same test dataset for all models (no exceptions)
- Same default threshold of 0.5 for all baseline comparisons
- Same five metrics for every model: Precision, Recall, F1 Score, PR-AUC, ROC-AUC

### Why PR-AUC Was Chosen as the Primary Metric

Standard accuracy is formally rejected for this project. On a 93.37% vs 6.63% imbalanced dataset, a lazy model that always predicts "No Default" achieves 93% accuracy while completely failing to catch a single defaulter.

ROC-AUC is also rejected as primary. A model can rank borrowers reasonably well while still setting a terrible decision threshold.

PR-AUC measures performance specifically on the minority class across all thresholds. It is the most honest metric for imbalanced problems.

### Evaluation Hierarchy

| Priority | Metric | Reason |
| --- | --- | --- |
| Primary | PR-AUC | Overall discriminative power on minority class |
| Secondary | Recall | Catching actual defaulters — the bank's top need |
| Tertiary | Precision | Avoiding false alarms — the bank's secondary need |
| Reference | F1, ROC-AUC | Balance indicators |

---

## Step 3: Logistic Regression — The Sanity Check

### Why Logistic Regression First

Logistic Regression is the simplest classification model that actually learns from data. It tests one specific hypothesis: are there linear relationships between features and default risk?

If even a straight-line model can significantly outperform random guessing, it proves that the dataset contains genuine predictive signal. This is the most important test of all — before wasting time on complex algorithms, verify that the data is worth modelling.

**Configuration:** `class_weight='balanced'`, `max_iter=2000`

### Results vs Dummy Classifier

| Metric | Dummy | Logistic Regression | Change |
| --- | --- | --- | --- |
| Precision | 0.062 | 0.212 | +241% |
| Recall | 0.062 | 0.731 | +1079% |
| F1 Score | 0.062 | 0.329 | — |
| PR-AUC | 0.066 | 0.364 | +451% |
| ROC-AUC | 0.498 | 0.847 | — |

### Interpretation

A 5.5x improvement in PR-AUC from a simple linear model is a strong positive signal. The data is learnable. The patterns are real.

However, Recall at 73% comes at the cost of Precision at 21%, meaning approximately 4 out of every 5 flagged borrowers are safe. This is the classic imbalanced classification problem — the model casts too wide a net.

> **Conclusion:** The dataset contains strong predictive signal. A linear model is already surprisingly effective. This sets a high bar for tree-based models to beat.

---

## Step 4: Random Forest — Non-Linear Complexity Test

### Hypothesis

If Logistic Regression captures linear relationships, perhaps a Random Forest can find non-linear patterns and interactions that a straight line cannot represent.

**Configuration:** `n_estimators=100`, `max_depth=10`, `min_samples_leaf=20`, `class_weight='balanced'`

### Results vs Logistic Regression

| Metric | Logistic Regression | Random Forest | Change |
| --- | --- | --- | --- |
| Precision | 0.2122 | 0.2201 | +0.0079 |
| Recall | 0.7300 | 0.7285 | -0.0015 |
| F1 Score | 0.3288 | 0.3381 | +0.0093 |
| PR-AUC | 0.3638 | 0.3640 | +0.0002 |
| ROC-AUC | 0.8472 | 0.8527 | +0.0055 |

### Interpretation

The improvement is almost zero. A model with 100 trees, non-linear splits, and interaction detection adds essentially nothing over a single straight line.

This means one of two things:
1. The problem is largely linearly separable, OR
2. The features do not contain enough information for complex models to exploit

Both interpretations point to the same conclusion: **the bottleneck is not model complexity — it is the features and the data itself.**

> **Conclusion:** Random Forest is not the answer. Adding complexity without improving features is wasted effort.

---

## Step 5: XGBoost — Boosting as the Final Baseline

### Why XGBoost

Unlike Random Forest which builds trees in parallel and averages them, XGBoost builds trees sequentially. Each new tree specifically corrects the mistakes made by all previous trees. This makes XGBoost highly effective at learning from difficult-to-classify minority class examples.

Additionally, XGBoost has a built-in parameter called `scale_pos_weight` that directly tells the algorithm how much extra attention to give to the minority class during training.

**Configuration:** `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight≈14`

### Full Model Comparison

| Metric | Logistic Reg | Random Forest | XGBoost |
| --- | --- | --- | --- |
| Precision | 0.2122 | 0.2201 | 0.2248 |
| Recall | 0.7300 | 0.7285 | 0.7356 |
| F1 Score | 0.3288 | 0.3381 | 0.3444 |
| PR-AUC | 0.3639 | 0.3640 | 0.3638 |
| ROC-AUC | 0.8472 | 0.8527 | 0.8522 |

### The PR-AUC Plateau — Critical Insight

All three models achieve essentially identical PR-AUC of **0.364**. This is not a coincidence. It is a mathematical signal.

When three fundamentally different algorithms — a linear model, a parallel ensemble, and a sequential booster — all produce the same PR-AUC, the bottleneck has been identified with certainty: **it is the features, not the algorithm.**

PR-AUC of 0.364 was a hard ceiling that all models hit regardless of architecture. No matter how sophisticated the algorithm, the existing 10 raw features cannot provide more discriminative power for the minority class.

> **Conclusion:** XGBoost is selected as the base algorithm going forward. But the real next step is not a better model — it is better features.

---

## Stage 4: The Strategic Decision Point — SMOTE vs Threshold Optimization

### Context

After establishing the baseline plateau at PR-AUC ≈ 0.364, two potential paths forward were identified and formally compared.

### Option 1 — SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates synthetic examples of the minority class by interpolating between existing defaulter records.

| When SMOTE Works | Why SMOTE Was Rejected Here |
| --- | --- |
| Recall is low (model is missing defaulters) | Recall was already at 73% |
| Model is biased toward the majority class | PR-AUC plateau across all algorithms suggests a data ceiling |
| The model hasn't seen enough minority examples | Adding synthetic noise to an already-learned pattern risks overfitting |

### Option 2 — Threshold Optimization (Selected)

By default, models classify as "Default" when probability > 0.5. On an imbalanced dataset with only 6.6% defaulters, this cutoff is mathematically inappropriate.

Threshold optimization uses the Precision-Recall curve to find the exact probability cutoff that maximizes the F1 Score.

**Decision:** Threshold Optimization was selected. The model already knows how to rank borrowers by risk. The problem is the decision boundary, not the learning. Changing the threshold is a zero-cost intervention that directly addresses the root cause.

### Results After Threshold Optimization

| Model | Precision | Recall | F1 Score | Threshold |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.3684 | 0.4772 | 0.4158 | 0.7361 |
| Random Forest | 0.3967 | 0.4438 | 0.4189 | 0.7611 |
| XGBoost | 0.3612 | 0.4970 | 0.4183 | 0.7464 |

**Key Discovery — The High Threshold:** The optimal threshold is approximately 0.74–0.76, far above the default 0.5. The model only assigns high confidence scores to genuinely risky borrowers. Most borderline predictions sit between 0.1 and 0.4 — the model is being cautious by default.

**Improvement without any code changes:** Precision jumped from 22% to 36–39%. F1 Score improved from 0.34 to 0.42.

---

## Stage 5: Advanced Optimization — Breaking the Ceiling

### Phase 1: Feature Engineering

The core problem: tree-based algorithms are mathematically literal. When given three separate columns for 30-day, 60-day, and 90-day late payments, the model treats them as three independent variables. It cannot inherently understand that chronic 90-day delinquency is categorically more severe than a one-time 30-day delay.

The solution is domain-driven feature engineering.

| Feature | Formula | Reasoning |
| --- | --- | --- |
| Total_Late_Events | Sum of all three late payment columns | Captures raw volume of delinquency |
| Weighted_Late_Score | (30-day × 1) + (60-day × 2) + (90-day × 3) | Severity weighting — not all late payments are equal |
| Credit_Utilization_Warning | Binary flag if Utilization > 0.85 | Gives tree models a clean split point at the behavioral cliff |
| Income_Per_Dependent | raw_income / (Dependents + 1), log-scaled | Normalizes income to reflect actual financial capacity |
| Absolute_Monthly_Debt | DebtRatio × raw_income | Converts ratio into actual dollars of monthly debt |
| Remaining_Living_Money | raw_income − Absolute_Monthly_Debt | Cash left after all debt obligations — direct default proxy |
| Is_Cash_Negative | Binary flag if Remaining_Living_Money < 0 | Clean signal for technical insolvency |
| Low_Buffer_Flag | Binary flag if Remaining_Living_Money < $500 | Identifies financially fragile borrowers |
| Has_Any_Late | Binary flag if Total_Late_Events > 0 | Sharpest binary split — zero history vs any history |
| Max_Late_Severity | Count of severity buckets crossed (0–3) | Captures depth of financial distress, not just volume |
| Youth_Utilization_Risk | Utilization × (1 / (age − 17)) | High utilization is more dangerous for younger borrowers |
| Debt_Squeeze | DebtRatio × Is_Cash_Negative | High debt ratio combined with negative cash — near-certain default indicator |
| Severity_Per_Line | Weighted_Late_Score / (OpenCreditLines + 1) | Normalizes delinquency by number of opportunities |
| Struggle_Index | Utilization × Total_Late_Events | Interaction of two strong signals — maxed out AND not paying |
| Age_Debt_Interaction | Absolute_Monthly_Debt / (age + 1) | Young people with high debt are in a more precarious position |
| Late_to_Open_Ratio | Total_Late_Events / (OpenCreditLines + 1) | Proportion of financial commitments that are failing |
| Utilization_Squared | Utilization² | Exponential penalty for extreme utilization |

**Features Dropped:**

| Feature | Reason |
| --- | --- |
| NumberRealEstateLoansOrLines | Near-zero Spearman correlation confirmed in EDA |
| NumberOfDependents | Same reason — carries no meaningful signal |
| Absolute_Monthly_Debt | Superseded by Debt_Squeeze |

**Result of Feature Engineering:** XGBoost PR-AUC improved from 0.364 to **0.3755**.

---

### Phase 2: A/B Testing the Pipeline

| Test | Change | Result | Conclusion |
| --- | --- | --- | --- |
| Test A — Median Imputation | Replaced RF imputer with median for MonthlyIncome | PR-AUC dropped; Precision dropped from 41% to 37% | RF imputer is superior; income correlates with other features |
| Test B — Feature Pruning | Dropped raw 30/60/90-day columns, kept engineered features | PR-AUC recovered to 0.3758; Precision improved to 37.21% | Removing echo reduced multicollinearity |

**Combined Setup Comparison:**

| Setup | Precision | Recall | F1 Score |
| --- | --- | --- | --- |
| Median + Pruned | 37.21% | 48.28% | 0.4203 |
| RF + Not Pruned | 41.27% | 42.60% | 0.4192 |

These results reveal the classic precision-recall seesaw. Both setups are mathematically valid for different business contexts.

---

### Phase 3: Hyperparameter Scan

**Key Insight on Training Objective Mismatch:** Previously, XGBoost was trained using `eval_metric='logloss'` but evaluated on PR-AUC — optimizing for the wrong objective. Changed to `eval_metric='aucpr'` to align training with evaluation.

**Scale Pos Weight Scan:**

| scale_pos_weight | PR-AUC |
| --- | --- |
| 3 | **0.3755** ← Winner |
| 5 | 0.3753 |
| 6 | 0.3750 |
| 7 | 0.3748 |
| 10 | 0.3754 |

**Final Hyperparameters:**

| Parameter | Value |
| --- | --- |
| max_depth | 5 |
| learning_rate | 0.01 |
| n_estimators | 400 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 20 |
| gamma | 1 |
| scale_pos_weight | 3 |
| eval_metric | aucpr |

---

## Stage 6: The Precision-Recall Ceiling — Final Verdict

### The Dataset Ceiling Diagnostic

| Model | PR-AUC |
| --- | --- |
| Logistic Regression | 0.3576 |
| XGBoost (best) | 0.3755 |
| Difference | 0.0179 |

A difference of only 0.018 between the simplest possible model and the most complex, heavily engineered model is a definitive signal. When a straight-line model achieves essentially the same score as an ensemble of 400 boosted trees, **the features themselves cannot provide more signal.**

> Top Kaggle solutions on this exact dataset using external data sources achieve PR-AUC of 0.87+. The gap is entirely explained by data richness. Real credit scoring models use 200–400 variables. This dataset has 10.

### The Strategic Final Decision

| Option | Description | Outcome |
| --- | --- | --- |
| Option A | Chase 50/50 Precision-Recall balance | Best achievable was ~40/40; no practical path beyond this |
| Option B (Selected) | High Recall strategy with documented reasoning | Prioritize catching actual defaulters; accept higher false positive rate |

**Business Justification:** The cost of missing a real defaulter (lending $50,000 to someone who cannot repay) is dramatically higher than the cost of flagging a safe borrower for additional review. This asymmetry justifies prioritizing Recall. The model serves as a first-pass screening tool.

**Cascade Classifier — Considered and Rejected:**

The fatal math: Recall is multiplicative across stages.

- Model 1 Recall: 0.70 × Model 2 Recall: 0.20 = **0.14 overall recall**

The second model filters out both bad and good predictions, collapsing final recall to 14% — worse than a single model.

---

## Final Model Performance

| Metric | Score |
| --- | --- |
| Precision | 0.2176 |
| Recall | 0.7503 |
| Accuracy | 0.8049 |
| Threshold | 0.1883 |
| PR-AUC | 0.3755 |
| ROC-AUC | 0.8542 |

> The model successfully captures **75% of actual defaulters**. This is the maximum performance achievable with this dataset.
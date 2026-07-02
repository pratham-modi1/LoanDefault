# Stage 2: Exploratory Data Analysis

---

## Why EDA Is Non-Negotiable

A machine learning model has no judgment. It will learn whatever patterns exist in the data — including the noise, the artifacts, and the accidents — with equal enthusiasm. Without EDA, there is no way to distinguish what the model *should* learn from what it *will* learn if left unchecked.

EDA serves three concrete purposes in this project:

| Purpose | What It Prevents |
| --- | --- |
| Identifying the accuracy trap before it misleads the project | Building a 93% accurate model that never catches a single defaulter |
| Establishing which features carry real signal vs. noise | Wasting model capacity on variables that contribute nothing |
| Setting a verifiable ground truth for SHAP validation | Having no way to audit whether the model learned genuine financial behavior |

The order of analysis matters. The correlation matrix was computed *before* any visualization was produced. This is intentional — it filters out features that don't deserve visual investigation, preventing "Death by a Thousand Plots."

---

## Part 1: Target Variable Analysis — The Class Imbalance Problem

### What Was Done

The distribution of the target variable `SeriousDlqin2yrs` was analyzed to understand how many borrowers in the dataset actually defaulted versus those who did not.

### Findings

| Class | Label | Count (approx.) | Proportion |
| --- | --- | --- | --- |
| 0 | Non-Defaulter | ~139,974 | 93.37% |
| 1 | Defaulter | ~9,929 | 6.63% |

The class ratio is approximately **14:1** in favor of the majority (non-default) class.

---

### The Accuracy Trap — Why This Finding Is Alarming

This single distribution finding defines every evaluation decision made in the project.

A completely naive model — one that ignores all features and simply predicts "No Default" for every borrower — achieves **93.37% accuracy**. This number sounds impressive. It is completely useless.

In loan default prediction, the minority class (defaulters) is the entire point of the problem. The business does not need a model to identify people who will repay — it needs a model to identify people who *won't*. A model that never predicts "Default" has 0% practical value regardless of its accuracy number.

> **The Accuracy Trap is formally avoided from this point forward. Standard accuracy is rejected as an evaluation metric for this project.**

---

### Formal Metric Decision

| Priority | Metric | Business Reasoning |
| --- | --- | --- |
| Primary | PR-AUC | Measures discriminative power specifically on the minority class across all thresholds — the most honest metric for imbalanced problems |
| Secondary | Recall | Proportion of actual defaulters correctly caught — missing a defaulter costs the bank money |
| Tertiary | Precision | Proportion of flagged borrowers who are genuine defaulters — false alarms have a cost too |
| Reference Only | ROC-AUC, F1 | Reported for comparison; not used for model selection decisions |

**Why ROC-AUC is also rejected as primary:** A model can achieve a high ROC-AUC score by ranking borrowers reasonably well while still setting a terrible classification threshold. On a 14:1 imbalanced dataset, the massive pool of true negatives artificially inflates ROC-AUC and creates a misleadingly optimistic picture of model quality.

---

### Implication for Modeling

The 6.63% minority proportion establishes a requirement: the model must be explicitly told to pay more attention to defaulters, because the natural mathematics of training on this data will push it toward ignoring them.

Two techniques will be used:
- **Algorithmic class weighting** (`class_weight='balanced'` in sklearn, `scale_pos_weight` in XGBoost) — adjusts the loss function to penalize minority-class mistakes proportionally more
- **Synthetic oversampling (SMOTE)** — creates new synthetic minority-class samples, but only inside cross-validation folds to prevent data leakage

---

## Part 2: Correlation Matrix — The Strategic Signal Filter

### Why the Matrix Comes Before Any Plot

Generating a visualization for every feature in a dataset is a beginner mistake. Most charts will show nothing useful, waste time to produce, and clutter any report. The Spearman correlation matrix is computed first to identify which features deserve visual investigation and which don't.

### Why Spearman, Not Pearson

Standard Pearson correlation assumes linearly distributed data and is sensitive to outliers. Most features in this dataset are heavily right-skewed and contain extreme values even after outlier treatment. Spearman correlation measures monotonic relationships — it asks whether two variables move in the same direction — without assuming anything about the shape of their distributions. It is the correct choice for financial data of this type.

---

### Results: Feature Correlation with Target

| Feature | Spearman Correlation | Signal Classification |
| --- | --- | --- |
| NumberOfTimes90DaysLate | +0.34 | Strong |
| NumberOfTime30-59DaysPastDueNotWorse | +0.30 | Strong |
| NumberOfTime60-89DaysPastDueNotWorse | +0.28 | Strong |
| RevolvingUtilizationOfUnsecuredLines | +0.24 | Moderate-Strong |
| age | −0.12 | Moderate (negative direction) |
| NumberOfDependents | +0.06 | Weak |
| NumberOfOpenCreditLinesAndLoans | −0.03 | Negligible |
| MonthlyIncome | −0.04 | Negligible |
| DebtRatio | +0.02 | Negligible |

---

### The SHAP Answer Key

This correlation table is not just an EDA finding — it is a formal prediction that will be verified in Stage 4.

When SHAP analysis is run on the final model, the global feature importance ranking must match the signal hierarchy established here. Specifically:

| EDA Prediction | Expected SHAP Rank |
| --- | --- |
| NumberOfTimes90DaysLate — strongest signal | Must appear at or near rank 1 |
| RevolvingUtilizationOfUnsecuredLines — fourth strongest | Must appear in top 5 |
| MonthlyIncome — near-zero correlation | Must rank in the bottom half |
| NumberOfDependents — near-zero correlation | Must rank in the bottom half |

**If the SHAP output contradicts these predictions**, it means the model has latched onto accidental correlations in the training data rather than genuine financial behavior. This is called hallucination — the model appears to work but has learned the wrong things. The EDA findings act as the audit mechanism that detects it.

---

### Pre-Planned Defense Against Overfitting

Identifying weak-signal features now creates a pre-built response to overfitting during modeling. If the final model performs well on training data but poorly on new data, the diagnosis is already prepared: the model has memorized noise from the near-zero correlation features.

The corrective action is also pre-planned:

| Feature | Action if Overfitting Occurs |
| --- | --- |
| DebtRatio | Drop first — lowest correlation |
| NumberOfDependents | Drop — near-zero signal, median-imputed |
| NumberOfOpenCreditLinesAndLoans | Drop — negative correlation suggests no real relationship |
| MonthlyIncome | Evaluate carefully — log-transformed; may contribute via interactions |

This response is ready to execute without any additional exploratory work.

---

## Part 3: Multicollinearity Assessment — The Echo Chamber Check

### What Multicollinearity Is and Why It Damages SHAP

Multicollinearity occurs when two features encode the same information. When a model is trained on redundant features, it cannot cleanly attribute predictive credit to either one. The importance gets split or misattributed between the correlated pair, making both features appear weaker than they actually are.

For a project where SHAP interpretability is a core deliverable, multicollinearity is not just a modeling inconvenience — it actively corrupts the explanations.

**The Courtroom Analogy:** Two witnesses shouting the exact same testimony at the same time confuse the judge rather than strengthen the case. In machine learning, correlated features create the same confusion, diluting the clear signal that should be attributed to the genuinely important variable.

---

### The Primary Risk: Are the Three Late Payment Columns Redundant?

The most intuitive concern is the three late payment features. It is reasonable to assume that a borrower who reaches 90 days late must first have been 30 and 60 days late, creating a near-perfect staircase correlation between the three columns.

**If this were true:** Keeping all three would be like giving the model the same information three times, drowning out other signals and producing misleading SHAP values.

### Feature-to-Feature Correlation Results

| Feature Pair | Spearman Correlation | Overlap |
| --- | --- | --- |
| 30-59 days vs 60-89 days | ~0.27 | ~27% |
| 30-59 days vs 90+ days | ~0.25 | ~25% |
| 60-89 days vs 90+ days | ~0.30 | ~30% |

The correlations are positive but moderate. Each feature contains approximately **70–75% unique information** that the others do not capture.

---

### What This Means Financially

The three features are not a staircase — they represent genuinely distinct borrower profiles:

| Borrower Profile | 30-59 Day Count | 60-89 Day Count | 90+ Day Count | Risk Interpretation |
| --- | --- | --- | --- | --- |
| Chronic Early Delinquent | High | Low | Low | Frequently slips but always recovers before escalation |
| Escalator | Low | Medium | High | Rarely late, but when late, goes all the way |
| Catastrophic | Low | Low | High | No warning signs, then sudden severe default |

These are three fundamentally different financial risk profiles. A model that sees all three features can distinguish between them. A model that only sees one aggregated count cannot.

> **The multicollinearity check passed. All three late payment features are retained without creating an echo chamber. SHAP values will not be distorted by their combined presence.**

---

## Part 4: Bivariate Analysis — Late Payments vs. Default

### Method

Bar charts were used to compare the average count of late payment events for defaulters vs. non-defaulters across all three delinquency windows. This directly visualizes the correlation found in the matrix.

### Results: Average Late Payment Counts by Class

| Feature | Non-Defaulters (Class 0) | Defaulters (Class 1) | Ratio |
| --- | --- | --- | --- |
| NumberOfTime30-59DaysPastDueNotWorse | ~0.31 | ~1.05 | ~3.4× |
| NumberOfTime60-89DaysPastDueNotWorse | ~0.17 | ~0.75 | ~4.4× |
| NumberOfTimes90DaysLate | ~0.21 | ~1.32 | ~6.3× |

### The Escalating Ratio Pattern

The most important finding is not the absolute values — it is the multiplier. Moving from the 30-day window to the 90-day window, the gap between defaulters and non-defaulters **widens dramatically**:

- At 30-59 days: defaulters have 3.4× more incidents
- At 60-89 days: defaulters have 4.4× more incidents
- At 90+ days: defaulters have 6.3× more incidents

This is not a gradual linear relationship. The 90-day threshold is the point of no return.

> **A borrower who crosses into 90-day delinquency is in a categorically different financial situation than one who has only experienced 30-day delays. The model must treat these as distinct stages of distress, not as interchangeable counts of "being late."**

---

### The First Incident Effect

The bar charts also reveal that even **a single 30-day late event** more than triples the average count for defaulters. The distance between zero incidents and one incident is the largest single jump in the entire distribution. This "first incident effect" will be confirmed and quantified in Stage 4 through the SHAP dependence plots.

---

## Part 5: Univariate Analysis — Distribution Overlap (KDE Plots)

### Method

Kernel Density Estimation (KDE) plots were used for continuous features — age, monthly income, and debt ratio. Each plot shows the full probability distribution of the feature, overlaid separately for defaulters (red) and non-defaulters (green).

**Critical technical note:** `common_norm=False` was used. Without this parameter, the defaulter distribution curve (representing only 6.63% of borrowers) would produce a nearly invisible line compared to the towering non-defaulter curve. `common_norm=False` normalizes each class independently so that both distributions have the same area under the curve, making their shapes comparable regardless of the class size difference.

### The Golden Rule of Density Plot Interpretation

| Visual Pattern | Statistical Meaning | Model Implication |
| --- | --- | --- |
| Two curves separated with minimal overlap | High correlation with target | Model will find clean split points |
| Two curves sitting on top of each other | Near-zero correlation | Feature adds minimal value on its own |

---

### Finding 1 — Age: Clear Demographic Separation

| Class | Peak Age Range |
| --- | --- |
| Defaulters (red) | 30–40 years |
| Non-Defaulters (green) | 50–60 years |

The two curves are visibly offset from each other. Younger borrowers default at significantly higher rates than older borrowers. This aligns with known financial behavior: younger borrowers have shorter credit histories, greater income volatility, and higher probability of life disruptions (job loss, relocation, family formation).

The model should find age to be a useful secondary splitting feature — not the primary signal, but a meaningful modifier that adjusts risk estimates based on demographic context.

---

### Finding 2 — MonthlyIncome: Near-Perfect Overlap

The KDE curves for `MonthlyIncome` are nearly identical for both classes. The density shapes, peak locations, and spread of the two distributions are almost indistinguishable.

**This is the visual proof of the near-zero Spearman correlation (-0.04).** High income borrowers default. Low income borrowers don't. Income level, on its own, does not determine default risk.

> A bank that makes lending decisions primarily based on income is making an analytically unsupported choice. This dataset provides mathematical evidence that income alone is an unreliable default signal.

**Important nuance:** MonthlyIncome may still contribute predictive value *in combination with other features* — for example, income relative to debt (which is what DebtRatio captures, and which can be engineered into a more informative form). But as a standalone predictor, it contributes almost nothing.

---

### Finding 3 — DebtRatio: Overlap Despite Intuitive Appeal

Similar to MonthlyIncome, the `DebtRatio` distributions for both classes overlap heavily. High debt ratios are shared across both defaulters and non-defaulters.

This is counterintuitive at first — one would expect people with crushing debt obligations relative to income to default more frequently. But the dataset reveals that many borrowers with technically high debt ratios continue to meet their obligations, while borrowers with moderate debt ratios sometimes fail catastrophically.

The explanation is behavioral: it is not *how much* debt you carry, it is *what you do when financial stress hits*. The payment behavior features (late counts) capture the behavioral response. The debt ratio only captures the structural setup.

> **Neither income nor debt ratio is a reliable standalone predictor of default. Behavioral features — how borrowers actually respond when under financial pressure — are the real signal.**

---

### Why Certain Features Were Not Plotted

`NumberRealEstateLoansOrLines` and `NumberOfOpenCreditLinesAndLoans` were not visualized.

This is not an oversight. These features showed Spearman correlations of -0.03 and essentially zero respectively. Plotting them would produce two almost perfectly overlapping density curves — a visual confirmation of what the math already proved. No new information would be gained. The decision not to plot them reflects the original principle: calculate everything, plot only what matters.

---

## Part 6: Credit Utilization Deep Dive

### Why a Dedicated Analysis

`RevolvingUtilizationOfUnsecuredLines` had a Spearman correlation of +0.24 with the target — the fourth-strongest signal in the dataset, but the strongest among non-behavioral features. Unlike the late payment counts, which directly record financial failures, credit utilization captures financial *posture* — how close a borrower operates to their credit limits in normal times.

This distinction makes it worth a dedicated analysis beyond the standard bivariate comparison.

---

### Statistical Grouping Results

| Class | Average Utilization |
| --- | --- |
| Non-Defaulters (Class 0) | 29.3% |
| Defaulters (Class 1) | 68.5% |

The gap is 39.2 percentage points — more than a **2.3× difference** in average credit stress level.

This is not a subtle distinction. Non-defaulters, on average, use roughly a third of their available revolving credit. Defaulters use more than two-thirds, operating in a state of chronic near-maxed-out credit usage.

---

### KDE Plot Findings

The density curves for credit utilization show substantial visual separation:

| Observation | Meaning |
| --- | --- |
| Non-defaulters cluster heavily near 0–30% utilization | Most safe borrowers maintain significant credit buffer |
| Defaulters show a peak approaching 100% utilization | Most defaulters operate with almost no remaining credit capacity |
| Both distributions have long right tails | Extreme utilization values exist in both classes, but disproportionately in defaulters |

---

### The Continuous Risk Relationship

Unlike the late payment features (which showed a discrete "cliff" at the 90-day threshold), credit utilization shows a **continuous and approximately linear relationship** with risk. Every additional percentage of credit used adds a measurable amount of default risk — there is no single safe threshold.

| Conventional Wisdom | What the Data Shows |
| --- | --- |
| "Stay below 30% utilization" | Risk increases continuously from 0% — the 30% rule is a social heuristic, not a mathematical threshold |
| "Once you're over 50%, you're high risk" | True directionally, but risk was already elevated long before 50% |

> **This finding will be confirmed in the SHAP dependence plots in Stage 4, which will show the shape of this continuous risk relationship across the full utilization range.**

---

### Interaction with Late Payment History

A crucial nuance emerged from the credit utilization analysis: high utilization on its own is not catastrophic. The truly high-risk profile is **high utilization combined with late payment history**.

A borrower with 80% utilization and clean payment history is meaningfully different from a borrower with 80% utilization and two 90-day late events. The first has stretched their credit thin but is managing it. The second has both the structural vulnerability and the behavioral record of failure.

This insight directly motivated the creation of the `Struggle_Index` engineered feature (Utilization × Total_Late_Events) in Stage 5, which captures this compounded risk profile in a single variable — and which SHAP ultimately confirmed as the strongest single predictor in the final model.

---

## Part 7: Validating the MonthlyIncome_was_missing Flag

### The Problem Being Solved

In Stage 1, a binary indicator called `MonthlyIncome_was_missing` was created before imputation, taking value 1 if income was absent and 0 if it was reported. The reasoning was that the fact of a value being missing might itself carry predictive signal.

But creating a feature does not justify keeping it. If borrowers with missing income default at the same rate as those who reported income, the flag is pure noise and should be dropped. Feeding noise into a tree model increases overfitting risk and dilutes the signal from genuinely useful features.

**This must be tested, not assumed.**

---

### Validation Method

The dataset was grouped by the `MonthlyIncome_was_missing` flag, and the default rate for each group was computed directly.

### Results

| MonthlyIncome_was_missing | Default Rate | Count |
| --- | --- | --- |
| 0 (income reported) | 6.89% | ~116,000 borrowers |
| 1 (income missing) | 5.50% | ~34,000 borrowers |

The difference is **1.39 percentage points** — borrowers who left their income blank defaulted at a statistically lower rate than those who reported it.

---

### Why This Signal Is Real

The pattern makes financial sense on reflection. Borrowers with missing income data in a loan application context may disproportionately include:
- Self-employed individuals who have irregular but stable income
- Borrowers who submitted the application with incomplete documentation but subsequently provided income verification verbally
- Higher-net-worth individuals who chose not to disclose income on forms

In any case, the measurable divergence in default rates proves that the missingness is not random — it carries information about the borrower's risk profile.

> **Validation passed. `MonthlyIncome_was_missing` provides genuine predictive signal (1.39% default rate divergence) and is retained as a feature for all subsequent modeling stages.**

---

## EDA Summary — What We Now Know

### The Signal Hierarchy (EDA Answer Key)

| Rank | Feature Category | Strength | Notes |
| --- | --- | --- | --- |
| 1 | 90-Day Late Payment Count | Strong | Strongest single predictor; point of no return |
| 2 | 30-Day and 60-Day Late Counts | Strong | Distinct stages, not redundant with 90-day |
| 3 | Credit Utilization | Moderate-Strong | Continuous relationship; compound effect with late counts |
| 4 | Age | Moderate | Clear demographic separation; younger = higher risk |
| 5 | MonthlyIncome_was_missing | Weak but valid | 1.39% divergence confirms genuine signal |
| 6 | MonthlyIncome, DebtRatio, Dependents, Open Lines | Weak/Negligible | Retained but expected to contribute minimally |

---

### Decisions Made in EDA That Drive Every Subsequent Stage

| EDA Finding | Downstream Decision |
| --- | --- |
| 14:1 class imbalance | PR-AUC as primary metric; class weighting in all models |
| Late payment features have only 25–30% overlap | All three retained; no consolidation needed |
| Income and debt ratio show near-zero target correlation | Pre-planned drop candidates if overfitting occurs |
| Credit utilization shows continuous risk increase | Motivates `Utilization_Squared` and `Struggle_Index` engineered features |
| Age shows clear demographic separation | Retained as secondary splitting feature |
| MonthlyIncome_was_missing divergence confirmed | Feature retained for all model stages |
| 90-day late is the dominant signal | SHAP audit will verify model agrees with this |

---

### The Anti-Hallucination Audit Setup

This EDA stage formally established what the model *should* learn. When SHAP analysis is run after model training, the results will be compared against this table:

| Expected | If Violated |
| --- | --- |
| 90-day late count in global SHAP top 2 | Model is hallucinating — audit the feature engineering pipeline |
| Credit utilization in SHAP top 5 | Model missed a primary signal — check for leakage or preprocessing errors |
| MonthlyIncome in SHAP bottom half | Confirmed — if income ranks high, model has overfit to noise |
| NumberOfDependents near bottom | Confirmed — EDA showed no meaningful relationship with default |


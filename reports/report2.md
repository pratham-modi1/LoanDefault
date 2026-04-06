# Stage 2: Exploratory Data Analysis

---

## 1. Objective of EDA

EDA in this project serves a specific purpose: **to understand what fraud looks like in data before asking a model to detect it.** Rather than generating random visualizations, the approach here is structured and investigative — starting with the most interpretable features and progressively moving toward data-driven feature selection.

The guiding question throughout: *What separates a fraudulent transaction from a legitimate one?*

---

## 2. Starting Point — Interpretable Features First

EDA began with `Amount` and `Hour` because these are the only features that can be understood directly without domain knowledge of PCA mathematics.

**Amount Analysis:**
Comparing distributions between fraud and legitimate classes revealed significant overlap. Fraud transactions showed a wider spread at lower amounts, with fewer high-value outliers compared to legitimate transactions.

> **Conclusion:** Amount is a weak individual predictor. Fraud does not consistently correspond to high or low amounts.

**Hour Analysis:**
Fraud transactions showed marginally higher concentration in early morning hours compared to legitimate ones, which peak during business hours. However, the overlap remained large.

> **Conclusion:** Hour provides a weak but non-zero signal. Worth retaining but cannot independently distinguish fraud.

---

## 3. Key Realization from Interpretable Features

After analyzing the two most understandable features and finding only weak separation, a critical insight emerged:

> **Fraud detection is not based on obvious patterns. No single feature cleanly separates fraud from legitimate transactions. The model must learn from the combination of many weak signals simultaneously.**

This finding directly justifies the use of machine learning over rule-based approaches. A human analyst cannot write a rule like *"flag transactions above X amount at Y hour"* — the patterns are too subtle and interdependent.

---

## 4. Moving to PCA Features — A Data-Driven Approach

With 28 PCA features available and no domain knowledge to prioritize them, random exploration would be inefficient. A **correlation-based approach** was used instead.

**Method:**
1. Compute Pearson correlation between each feature and the target variable (`Class`)
2. Sort by absolute correlation value
3. Select top positive and top negative correlations for analysis

**Interpretation:**
- Positive correlation → higher value → higher fraud probability
- Negative correlation → higher value → lower fraud probability

| Correlation Strength | Threshold |
| --- | --- |
| Strong | 0.30+ |
| Moderate | 0.10 – 0.30 |
| Weak but useful | 0.01 – 0.10 |

> Note: Even small values matter in the context of severe class imbalance.

**Top positive correlations with fraud:** V11, V4, V2

**Top negative correlations with fraud:** V14, V17, V12

This ranking transforms EDA from random exploration into a targeted investigation.

---

## 5. Distribution Analysis of Key Features

For each top-ranked feature, boxplots comparing fraud and legitimate distributions were examined.

**V11 (strong positive correlation):**
The median value for fraud transactions is substantially higher than for legitimate ones. The interquartile ranges show limited overlap. When V11 is elevated, fraud probability increases significantly.

**V4 (moderate positive correlation):**
Fraud transactions cluster at higher V4 values. The separation is clear enough to be a meaningful predictor.

**V14 (strong negative correlation):**
Fraud transactions cluster at significantly lower V14 values. When V14 is deeply negative, the probability of fraud rises substantially.

**V17 (strong negative correlation):**
Fraud transactions show a pronounced downward shift. This feature provides strong discriminative power in the negative direction.

---

## 6. Pattern Recognition — What EDA Reveals

After systematic analysis, a consistent pattern emerged:

> Fraud transactions do not have unusually high or low values on all features. Rather, **specific features shift predictably in specific directions** when fraud is present — some increase (V11, V4) while others decrease (V14, V17).

This multi-directional pattern confirms that fraud detection requires a model capable of weighing multiple features simultaneously, not a single threshold on any one variable.

---

## 7. Why Not All 28 Features Were Analyzed

Analysis stopped after the top correlated features because:

1. The model will use all 28 features regardless of EDA findings — EDA is about building intuition, not manually selecting features
2. Since V1–V28 are PCA components, they are already uncorrelated by construction — interaction analysis adds no interpretable value
3. After analyzing six to eight features, the same patterns repeated — insight saturation was reached

---

## 8. EDA Summary and Transition to Modeling

| Feature Group | Signal Strength | Notes |
| --- | --- | --- |
| Amount | Weak | Overlapping distributions |
| Hour | Weak | Slight early-morning fraud spike |
| V11, V4 | Strong positive | Clear upward shift during fraud |
| V14, V17 | Strong negative | Clear downward shift during fraud |

**Key takeaway:** No single feature provides clean separation. Fraud is detectable only through combinations of features. The logical next step is not to create manual rules based on EDA findings, but to train a model that converts these observed patterns into a mathematical decision function.

> *EDA identifies what matters. The model learns how it matters.*
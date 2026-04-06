# Stage 0: Project Setup & Data Understanding

---

## 1. Repository Setup

- Created a GitHub repository for the Loan Default Predictor project
- Initialized Git in local project folder
- Added `.gitignore` file to exclude CSV dataset files
- Committed initial project files
- Connected local repo to GitHub and pushed code

---

## 2. Dataset Loading

- Imported dataset (`CS-training.csv`) using pandas
- Removed unnecessary index column
- Reorganized columns into meaningful categories:
  - Credit Usage
  - Payment Behavior
  - Financial Capacity
  - Personal Profile
  - Target Variable

---

## 3. Data Understanding

- Interpreted each feature in the dataset
- Understood meaning of key financial terms:
  - Credit limit and utilization
  - Loan vs credit lines
  - Delinquency (late payments)

---

## 4. Target Variable Clarity

- Identified target: `SeriousDlqin2yrs`
- Defined as whether a person had 90+ days payment delay in last 2 years
- Distinguished from other delay count features

---

**Status:** Dataset structure understood · Features interpreted · Project environment and repository successfully set up

---
---

# Stage 1: Data Preprocessing

---

## Part 1: Duplicate & Missing Value Handling

### 1. Duplicate Handling

- Checked for duplicate rows in the dataset
- Identified number of duplicates using `df.duplicated().sum()`
- Removed duplicates using `df.drop_duplicates()`
- **Reason:** Duplicate data can bias the model and distort learning patterns

---

### 2. Missing Value Analysis

- Checked missing values column-wise using `df.isna().sum()`
- Observations:
  - `MonthlyIncome` had ~20% missing values
  - `NumberOfDependents` had ~2–5% missing values
  - Other columns had negligible or no missing values

---

### 3. Handling Missing Values: MonthlyIncome

**Problem:** MonthlyIncome is a continuous variable with a wide range and high variance.

**Approach Used:** Model-based imputation using `RandomForestRegressor`

**Steps:**
1. Added a binary indicator feature `MonthlyIncome_was_missing` to preserve missingness information before imputation
2. Split the dataset into two subsets: known (non-missing income) and unknown (missing income)
3. Trained a Random Forest model on the known subset using other features as predictors
4. Predicted missing income values for the unknown subset using the trained model
5. Filled the predicted values back into the dataset to replace missing entries in `MonthlyIncome`

| Approach | Why Not? |
| --- | --- |
| Median | Basic approximation; does not capture relationships between features; ignores patterns like income vs credit behavior |
| Random Forest | Learns complex relationships, handles non-linear dependencies, produces realistic estimates, improves feature quality |

---

### 4. Handling Missing Values: NumberOfDependents

**Problem:** Small integer feature with limited range.

**Approach Used:** Median imputation

**Reasoning:**
- Low variability → median is representative
- Relationships with other features are weak
- Model-based approach may introduce noise (e.g., predicting 1.7 dependents)
- Simpler and more stable method preferred

---

### 5. Key Insight

> Not all missing values should be treated the same way. Complex continuous features benefit from model-based imputation, while simple discrete features are better handled with statistical methods like median.

---

**Status:** Duplicates removed · Missing values handled intelligently · Dataset cleaned and ready for further preprocessing steps

---

## Part 2: Outlier Handling

### 1. Objective

After handling duplicates and missing values, the next step was to handle outliers. The goal was to:
- Remove logically incorrect data
- Identify suspicious extreme values
- Reduce skewness without losing important information
- Prepare dataset for stable model training

---

### 2. Removal of Logically Incorrect Values

Applied domain-based filtering to remove impossible values:
- `age < 18` or `age > 100` → removed
- Negative values in financial/count features → removed
- Counts (like delays, loans, dependents) cannot be negative

**Result:** Around 3,800 rows removed (~2.5% of dataset). Data integrity improved without major data loss.

---

### 3. Handling Encoded Outliers in Late Payment Features

**Columns:**
- `NumberOfTime30-59DaysPastDueNotWorse`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfTimes90DaysLate`

**Observed values:** Normal range 0 to ~15–18; sudden spikes at 96 and 98.

**Interpretation:** These values are not realistic counts — they represent encoded errors or placeholder values.

**Action:** Removed rows where any of these values ≥ 90.

**Reason:** These are invalid data points, not real behavior. Keeping them would mislead the model.

---

### 4. Detection of Statistical Outliers

After removing logically incorrect values and encoded anomalies, statistical outliers were detected in continuous features:
- `DebtRatio`
- `RevolvingUtilizationOfUnsecuredLines`
- `MonthlyIncome`

**Methods used:**
- `df.describe()` → understand overall spread and range
- `df.quantile()` → analyze percentile distribution (90%, 95%, 99%, 99.9%)
- Sorting top values → inspect extreme cases

**Key observation:** Large gap between 99th percentile and maximum values, indicating the presence of extreme outliers.

---

### 5. Handling Extreme but Valid Values

An important distinction was made:
- **Invalid values** → already removed
- **Valid but extreme values** → handled using transformation techniques

---

#### 5.1 RevolvingUtilizationOfUnsecuredLines

| Property | Detail |
| --- | --- |
| Normal range | 0 to 1 |
| 99th percentile | ≈ 1.09 |
| Extreme values | Up to 50,000+ |
| Action | Capping at 99th percentile |
| Method | `df.clip(upper=df.quantile(0.99))` |
| Reason | Extreme values are unrealistic and distort model behavior |

---

#### 5.2 DebtRatio

| Property | Detail |
| --- | --- |
| Median | ≈ 0.35 |
| 99th percentile | ≈ 4,900 |
| Maximum | Above 300,000 |
| Action | Capping at 99th percentile |
| Method | `df.clip()` after `df.quantile(0.99)` |
| Reason | Very large values are likely anomalies; capping prevents domination of the model |

---

#### 5.3 MonthlyIncome

| Property | Detail |
| --- | --- |
| Distribution | Highly right-skewed |
| 99.9th percentile | ≈ 75,000 |
| Action | Capping at 99.9th percentile + log transformation |
| Method | `df.clip()` then `np.log1p()` |
| Reason | Log transformation compresses scale, improves distribution, and reduces tail influence |

---

### 6. Handling of Remaining Features

| Feature | Action | Reason |
| --- | --- | --- |
| Age | No transformation | Already cleaned (18–100); naturally bounded |
| NumberOfOpenCreditLinesAndLoans | No transformation | High values are rare but possible |
| NumberRealEstateLoansOrLines | No transformation | Extreme values very rare; no strong evidence to modify |
| NumberOfDependents | No transformation | Small values mostly (0–4); higher values are real-world valid |
| SeriousDlqin2yrs (target) | Not modified | Avoid leakage or distortion |

---

### 7. Key Techniques Used

| Technique | Purpose |
| --- | --- |
| Logical filtering | Removed impossible values |
| Pattern detection | Removed encoded anomalies (96, 98) |
| Percentile analysis | Identified extreme values |
| Capping (Winsorization) | Controlled extreme values |
| Log transformation | Handled skewed distributions |

---

**Status:** Outlier handling completed using domain knowledge and statistical methods · Dataset ready for next step
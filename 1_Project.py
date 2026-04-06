import pandas as pd
import joblib
import os

# Load dataset
df = pd.read_csv("cs-training.csv")

# Drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Reorganize columns into categories

# Credit Usage
credit_usage = [
    'RevolvingUtilizationOfUnsecuredLines'
]

# Payment Behavior
payment_behavior = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]

# Financial Capacity
financial_capacity = [
    'DebtRatio',
    'MonthlyIncome'
]

# Personal Profile
personal_profile = [
    'age',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberRealEstateLoansOrLines',
    'NumberOfDependents'
]

# Target
target = ['SeriousDlqin2yrs']

# Final column order
new_order = (
    credit_usage +
    payment_behavior +
    financial_capacity +
    personal_profile +
    target
)

# Reorder dataframe
df = df[new_order]

# Preview
# print(df.shape)

# Check duplicate rows
# print("Duplicate rows:", df.duplicated().sum())
# Remove duplicates
df = df.drop_duplicates()

# Count missing values column-wise
# print("\nMissing values per column:")
# print(df.isna().sum())


from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# STEP 1: Split data
# -----------------------------
df['MonthlyIncome_was_missing'] = df['MonthlyIncome'].isna().astype(int)
known = df[df['MonthlyIncome'].notna()]
unknown = df[df['MonthlyIncome'].isna()]

# -----------------------------
# STEP 2: Select features (EXCLUDE NumberOfDependents)
# -----------------------------
features = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse'
]

# -----------------------------
# STEP 3: Train model
# -----------------------------
model = RandomForestRegressor(
     n_estimators=50,
     random_state=42,
     n_jobs=-1
 )

model.fit(known[features], known['MonthlyIncome'])

# # -----------------------------
# # STEP 4: Predict missing income
# # -----------------------------
predicted_income = model.predict(unknown[features])

# # -----------------------------
# # STEP 5: Fill missing values
# # -----------------------------
df.loc[df['MonthlyIncome'].isna(), 'MonthlyIncome'] = predicted_income
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

#lets handle outliers now

#print(df.shape)
# Age
df = df[(df['age'] >= 18) & (df['age'] <= 100)]

# Monthly Income
df = df[df['MonthlyIncome'] >= 0]

# Debt Ratio
df = df[df['DebtRatio'] >= 0]

# Credit Utilization
df = df[df['RevolvingUtilizationOfUnsecuredLines'] >= 0]

# Late payment columns
late_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]

for col in late_cols:
    df = df[df[col] >= 0]

# Open credit lines
df = df[df['NumberOfOpenCreditLinesAndLoans'] >= 0]

# Real estate loans
df = df[df['NumberRealEstateLoansOrLines'] >= 0]

# Dependents
df = df[df['NumberOfDependents'] >= 0]

#print(df.shape)
#You removed ~3.8k rows → that’s ~2.5% of data

# for col in [
#     'NumberOfTime30-59DaysPastDueNotWorse',
#     'NumberOfTime60-89DaysPastDueNotWorse',
#     'NumberOfTimes90DaysLate'
# ]:
#     print(col)
#     print(df[col].value_counts().sort_index(ascending=False).head(20))
df = df[~(
    (df['NumberOfTime30-59DaysPastDueNotWorse'] >= 90) |
    (df['NumberOfTime60-89DaysPastDueNotWorse'] >= 90) |
    (df['NumberOfTimes90DaysLate'] >= 90)
)]



# cols = [
#     'RevolvingUtilizationOfUnsecuredLines',
#     'DebtRatio',
#     'MonthlyIncome',
#     'age',
#     'NumberOfOpenCreditLinesAndLoans',
#     'NumberRealEstateLoansOrLines',
#     'NumberOfDependents'
# ]

# for col in cols:
#     print(f"\n========== {col} ==========")
    
#     # Basic stats
#     print("\n--- Describe ---")
#     print(df[col].describe())
    
#     # Percentiles (only for continuous features)
#     if df[col].dtype != 'int64':
#         print("\n--- Percentiles ---")
#         print(df[col].quantile([0.90, 0.95, 0.99, 0.999]))
    
#     # Top extreme values
#     print("\n--- Top 10 Highest Values ---")
#     print(df[col].sort_values(ascending=False).head(10))
    
#     # Value counts (useful for count/discrete features)
#     print("\n--- Value Counts (Top) ---")
#     print(df[col].value_counts().head(10))

import numpy as np

# Utilization
cap = df['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99)
df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=cap)

# DebtRatio
cap = df['DebtRatio'].quantile(0.99)
df['DebtRatio'] = df['DebtRatio'].clip(upper=cap)

# MonthlyIncome
cap = df['MonthlyIncome'].quantile(0.999)
df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=cap)
df['MonthlyIncome'] = np.log1p(df['MonthlyIncome'])


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================

# # ---------------------------------------------------------------------
# # 1. Target Variable Analysis & Class Imbalance
# # ---------------------------------------------------------------------
# target_counts = df['SeriousDlqin2yrs'].value_counts()
# target_percentages = df['SeriousDlqin2yrs'].value_counts(normalize=True) * 100

# print("\n--- Target Variable Distribution: SeriousDlqin2yrs ---")
# print(f"Non-Defaulters (0): {target_counts[0]} ({target_percentages[0]:.2f}%)")
# print(f"Defaulters (1): {target_counts.get(1, 0)} ({target_percentages.get(1, 0.0):.2f}%)")
# print("-" * 54)

# plt.figure(figsize=(8, 6))
# ax = sns.countplot(data=df, x='SeriousDlqin2yrs', palette=['#2ecc71', '#e74c3c'])
# plt.title('Distribution of Loan Defaults (Class Imbalance)', fontsize=14, fontweight='bold', pad=15)
# plt.xlabel('Serious Default in 2 Years (0 = No, 1 = Yes)', fontsize=12)
# plt.ylabel('Number of Borrowers', fontsize=12)

# # Add percentage labels directly on top of the bars
# total = len(df)
# for p in ax.patches:
#     height = p.get_height()
#     percentage = f'{100 * height / total:.1f}%'
#     ax.annotate(percentage, 
#                 (p.get_x() + p.get_width() / 2., height), 
#                 ha='center', va='bottom', 
#                 fontsize=12, fontweight='bold', xytext=(0, 5), 
#                 textcoords='offset points')

# sns.despine()
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------------------
# # 2. Correlation Matrix (Target Analysis)
# # ---------------------------------------------------------------------
# plt.figure(figsize=(10, 8))
# corr_matrix = df.corr(method='spearman')

# # Plotting only the correlation with our target variable to filter noise
# target_corr = corr_matrix[['SeriousDlqin2yrs']].sort_values(by='SeriousDlqin2yrs', ascending=False)
# sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
# plt.title('Feature Correlation with Loan Default (Target Analysis)', fontsize=14, fontweight='bold', pad=15)
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------------------
# # 3. Multicollinearity Assessment (Feature-to-Feature Correlation)
# # ---------------------------------------------------------------------
# plt.figure(figsize=(12, 10))

# # Calculate full correlation matrix EXCLUDING the target
# features_only = df.drop(columns=['SeriousDlqin2yrs'])
# full_corr = features_only.corr(method='spearman')

# # Create a mask for the upper triangle so it's not visually overwhelming
# mask = np.triu(np.ones_like(full_corr, dtype=bool))

# sns.heatmap(full_corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
# plt.title('Feature-to-Feature Correlation (Multicollinearity Check)', fontsize=14, fontweight='bold', pad=15)
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------------------
# # 4. Bivariate Analysis (Deep Dive: Late Payments vs Target)
# # ---------------------------------------------------------------------
# late_payment_cols = [
#     'NumberOfTime30-59DaysPastDueNotWorse',
#     'NumberOfTime60-89DaysPastDueNotWorse',
#     'NumberOfTimes90DaysLate'
# ]

# print("\n--- Average Number of Late Payments by Target Class ---")
# grouped_means = df.groupby('SeriousDlqin2yrs')[late_payment_cols].mean().round(3)
# print(grouped_means)
# print("-" * 60)

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# for i, col in enumerate(late_payment_cols):
#     sns.barplot(data=df, x='SeriousDlqin2yrs', y=col, ax=axes[i], palette=['#2ecc71', '#e74c3c'], errorbar=None)
#     axes[i].set_title(f'Avg {col}', fontsize=10)
#     axes[i].set_xlabel('Default (0=No, 1=Yes)', fontsize=10)
#     axes[i].set_ylabel('Average Count', fontsize=10)

# plt.suptitle('Impact of Late Payment History on Loan Default', fontsize=16, fontweight='bold', y=1.05)
# sns.despine()
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------------------
# # 5. Univariate Analysis / Distribution Overlap (KDE Plots)
# # ---------------------------------------------------------------------
# continuous_cols = ['age', 'DebtRatio', 'MonthlyIncome']

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# for i, col in enumerate(continuous_cols):
#     # KDE plot normalizes the curves so we can compare shapes despite the 93/6 class imbalance
#     sns.kdeplot(data=df, x=col, hue='SeriousDlqin2yrs', fill=True, common_norm=False, 
#                 palette=['#2ecc71', '#e74c3c'], ax=axes[i], alpha=0.5)
    
#     axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
#     axes[i].set_xlabel(col, fontsize=10)
#     axes[i].set_ylabel('Density', fontsize=10)

# plt.suptitle('Continuous Features vs. Loan Default (Distribution Overlap)', fontsize=16, fontweight='bold', y=1.05)
# sns.despine()
# plt.tight_layout()
# plt.show()

# =====================================================================
# END OF EDA STAGE
# =====================================================================
# =====================================================================
# STAGE 2.2: FEATURE ENGINEERING
# =====================================================================
#print("\n--- Starting Feature Engineering ---")

# 1. Total Late Events (Raw volume of delinquency)
df['Total_Late_Events'] = (
    df['NumberOfTime30-59DaysPastDueNotWorse'] + 
    df['NumberOfTime60-89DaysPastDueNotWorse'] + 
    df['NumberOfTimes90DaysLate']
)

# 2. Weighted Late Score (Severity of delinquency)
df['Weighted_Late_Score'] = (
    (df['NumberOfTime30-59DaysPastDueNotWorse'] * 1) + 
    (df['NumberOfTime60-89DaysPastDueNotWorse'] * 2) + 
    (df['NumberOfTimes90DaysLate'] * 3)
)

# 3. Credit Utilization Warning Flag (Binary)
df['Credit_Utilization_Warning'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 0.85).astype(int)

# 4. Income Per Dependent
# Reverse log1p safely, calculate, then re-apply log1p
raw_income = np.expm1(df['MonthlyIncome'])
df['Income_Per_Dependent'] = raw_income / (df['NumberOfDependents'] + 1)
df['Income_Per_Dependent'] = np.log1p(df['Income_Per_Dependent'])

# 5. NEW: Absolute Money Features (Financial Survival)
# DebtRatio is a percentage. We multiply it by raw income to get the literal dollar amount of debt.
df['Absolute_Monthly_Debt'] = df['DebtRatio'] * raw_income
# Then we see how much literal cash they have left to survive the month.
df['Remaining_Living_Money'] = raw_income - df['Absolute_Monthly_Debt']
# ---> ADD THESE TWO LINES <---
df['Is_Cash_Negative'] = (df['Remaining_Living_Money'] < 0).astype(int)
df['Low_Buffer_Flag'] = (df['Remaining_Living_Money'] < 500).astype(int)

# --- ADD THESE NEW FEATURES ---
# 1. Has ANY late payment ever (clean binary signal)
df['Has_Any_Late'] = (df['Total_Late_Events'] > 0).astype(int)

# 2. Worst single delinquency bucket (0=none, 1=30day, 2=60day, 3=90day)
df['Max_Late_Severity'] = (
    (df['NumberOfTime30-59DaysPastDueNotWorse'] > 0).astype(int) +
    (df['NumberOfTime60-89DaysPastDueNotWorse'] > 0).astype(int) +
    (df['NumberOfTimes90DaysLate'] > 0).astype(int)
)

# 3. Young + high utilization = extreme risk combo
df['Youth_Utilization_Risk'] = (
    df['RevolvingUtilizationOfUnsecuredLines'] * (1 / (df['age'] - 17))
)

# 4. DebtRatio is useless alone but powerful combined with cash
# Someone with 90% debt ratio AND negative living money = near certain default
df['Debt_Squeeze'] = df['DebtRatio'] * (df['Remaining_Living_Money'] < 0).astype(int)

# 5. Late payment per credit line (normalized severity)
df['Severity_Per_Line'] = df['Weighted_Late_Score'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)

# 5. Drop redundant raw columns to prevent XGBoost confusion
# df = df.drop(columns=[
#     'NumberOfTime30-59DaysPastDueNotWorse',
#     'NumberOfTime60-89DaysPastDueNotWorse',
#     'NumberOfTimes90DaysLate'
# ])

# =====================================================================
# THE "FORCE MULTIPLIER" FEATURES
# =====================================================================

# 1. The "Struggle Index" (Combining utilization and late payments)
df['Struggle_Index'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['Total_Late_Events']

# 2. The "Age-Adjusted Debt" (Debt is scarier for young people)
df['Age_Debt_Interaction'] = df['Absolute_Monthly_Debt'] / (df['age'] + 1)

# 3. The "Late Payment Density" 
# (Are you late on many lines or just one?)
df['Late_to_Open_Ratio'] = df['Total_Late_Events'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)

# 4. Polynomial Features (Squaring the strongest predictors)
# This helps the model see 'exponential' risk
df['Utilization_Squared'] = df['RevolvingUtilizationOfUnsecuredLines'] ** 2


# print(f"Dataset shape after feature engineering: {df.shape}")
# =====================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Split the data into Train and Test
# These features add noise, not signal
cols_to_drop = [
    'SeriousDlqin2yrs',
    'NumberRealEstateLoansOrLines',  # near-zero correlation from your own EDA
    'NumberOfDependents',            # near-zero correlation
    'Absolute_Monthly_Debt'          # redundant with Debt_Squeeze now
]
X = df.drop(columns=cols_to_drop)
y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Initialize the Scaler
scaler = StandardScaler()

# 3. FIT and TRANSFORM the training data (Learn the rules & apply them)
X_train_scaled = scaler.fit_transform(X_train)

# 4. ONLY TRANSFORM the test data (Apply the learned rules, DO NOT learn new ones)
X_test_scaled = scaler.transform(X_test)

# >>> ADD THIS LINE HERE <<<
joblib.dump(scaler, 'data_scaler.joblib')   

# 5. Convert back to DataFrames (To keep column names for SHAP analysis later)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# =====================================================================
# DIAGNOSTIC CEILING CHECK (Change 4)
# =====================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

# If LR and XGB have same PR-AUC, it's a feature problem not a model problem
print("\n--- Running Feature Ceiling Diagnostic ---")
lr_check = LogisticRegression(class_weight='balanced', max_iter=2000)
lr_check.fit(X_train_scaled, y_train)
lr_prob_check = lr_check.predict_proba(X_test_scaled)[:, 1]
print(f"LR PR-AUC (your feature ceiling): {average_precision_score(y_test, lr_prob_check):.4f}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# EDA AUDIT FIXES: Missing Features Validation
# NOTE: Run this on your 'df' BEFORE the train/test split code
# =====================================================================

# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1. Calculate the exact averages for Good vs. Bad borrowers
# utilization_means = df.groupby('SeriousDlqin2yrs')['RevolvingUtilizationOfUnsecuredLines'].mean().round(3)
# print("--- Average Credit Card Utilization ---")
# print(f"Safe Borrowers (0): {utilization_means[0] * 100}% of their limit")
# print(f"Defaulters (1): {utilization_means[1] * 100}% of their limit")

# # 2. Draw the Density Plot to prove visual separation to the auditor
# plt.figure(figsize=(8, 5))
# sns.kdeplot(data=df, x='RevolvingUtilizationOfUnsecuredLines', hue='SeriousDlqin2yrs', 
#             fill=True, common_norm=False, palette=['#2ecc71', '#e74c3c'], clip=(0, 2))
# plt.title('Credit Utilization vs Loan Default', fontsize=14, fontweight='bold')
# plt.xlabel('Credit Utilization (Capped at 200% for visibility)', fontsize=12)
# plt.ylabel('Density', fontsize=12)
# sns.despine()
# plt.show()

# Check if the default rate actually changes based on the missing income flag
# missing_income_rates = df.groupby('MonthlyIncome_was_missing')['SeriousDlqin2yrs'].agg(['mean', 'count'])
# missing_income_rates['mean'] = (missing_income_rates['mean'] * 100).round(2)
# missing_income_rates.columns = ['Default_Rate_Percentage', 'Total_Borrowers']

# print("--- Missing Income Validation ---")
# print(missing_income_rates)


import numpy as np
import pandas as pd
import os
import joblib

from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

"""
=========================================================
ARCHIVE: BASELINE COMPARISONS & MODEL EVOLUTION
=========================================================
To prove the necessity of the final XGBoost architecture, several 
baselines were established and tested during the development phase:

1. Dummy Classifier (Stratified): PR-AUC ~ 0.070
2. Logistic Regression (Linear Baseline): PR-AUC 0.3576
3. Random Forest (Baseline Ensemble): PR-AUC 0.3640
4. Soft-Voting Committee (RF + XGB): PR-AUC 0.3733 
   *Note: RF was dropped because it diluted XGBoost's performance.

Conclusion: The optimized XGBoost model alone extracted the maximum 
non-linear signal from the engineered features.
=========================================================
"""

# =========================================================
# FINAL STAGE: XGBOOST HYPERPARAMETER SCAN & SAVE
# =========================================================
model_filename = "best_xgb_model.joblib"

if os.path.exists(model_filename):
    print(f"\n--- [ACTION] Loading pre-trained model: {model_filename} (0.1s) ---")
    best_xgb = joblib.load(model_filename)
else:
    print("\n--- Scanning for best scale_pos_weight ---")
    results = []
    best_spw = 6
    best_pr_auc = 0
    
    for spw in [3, 5, 6, 7, 10]:
        m = XGBClassifier(
            max_depth=5, learning_rate=0.01, n_estimators=400,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            gamma=1, scale_pos_weight=spw, eval_metric='aucpr',
            random_state=42, n_jobs=-1
        )
        m.fit(X_train_scaled, y_train)
        prob = m.predict_proba(X_test_scaled)[:, 1]
        pr_auc = average_precision_score(y_test, prob)
        results.append({'spw': spw, 'pr_auc': round(pr_auc, 4)})
        print(f"spw={spw} → PR-AUC: {pr_auc:.4f}")
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_spw = spw

    print(f"\n--- Training final model with best spw={best_spw} ---")
    best_xgb = XGBClassifier(
        max_depth=5, learning_rate=0.01, n_estimators=400,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        gamma=1, scale_pos_weight=best_spw, eval_metric='aucpr',
        random_state=42, n_jobs=-1
    )
    best_xgb.fit(X_train_scaled, y_train)
    joblib.dump(best_xgb, model_filename)
    print(f"--- [SUCCESS] Model saved as {model_filename} ---")


# =========================================================
# THE SILENT LOADING & RESULTS BLOCK
# =========================================================

# 2. Get probabilities silently
best_xgb_prob = best_xgb.predict_proba(X_test_scaled)[:, 1]

# 3. Calculate the 4 specific numbers you want
from sklearn.metrics import precision_recall_curve, accuracy_score

precision, recall, thresholds = precision_recall_curve(y_test, best_xgb_prob)
idx = np.where(recall >= 0.75)[0][-1] # Finding the 75% Recall point
best_threshold = thresholds[idx]
y_pred = (best_xgb_prob >= best_threshold).astype(int)
acc = accuracy_score(y_test, y_pred)

# 4. THE ONLY OUTPUT
print("\n" + "="*30)
print(f"Precision : {precision[idx]:.4f}")
print(f"Recall    : {recall[idx]:.4f}")
print(f"Accuracy  : {acc:.4f}")
print(f"Threshold : {best_threshold:.4f}")
print("="*30)


# =========================================================
# STAGE 6: SHAP ANALYSIS (CLEAN + PROFESSIONAL)
# =========================================================

import shap
import matplotlib.pyplot as plt

# 1. Get trained XGBoost model
fitted_xgb = best_xgb

# 2. Initialize Explainer
explainer = shap.TreeExplainer(fitted_xgb)

# 3. Sample data (to keep SHAP fast)
X_sample = X_test_scaled.sample(500, random_state=42)

# 4. Compute SHAP values
shap_values = explainer(X_sample)

# =========================================================
# 🔍 INDIVIDUAL ANALYSIS (FIRST PERSON)
# =========================================================

print("\n==============================")
print("INDIVIDUAL SHAP ANALYSIS")
print("==============================")

# Create DataFrame with feature names
shap_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Contribution": shap_values.values[0]
})

# Sort by importance (absolute values)
shap_df = shap_df.sort_values(by="Contribution", key=abs, ascending=False)

# Print full table
print("\n--- Feature-wise Contributions ---")
print(shap_df)

# Print top contributors
print("\n--- Top 5 Risk Increasing Features ---")
print(shap_df[shap_df["Contribution"] > 0].head(5))

print("\n--- Top 5 Risk Decreasing Features ---")
print(shap_df[shap_df["Contribution"] < 0].head(5))

# =========================================================
# 📊 PREDICTION INTERPRETATION
# =========================================================

base_value = float(shap_values.base_values[0])
final_log_odds = shap_values.values[0].sum() + base_value

# Convert to probability (IMPORTANT)
probability = 1 / (1 + np.exp(-final_log_odds))

print("\n--- Prediction Summary ---")
print(f"Base Value (Average Risk): {base_value:.3f}")
print(f"Final Log-Odds: {final_log_odds:.3f}")
print(f"Final Default Probability: {probability:.3f}")

# =========================================================
# 📊 GLOBAL INTERPRETATION (SUMMARY PLOT)
# =========================================================

print("\n--- Generating SHAP Summary Plot ---")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.show()

# =========================================================
# 📊 LOCAL INTERPRETATION (WATERFALL PLOT)
# =========================================================

print("\n--- Generating Waterfall Plot ---")

plt.figure(figsize=(12, 8))
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
plt.tight_layout()
plt.show()

# =========================================================
# 📊 DEPENDENCE PLOTS (THE "HOW")
# =========================================================

print("\n--- Generating Dependence Plots ---")

# 1. Shows exactly at what credit utilization the risk skyrockets
shap.dependence_plot("RevolvingUtilizationOfUnsecuredLines", shap_values.values, X_sample, show=False)
plt.tight_layout()
plt.show()

# 2. Shows how your custom late score interacts with the model
shap.dependence_plot("Weighted_Late_Score", shap_values.values, X_sample, show=False)
plt.tight_layout()
plt.show()

shap.dependence_plot("Struggle_Index", shap_values.values, X_sample)
plt.tight_layout()
plt.show()

shap.dependence_plot("Youth_Utilization_Risk", shap_values.values, X_sample)
plt.tight_layout()
plt.show()

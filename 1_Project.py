import pandas as pd

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
print("\n--- Starting Feature Engineering ---")

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

# 5. Drop redundant raw columns to prevent XGBoost confusion
df = df.drop(columns=[
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
])
print(f"Dataset shape after feature engineering: {df.shape}")
# =====================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Split the data into Train and Test
X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Initialize the Scaler
scaler = StandardScaler()

# 3. FIT and TRANSFORM the training data (Learn the rules & apply them)
X_train_scaled = scaler.fit_transform(X_train)

# 4. ONLY TRANSFORM the test data (Apply the learned rules, DO NOT learn new ones)
X_test_scaled = scaler.transform(X_test)

# 5. Convert back to DataFrames (To keep column names for SHAP analysis later)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)


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

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score
)

# =========================================================
# 1. EVALUATION FUNCTION (CORE)
# =========================================================
def evaluate_model(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

# =========================================================
# 2. DUMMY MODEL
# =========================================================
# dummy = DummyClassifier(strategy='stratified', random_state=42)
# dummy.fit(X_train, y_train)

# dummy_prob = dummy.predict_proba(X_test)[:, 1]
# evaluate_model("Dummy Classifier", y_test, dummy_prob)

# =========================================================
# 3. LOGISTIC REGRESSION
# =========================================================
# lr = LogisticRegression(class_weight='balanced', max_iter=2000)
# lr.fit(X_train_scaled, y_train)

# lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
# evaluate_model("Logistic Regression", y_test, lr_prob)

# =========================================================
# 4. RANDOM FOREST (BASELINE)
# =========================================================
# rf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=10,
#     min_samples_leaf=20,
#     class_weight='balanced',
#     random_state=42,
#     n_jobs=-1
# )

# rf.fit(X_train_scaled, y_train)

# rf_prob = rf.predict_proba(X_test_scaled)[:, 1]
# evaluate_model("Random Forest", y_test, rf_prob)

from xgboost import XGBClassifier

# =========================================================
# 5. XGBOOST (SLIGHTLY OPTIMIZED)
# =========================================================

# Calculate imbalance ratio
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train_scaled, y_train)

xgb_prob = xgb.predict_proba(X_test_scaled)[:, 1]

# evaluate_model("XGBoost", y_test, xgb_prob)


def find_best_threshold(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\n{model_name} - Optimal Threshold")
    print("-" * 40)
    print("Best Threshold:", round(best_threshold, 4))
    print("Best F1:", round(f1_scores[best_idx], 4))
    print("Precision:", round(precision[best_idx], 4))
    print("Recall:", round(recall[best_idx], 4))

    return best_threshold

# find_best_threshold(y_test, lr_prob, "Logistic Regression")
# find_best_threshold(y_test, rf_prob, "Random Forest")
find_best_threshold(y_test, xgb_prob, "XGBoost")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, average_precision_score

from sklearn.calibration import CalibratedClassifierCV

# =====================================================================
# STAGE 3.5: THE 60/40 TARGETED EXPERIMENT (THE "NUCLEAR" OPTION)
# =====================================================================
print("\n" + "="*50)
print("TRAINING TARGETED XGBOOST (Optimizing for AUCPR + Constraints)...")
print("="*50)

# Define which features should always increase risk (1) or decrease risk (-1)
# This prevents the model from "hallucinating" patterns in noise
constraints = {
    'RevolvingUtilizationOfUnsecuredLines': 1,
    'Weighted_Late_Score': 1,
    'Total_Late_Events': 1,
    'DebtRatio': 1,
    'Absolute_Monthly_Debt': 1,
    'age': -1, # Older people are generally lower risk
    'Remaining_Living_Money': -1 # More cash = lower risk
}

# Map constraints to the columns in X_train_scaled
# (We fill 0 for columns that don't have a strict rule)
monotonic_vec = [constraints.get(col, 0) for col in X_train_scaled.columns]

targeted_xgb = XGBClassifier(
    max_depth=4, # Shallower trees = less overfitting to noise
    learning_rate=0.01,
    n_estimators=600,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=25,
    gamma=2,
    scale_pos_weight=5, # Softer weighting to protect Precision
    eval_metric='aucpr', # Optimizing the specific curve we care about
    monotone_constraints=tuple(monotonic_vec), # Forcing the model to be logical
    random_state=42,
    n_jobs=-1
)

targeted_xgb.fit(X_train_scaled, y_train)
targeted_prob = targeted_xgb.predict_proba(X_test_scaled)[:, 1]

# =========================================================
# STAGE 3.6: THE "60/40" SEARCH
# =========================================================
def find_60_40_target(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # We find the point closest to 60% Recall
    target_recall = 0.60
    idx = np.argmin(np.abs(recalls - target_recall))
    
    best_threshold = thresholds[min(idx, len(thresholds)-1)]

    print(f"\n🎯 TARGETING 60% RECALL")
    print("-" * 40)
    print(f"Threshold used : {best_threshold:.4f}")
    print(f"Final Precision: {precisions[idx]:.4f}")
    print(f"Final Recall   : {recalls[idx]:.4f}")
    
    f1 = 2 * (precisions[idx] * recalls[idx]) / (precisions[idx] + recalls[idx])
    print(f"Final F1 Score : {f1:.4f}")

find_60_40_target(y_test, targeted_prob)
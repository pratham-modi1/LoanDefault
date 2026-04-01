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
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(known[features], known['MonthlyIncome'])

# -----------------------------
# STEP 4: Predict missing income
# -----------------------------
predicted_income = model.predict(unknown[features])

# -----------------------------
# STEP 5: Fill missing values
# -----------------------------
df.loc[df['MonthlyIncome'].isna(), 'MonthlyIncome'] = predicted_income

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



cols = [
    'RevolvingUtilizationOfUnsecuredLines',
    'DebtRatio',
    'MonthlyIncome',
    'age',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberRealEstateLoansOrLines',
    'NumberOfDependents'
]

for col in cols:
    print(f"\n========== {col} ==========")
    
    # Basic stats
    print("\n--- Describe ---")
    print(df[col].describe())
    
    # Percentiles (only for continuous features)
    if df[col].dtype != 'int64':
        print("\n--- Percentiles ---")
        print(df[col].quantile([0.90, 0.95, 0.99, 0.999]))
    
    # Top extreme values
    print("\n--- Top 10 Highest Values ---")
    print(df[col].sort_values(ascending=False).head(10))
    
    # Value counts (useful for count/discrete features)
    print("\n--- Value Counts (Top) ---")
    print(df[col].value_counts().head(10))
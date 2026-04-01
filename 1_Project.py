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
print(df.head())
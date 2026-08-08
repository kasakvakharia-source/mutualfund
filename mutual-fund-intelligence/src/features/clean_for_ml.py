import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/fund_metrics_enriched.csv")
df['scheme_name'] = df['scheme_name'].fillna('')
df['scheme_category'] = df['scheme_category'].fillna('')
df['fund_house'] = df['fund_house'].fillna('')

# 1. Exclude dividend/IDCW plans (NAV-drop artifact corrupts return/vol calcs)
plan_upper = df['scheme_name'].str.upper()
is_dividend = plan_upper.str.contains('IDCW|DIVIDEND|INCOME DISTRIBUTION')

# 2. Must have core identity fields (drops the 30 failed detail-pulls)
has_core_fields = df['scheme_name'].str.len() > 0

# 3. Drop impossible volatility values (data errors, not real risk)
reasonable_vol = df['annual_volatility'] < 2.0  # >200% annualized is not real

# 4. Broad category grouping
def broad_cat(cat):
    c = cat.lower()
    if 'index' in c or 'fof' in c: return 'Passive/FoF'
    if 'liquid' in c or 'overnight' in c or 'money market' in c: return 'Cash Management'
    if 'equity' in c: return 'Equity'
    if 'debt' in c or 'income' in c or 'gilt' in c or 'bond' in c: return 'Debt'
    if 'hybrid' in c: return 'Hybrid'
    return 'Other'
df['category_group'] = df['scheme_category'].apply(broad_cat)

# 5. Scope: actively-managed skill-differentiated categories only
target_scope = df['category_group'].isin(['Equity', 'Debt'])

# 6. Must have sharpe_3y (needed for target)
has_sharpe = df['sharpe_3y'].notna()

clean = df[~is_dividend & has_core_fields & reasonable_vol & target_scope & has_sharpe].copy()

print(f"Rows before cleaning: {len(df)}")
print(f"Rows after cleaning: {len(clean)}")
print(clean['category_group'].value_counts())

clean.to_csv("data/processed/fund_metrics_ml_clean.csv", index=False)
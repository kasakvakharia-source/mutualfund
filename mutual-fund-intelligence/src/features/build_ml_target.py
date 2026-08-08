import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/fund_metrics_ml_clean.csv", parse_dates=['data_start', 'data_end'])

# Fund age as a feature (older funds have longer track records to judge)
df['fund_age_years'] = (df['data_end'] - df['data_start']).dt.days / 365.25

# Impute cagr_5y (some funds too young) + flag it — missingness itself can be informative
df['cagr_5y_missing'] = df['cagr_5y'].isna().astype(int)
df['cagr_5y'] = df['cagr_5y'].fillna(df.groupby('category_group')['cagr_5y'].transform('median'))

# TARGET: is this fund above its OWN category's median risk-adjusted return?
# Category-relative because Debt funds have structurally lower Sharpe than Equity —
# comparing them on an absolute scale would be unfair and meaningless.
df['category_median_sharpe'] = df.groupby('category_group')['sharpe_3y'].transform('median')
df['top_half_performer'] = (df['sharpe_3y'] > df['category_median_sharpe']).astype(int)

print(df.groupby('category_group')['top_half_performer'].value_counts())

# FEATURES: deliberately excluding annual_volatility and cagr_3y —
# sharpe_3y = (cagr_3y - risk_free) / annual_volatility, so including either
# would leak the exact formula that produced the target. Also excluding
# cagr_10y (too sparse) and fund_house (32 unique houses across 74 rows —
# too high-cardinality to encode reliably at this sample size).
feature_cols = [
    'cagr_1y', 'cagr_5y', 'cagr_5y_missing', 'max_drawdown',
    'expense_ratio', 'aum_cr', 'fund_age_years', 'category_group'
]

ml_ready = df[['scheme_code', 'scheme_name', 'fund_house'] + feature_cols + ['sharpe_3y', 'top_half_performer']]
ml_ready.to_csv("data/processed/fund_metrics_ml_ready.csv", index=False)
print(f"\nSaved {len(ml_ready)} rows, {len(feature_cols)} features, ready for modeling")
print(ml_ready.head())
import pandas as pd
import numpy as np
import glob
import os

FEATURE_LOOKBACK_YEARS = 3
TARGET_FORWARD_YEARS = 1
SNAPSHOT_FREQUENCY_MONTHS = 6

nav_files = glob.glob("data/raw/ml_nav_history/*.csv")
all_rows = []

for filepath in nav_files:
    scheme_code = os.path.basename(filepath).replace(".csv", "")
    df = pd.read_csv(filepath, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    if df.empty:
        continue

    data_start, data_end = df['date'].min(), df['date'].max()
    earliest_asof = data_start + pd.DateOffset(years=FEATURE_LOOKBACK_YEARS)
    latest_asof = data_end - pd.DateOffset(years=TARGET_FORWARD_YEARS)
    if earliest_asof >= latest_asof:
        continue

    asof_dates = pd.date_range(earliest_asof, latest_asof, freq=f"{SNAPSHOT_FREQUENCY_MONTHS}MS")

    for asof in asof_dates:
        hist = df[df['date'] <= asof]
        lookback = hist[hist['date'] >= asof - pd.DateOffset(years=FEATURE_LOOKBACK_YEARS)]
        future = df[df['date'] >= asof + pd.DateOffset(years=TARGET_FORWARD_YEARS)]

        if len(lookback) < 100 or future.empty:
            continue

        nav_asof = hist.iloc[-1]['nav']

        one_yr_ago = hist[hist['date'] <= asof - pd.DateOffset(years=1)]
        trailing_1y_return = (nav_asof / one_yr_ago.iloc[-1]['nav'] - 1) if not one_yr_ago.empty else np.nan

        returns = lookback['nav'].pct_change().dropna()
        trailing_vol = returns.std() * np.sqrt(252)

        running_max = lookback['nav'].cummax()
        trailing_mdd = (lookback['nav'] / running_max - 1).min()

        forward_nav = future.iloc[0]['nav']
        target_forward_1y_return = forward_nav / nav_asof - 1

        all_rows.append({
            "scheme_code": scheme_code,
            "as_of_date": asof,
            "trailing_1y_return": trailing_1y_return,
            "trailing_volatility": trailing_vol,
            "trailing_max_drawdown": trailing_mdd,
            "target_forward_1y_return": target_forward_1y_return
        })

training_df = pd.DataFrame(all_rows)
universe = pd.read_csv("data/raw/ml_universe.csv")
universe['scheme_code'] = universe['scheme_code'].astype(str)
training_df['scheme_code'] = training_df['scheme_code'].astype(str)
training_df = training_df.merge(universe[['scheme_code', 'scheme_name', 'category']], on='scheme_code', how='left')

training_df.to_csv("data/processed/ml_training_dataset.csv", index=False)
print(f"Generated {len(training_df)} point-in-time rows from {training_df['scheme_code'].nunique()} funds")
print(training_df['target_forward_1y_return'].describe())

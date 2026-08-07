import pandas as pd
import numpy as np
import os

nav = pd.read_csv("data/processed/nav_master.csv", parse_dates=['date'])
details = pd.read_csv("data/raw/scheme_details.csv")
details['scheme_code'] = details['scheme_code'].astype(str)
nav['scheme_code'] = nav['scheme_code'].astype(str)

RISK_FREE_RATE = 0.06  # approx Indian risk-free rate assumption, document this choice

def cagr(df, years):
    """Trailing CAGR ending at the fund's latest available date."""
    latest_date = df['date'].max()
    start_target = latest_date - pd.DateOffset(years=years)
    hist = df[df['date'] <= start_target]
    if hist.empty:
        return np.nan  # not enough history for this window
    start_nav = hist.iloc[-1]['nav']
    end_nav = df.iloc[-1]['nav']
    actual_years = (latest_date - hist.iloc[-1]['date']).days / 365.25
    if actual_years <= 0 or start_nav <= 0:
        return np.nan
    return (end_nav / start_nav) ** (1 / actual_years) - 1

def annual_volatility(df):
    returns = df['nav'].pct_change().dropna()
    returns = returns[np.isfinite(returns)]
    if returns.empty:
        return np.nan
    return returns.std() * np.sqrt(252)

def max_drawdown(df):
    running_max = df['nav'].cummax()
    drawdown = df['nav'] / running_max - 1
    return drawdown.min()

rows = []
for code, group in nav.groupby('scheme_code'):
    group = group.sort_values('date')
    vol = annual_volatility(group)
    c1 = cagr(group, 1)
    c3 = cagr(group, 3)
    c5 = cagr(group, 5)
    c10 = cagr(group, 10)
    mdd = max_drawdown(group)
    sharpe = (c3 - RISK_FREE_RATE) / vol if pd.notna(c3) and vol > 0 else np.nan

    rows.append({
        "scheme_code": code,
        "cagr_1y": c1, "cagr_3y": c3, "cagr_5y": c5, "cagr_10y": c10,
        "annual_volatility": vol,
        "max_drawdown": mdd,
        "sharpe_3y": sharpe,
        "data_start": group['date'].min(),
        "data_end": group['date'].max(),
    })

metrics = pd.DataFrame(rows)
metrics = metrics.merge(details[['scheme_code', 'scheme_name', 'fund_house', 'scheme_category']],
                         on='scheme_code', how='left')

os.makedirs("data/processed", exist_ok=True)
metrics.to_csv("data/processed/fund_metrics.csv", index=False)
print(metrics.sort_values('cagr_5y', ascending=False))
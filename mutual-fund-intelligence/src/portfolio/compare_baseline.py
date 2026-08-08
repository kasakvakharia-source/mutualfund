import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

metrics = pd.read_csv("data/processed/fund_metrics_ml_ready.csv")
metrics['scheme_code'] = metrics['scheme_code'].astype(str)
nav = pd.read_csv("data/processed/nav_master.csv", parse_dates=['date'])
nav['scheme_code'] = nav['scheme_code'].astype(str)
candidates = pd.read_csv("data/processed/candidate_pool.csv")
candidates['scheme_code'] = candidates['scheme_code'].astype(str)

# Fair baseline: SAME category mix (10 Equity + 5 Debt) as your ML pool,
# but selected by raw 1-year momentum instead of the ML quality_score
naive_equity = metrics[metrics['category_group'] == 'Equity'].nlargest(10, 'cagr_1y')['scheme_code'].tolist()
naive_debt = metrics[metrics['category_group'] == 'Debt'].nlargest(5, 'cagr_1y')['scheme_code'].tolist()
naive_matched_codes = naive_equity + naive_debt

def optimize(codes):
    recent = nav[nav['scheme_code'].isin(codes)].copy()
    cutoff = recent['date'].max() - pd.DateOffset(years=3)
    recent = recent[recent['date'] >= cutoff]
    price_matrix = recent.pivot(index='date', columns='scheme_code', values='nav')
    price_matrix = price_matrix.dropna(axis=1, thresh=int(len(price_matrix)*0.95)).ffill().dropna()
    mu = expected_returns.mean_historical_return(price_matrix)
    S = risk_models.CovarianceShrinkage(price_matrix).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.06)
    return ef.portfolio_performance(risk_free_rate=0.06)

ml_perf = optimize(candidates['scheme_code'].tolist())
naive_unrestricted_perf = optimize(metrics.nlargest(15, 'cagr_1y')['scheme_code'].tolist())
naive_matched_perf = optimize(naive_matched_codes)

print(f"ML-selected (10E+5D by quality_score):     return={ml_perf[0]*100:.2f}%  vol={ml_perf[1]*100:.2f}%  Sharpe={ml_perf[2]:.2f}")
print(f"Naive unrestricted (top-15 by 1Y CAGR):     return={naive_unrestricted_perf[0]*100:.2f}%  vol={naive_unrestricted_perf[1]*100:.2f}%  Sharpe={naive_unrestricted_perf[2]:.2f}")
print(f"Naive SAME MIX (10E+5D by 1Y CAGR):         return={naive_matched_perf[0]*100:.2f}%  vol={naive_matched_perf[1]*100:.2f}%  Sharpe={naive_matched_perf[2]:.2f}")
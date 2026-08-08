import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
import matplotlib.pyplot as plt

nav = pd.read_csv("data/processed/nav_master.csv", parse_dates=['date'])
nav['scheme_code'] = nav['scheme_code'].astype(str)
candidates = pd.read_csv("data/processed/candidate_pool.csv")
candidates['scheme_code'] = candidates['scheme_code'].astype(str)
code_to_name = dict(zip(candidates['scheme_code'], candidates['scheme_name']))
candidate_codes = candidates['scheme_code'].tolist()

# Use the last 3 years for the covariance estimate — matches the 3Y Sharpe
# window your scoring model was already built on, keeps things consistent
recent = nav[nav['scheme_code'].isin(candidate_codes)].copy()
cutoff = recent['date'].max() - pd.DateOffset(years=3)
recent = recent[recent['date'] >= cutoff]

price_matrix = recent.pivot(index='date', columns='scheme_code', values='nav')
# Drop any fund with too many gaps in this window, forward-fill small gaps in the rest
price_matrix = price_matrix.dropna(axis=1, thresh=int(len(price_matrix) * 0.95))
price_matrix = price_matrix.ffill().dropna()

print(f"Using {price_matrix.shape[1]} funds with clean 3-year history")

mu = expected_returns.mean_historical_return(price_matrix)
S = risk_models.CovarianceShrinkage(price_matrix).ledoit_wolf()  # shrinkage = more stable with few assets

# Max Sharpe portfolio
ef_sharpe = EfficientFrontier(mu, S)
ef_sharpe.max_sharpe(risk_free_rate=0.06)
weights_sharpe = ef_sharpe.clean_weights()
perf_sharpe = ef_sharpe.portfolio_performance(risk_free_rate=0.06)

# Min Volatility portfolio (more conservative alternative)
ef_minvol = EfficientFrontier(mu, S)
ef_minvol.min_volatility()
weights_minvol = ef_minvol.clean_weights()
perf_minvol = ef_minvol.portfolio_performance(risk_free_rate=0.06)

print("\n=== MAX SHARPE PORTFOLIO ===")
for code, w in weights_sharpe.items():
    if w > 0.01:
        print(f"  {code_to_name.get(code, code)[:40]:42s} {w*100:.1f}%")
print(f"Expected return: {perf_sharpe[0]*100:.2f}%  Volatility: {perf_sharpe[1]*100:.2f}%  Sharpe: {perf_sharpe[2]:.2f}")

print("\n=== MIN VOLATILITY PORTFOLIO ===")
for code, w in weights_minvol.items():
    if w > 0.01:
        print(f"  {code_to_name.get(code, code)[:40]:42s} {w*100:.1f}%")
print(f"Expected return: {perf_minvol[0]*100:.2f}%  Volatility: {perf_minvol[1]*100:.2f}%  Sharpe: {perf_minvol[2]:.2f}")

# Save both proposals
pd.DataFrame([
    {"scheme_code": c, "scheme_name": code_to_name.get(c, c), "weight_pct": round(w*100,2), "portfolio": "Max Sharpe"}
    for c, w in weights_sharpe.items() if w > 0.01
] + [
    {"scheme_code": c, "scheme_name": code_to_name.get(c, c), "weight_pct": round(w*100,2), "portfolio": "Min Volatility"}
    for c, w in weights_minvol.items() if w > 0.01
]).to_csv("data/processed/proposed_allocations.csv", index=False)

# Plot the efficient frontier
fig, ax = plt.subplots(figsize=(9, 6))
ef_plot = EfficientFrontier(mu, S)
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
ax.scatter(perf_sharpe[1], perf_sharpe[0], marker='*', s=200, color='red', label='Max Sharpe (proposed)')
ax.scatter(perf_minvol[1], perf_minvol[0], marker='*', s=200, color='blue', label='Min Volatility (alt.)')
ax.legend()
plt.savefig("data/processed/efficient_frontier.png", bbox_inches='tight')
print("\nSaved efficient_frontier.png and proposed_allocations.csv")
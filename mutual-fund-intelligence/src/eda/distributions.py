import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/fund_metrics_enriched.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].hist(df['cagr_5y'].dropna(), bins=10, edgecolor='black')
axes[0,0].set_title("5-Year CAGR Distribution")

axes[0,1].hist(df['annual_volatility'].dropna(), bins=10, edgecolor='black')
axes[0,1].set_title("Volatility Distribution")

axes[1,0].boxplot(df['sharpe_3y'].dropna())
axes[1,0].set_title("Sharpe Ratio (3Y) Spread")

axes[1,1].boxplot(df['expense_ratio'].dropna())
axes[1,1].set_title("Expense Ratio Spread")

plt.tight_layout()
plt.savefig("data/processed/eda_distributions.png")
print("Saved eda_distributions.png")
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/fund_metrics_enriched.csv")

amc_avg = df.groupby('fund_house')[['cagr_5y', 'sharpe_3y', 'expense_ratio']].mean().sort_values('sharpe_3y', ascending=False)
print(amc_avg)

amc_avg['sharpe_3y'].plot(kind='barh', figsize=(8,5), title="Avg Sharpe Ratio by AMC")
plt.tight_layout()
plt.savefig("data/processed/amc_sharpe_comparison.png")
print("Saved amc_sharpe_comparison.png")
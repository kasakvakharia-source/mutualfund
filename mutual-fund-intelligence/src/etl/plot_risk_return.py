import pandas as pd
import matplotlib.pyplot as plt

metrics = pd.read_csv("data/processed/fund_metrics.csv")

plt.figure(figsize=(10, 6))
plt.scatter(metrics['annual_volatility'], metrics['cagr_5y'], s=60)

for _, row in metrics.iterrows():
    plt.annotate(row['scheme_name'][:20], (row['annual_volatility'], row['cagr_5y']), fontsize=7)

plt.xlabel("Annual Volatility (Risk)")
plt.ylabel("5-Year CAGR (Return)")
plt.title("Risk vs Return - Selected Funds")
plt.grid(True, alpha=0.3)
plt.savefig("data/processed/risk_return_chart.png", bbox_inches='tight')
print("Saved chart")
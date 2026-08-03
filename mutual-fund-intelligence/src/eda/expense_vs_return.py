import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/fund_metrics_enriched.csv").dropna(subset=['expense_ratio', 'cagr_5y'])

correlation = df['expense_ratio'].corr(df['cagr_5y'])
print(f"Correlation between expense ratio and 5Y CAGR: {correlation:.3f}")

plt.figure(figsize=(8,6))
plt.scatter(df['expense_ratio'], df['cagr_5y'])
for _, row in df.iterrows():
    plt.annotate(row['scheme_name'][:15], (row['expense_ratio'], row['cagr_5y']), fontsize=7)
plt.xlabel("Expense Ratio (%)")
plt.ylabel("5-Year CAGR")
plt.title(f"Expense Ratio vs Return (correlation = {correlation:.2f})")
plt.grid(True, alpha=0.3)
plt.savefig("data/processed/expense_vs_return.png")
print("Saved expense_vs_return.png")
import pandas as pd

metrics = pd.read_csv("data/processed/fund_metrics.csv")
metrics['scheme_code'] = metrics['scheme_code'].astype(str)

expense = pd.read_csv("data/raw/expense_aum.csv")
expense['scheme_code'] = expense['scheme_code'].astype(str)

enriched = metrics.merge(expense, on='scheme_code', how='left')
enriched.to_csv("data/processed/fund_metrics_enriched.csv", index=False)

print(f"Funds with expense ratio: {enriched['expense_ratio'].notna().sum()} / {len(enriched)}")
print(enriched.head())
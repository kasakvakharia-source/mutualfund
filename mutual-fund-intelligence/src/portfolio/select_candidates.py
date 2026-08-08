
import pandas as pd

scores = pd.read_csv("data/processed/fund_scores.csv")
scores['scheme_code'] = scores['scheme_code'].astype(str)

# Top-scored funds from each category — gives the optimizer a genuinely
# good candidate pool to allocate across, not just "everything"
top_equity = scores[scores['category_group'] == 'Equity'].nlargest(10, 'quality_score')
top_debt = scores[scores['category_group'] == 'Debt'].nlargest(5, 'quality_score')

candidates = pd.concat([top_equity, top_debt])
candidates.to_csv("data/processed/candidate_pool.csv", index=False)
print(candidates[['scheme_name', 'category_group', 'quality_score', 'sharpe_3y']])
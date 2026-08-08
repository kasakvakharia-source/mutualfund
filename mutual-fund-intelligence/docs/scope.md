# Scope Decisions
- Fund category: Flexicap
- Time horizon: Jan 2015 - present
- Number of funds: 15-25
- User persona: Analyst at my own AMC deciding what to build

# Day 2: 
bulk pull NAV history and scheme details for selected funds

## Data Notes (Day 3)
- Risk-free rate assumption for Sharpe: 6%
- Funds with <10yrs history show NaN for cagr_10y (expected)
- [note any funds that looked odd/outlier here]

## Day 7 - ML Data Prep
- Excluded IDCW/Dividend plans (NAV-drop artifact inflates volatility)
- Excluded Passive/FoF and Cash Management categories from supervised scoring
  (skill-based scoring doesn't apply to index-tracking or cash instruments)
- Hybrid excluded for now — only 11 clean rows, too few to model
- Final ML dataset: 74 rows (Equity + Debt), 8 features
- Target: category-relative top-half Sharpe performer (median split, not quartile,
  chosen because quartile splits would leave too few positive cases per category at this sample size)
- Excluded annual_volatility and cagr_3y from features (leakage — they define sharpe_3y directly)
- Excluded fund_house as a feature (32 unique values across 74 rows — too sparse to encode)
# Model Card

# Fund Quality Score - Model Card

**Task**: Binary classification — is a fund in the top half of its category (Equity/Debt) by 3-year Sharpe ratio?
**Data**: 74 funds (52 Equity, 22 Debt), cleaned from an initial 472-row pull
**Features**: cagr_1y, cagr_5y (+missing flag), max_drawdown, expense_ratio, aum_cr, fund_age_years, is_equity
**Excluded features**: annual_volatility, cagr_3y (leak the Sharpe formula directly), cagr_10y (43% missing), fund_house (32 unique values, too sparse)
**Model**: Soft-voting ensemble of Logistic Regression, Random Forest, XGBoost
**Validation**: Repeated 5-fold stratified CV (10 repeats = 50 evaluations)
**Result**: ~82% accuracy, AUC ~0.92, vs 47% baseline
**Key limitation**: Small sample (74 rows) — CV numbers should be read as "meaningfully better than chance," not as precise figures. Model reflects historical patterns only; not predictive of future fund performance.
**Top features**: cagr_5y and cagr_1y (return persistence) dominate; category (Equity vs Debt) has negligible importance once category-relative target normalization is applied.

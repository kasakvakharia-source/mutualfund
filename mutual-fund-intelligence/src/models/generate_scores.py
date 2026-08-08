import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

df = pd.read_csv("data/processed/fund_metrics_ml_ready.csv")
df['is_equity'] = (df['category_group'] == 'Equity').astype(int)
feature_cols = ['cagr_1y', 'cagr_5y', 'cagr_5y_missing', 'max_drawdown',
                'expense_ratio', 'aum_cr', 'fund_age_years', 'is_equity']
X = df[feature_cols]
y = df['top_half_performer']

lr = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42)
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_clf)], voting='soft')

# Fit on ALL available data for the final scoring model
# (the CV in Step 1 is what tells us this is trustworthy — this fit is for production scoring)
ensemble.fit(X, y)

df['quality_score'] = (ensemble.predict_proba(X)[:, 1] * 100).round(1)

output = df[['scheme_code', 'scheme_name', 'fund_house', 'category_group',
             'sharpe_3y', 'quality_score']].sort_values('quality_score', ascending=False)
output.to_csv("data/processed/fund_scores.csv", index=False)
print(output.head(15))
print(f"\nSaved scores for {len(output)} funds")
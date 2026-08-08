import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/fund_metrics_ml_ready.csv")
df['is_equity'] = (df['category_group'] == 'Equity').astype(int)
feature_cols = ['cagr_1y', 'cagr_5y', 'cagr_5y_missing', 'max_drawdown',
                'expense_ratio', 'aum_cr', 'fund_age_years', 'is_equity']
X = df[feature_cols]
y = df['top_half_performer']

rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42).fit(X, y)
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0).fit(X, y)

imp = pd.DataFrame({
    'feature': feature_cols,
    'rf_importance': rf.feature_importances_,
    'xgb_importance': xgb_clf.feature_importances_
})
imp['avg_importance'] = (imp['rf_importance'] + imp['xgb_importance']) / 2
imp = imp.sort_values('avg_importance', ascending=False)
print(imp)

imp.plot(x='feature', y='avg_importance', kind='barh', figsize=(8,5), legend=False,
          title='Feature Importance (avg of RF + XGBoost)')
plt.tight_layout()
plt.savefig("data/processed/feature_importance.png")
print("Saved feature_importance.png")
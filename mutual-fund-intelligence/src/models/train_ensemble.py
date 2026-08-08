import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
import xgboost as xgb

df = pd.read_csv("data/processed/fund_metrics_ml_ready.csv")
df['is_equity'] = (df['category_group'] == 'Equity').astype(int)

feature_cols = ['cagr_1y', 'cagr_5y', 'cagr_5y_missing', 'max_drawdown',
                'expense_ratio', 'aum_cr', 'fund_age_years', 'is_equity']
X = df[feature_cols]
y = df['top_half_performer']

# 5-fold CV repeated 10 times = 50 total evaluations, gives a much more stable
# estimate than a single train/test split would with only 74 rows
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

print("=== Baseline (always guess majority class) ===")
baseline = DummyClassifier(strategy='most_frequent')
base_scores = cross_val_score(baseline, X, y, cv=cv, scoring='accuracy')
print(f"Accuracy: {base_scores.mean():.3f}")

lr = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42)
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)

print("\n=== Individual models ===")
for name, model in [('LogisticRegression', lr), ('RandomForest', rf), ('XGBoost', xgb_clf)]:
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"{name}: accuracy={acc.mean():.3f} (+/-{acc.std():.3f})  AUC={auc.mean():.3f}")

print("\n=== Ensemble (soft voting) ===")
ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_clf)], voting='soft')
ens_acc = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
ens_auc = cross_val_score(ensemble, X, y, cv=cv, scoring='roc_auc')
print(f"Ensemble: accuracy={ens_acc.mean():.3f} (+/-{ens_acc.std():.3f})  AUC={ens_auc.mean():.3f}")

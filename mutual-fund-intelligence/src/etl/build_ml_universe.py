import pandas as pd

matched = pd.read_csv("data/raw/broad_universe_matched.csv")

# Keep it coherent: diversified equity categories only, similar spirit to your original scope
target_categories = ["Flexi Cap Fund", "Large Cap Fund", "Large & Mid Cap Fund", "Multi Cap Fund", "Mid Cap Fund"]
ml_universe = matched[matched['category'].isin(target_categories)].drop_duplicates(subset='scheme_code')

ml_universe = ml_universe[['scheme_code', 'matched_name', 'category']].rename(columns={'matched_name': 'scheme_name'})
ml_universe.to_csv("data/raw/ml_universe.csv", index=False)
print(f"ML universe: {len(ml_universe)} funds")
print(ml_universe['category'].value_counts())
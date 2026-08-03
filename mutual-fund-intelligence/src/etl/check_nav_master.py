import pandas as pd

df = pd.read_csv("data/processed/nav_master.csv", parse_dates=['date'])

# Check for duplicate date/scheme rows
dupes = df.duplicated(subset=['scheme_code', 'date']).sum()
print(f"Duplicate rows: {dupes}")

# Check NAV value sanity
print(f"NAV min: {df['nav'].min()}, max: {df['nav'].max()}")
print(f"Any zero or negative NAVs? {(df['nav'] <= 0).sum()}")

# Check date coverage per fund
coverage = df.groupby('scheme_code')['date'].agg(['min', 'max', 'count'])
print(coverage)
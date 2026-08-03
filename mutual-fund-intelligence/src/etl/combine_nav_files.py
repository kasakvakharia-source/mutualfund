import pandas as pd
import glob
import os

nav_files = glob.glob("data/raw/nav_history/*.csv")
all_dfs = []

for filepath in nav_files:
    scheme_code = os.path.basename(filepath).replace(".csv", "")
    df = pd.read_csv(filepath, parse_dates=['date'])
    df['scheme_code'] = scheme_code
    all_dfs.append(df)

master = pd.concat(all_dfs, ignore_index=True)
master = master.sort_values(['scheme_code', 'date']).reset_index(drop=True)

os.makedirs("data/processed", exist_ok=True)
master.to_csv("data/processed/nav_master.csv", index=False)

print(f"Combined {len(nav_files)} files into {len(master)} total rows")
print(master.head())
print(master['scheme_code'].nunique(), "unique funds")
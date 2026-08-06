import pandas as pd
from difflib import get_close_matches

broad = pd.read_csv("data/raw/broad_universe_performance.csv")
all_codes = pd.read_csv("data/raw/all_scheme_codes.csv")

# Focus matching on Direct-Growth plans only - cleanest, most comparable set
direct_growth = all_codes[
    all_codes['scheme_name'].str.contains("Direct", case=False, na=False) &
    all_codes['scheme_name'].str.contains("Growth", case=False, na=False)
]
name_to_code = dict(zip(direct_growth['scheme_name'], direct_growth['scheme_code']))
name_pool = list(name_to_code.keys())

def match(perf_name):
    close = get_close_matches(perf_name, name_pool, n=1, cutoff=0.55)
    if close:
        return name_to_code[close[0]], close[0]
    return None, None

matched_codes, matched_names = [], []
for name in broad['scheme_name']:
    code, matched_name = match(str(name))
    matched_codes.append(code)
    matched_names.append(matched_name)

broad['scheme_code'] = matched_codes
broad['matched_name'] = matched_names

matched = broad.dropna(subset=['scheme_code']).copy()
print(f"Matched {len(matched)} / {len(broad)} funds")

matched.to_csv("data/raw/broad_universe_matched.csv", index=False)
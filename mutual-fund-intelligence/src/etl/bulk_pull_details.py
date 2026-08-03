from mftool import Mftool
import pandas as pd
import time

mf = Mftool()

selected = pd.read_csv("data/raw/selected_schemes.csv")
rows = []

for _, row in selected.iterrows():
    code = str(row['scheme_code'])
    try:
        details = mf.get_scheme_details(code)
        details['scheme_code'] = code
        rows.append(details)
        print(f"OK  {code}")
    except Exception as e:
        print(f"FAIL {code} - {e}")
    time.sleep(0.5)

pd.DataFrame(rows).to_csv("data/raw/scheme_details.csv", index=False)
print("Saved data/raw/scheme_details.csv")
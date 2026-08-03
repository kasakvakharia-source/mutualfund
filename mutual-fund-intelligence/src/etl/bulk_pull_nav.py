from mftool import Mftool
import pandas as pd
import time
import os

mf = Mftool()

os.makedirs("data/raw/nav_history", exist_ok=True)

selected = pd.read_csv("data/raw/selected_schemes.csv")
log = []

for _, row in selected.iterrows():
    code = str(row['scheme_code'])
    name = row['scheme_name']
    try:
        df = mf.get_scheme_historical_nav(code, as_Dataframe=True).reset_index()
        df['nav'] = df['nav'].astype(float)
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df = df.sort_values('date').reset_index(drop=True)

        outfile = f"data/raw/nav_history/{code}.csv"
        df.to_csv(outfile, index=False)

        log.append({
            "scheme_code": code, "scheme_name": name,
            "status": "success", "rows": len(df),
            "start_date": df['date'].min(), "end_date": df['date'].max()
        })
        print(f"OK  {code} - {name} - {len(df)} rows")

    except Exception as e:
        log.append({"scheme_code": code, "scheme_name": name, "status": f"FAILED: {e}"})
        print(f"FAIL {code} - {name} - {e}")

    time.sleep(0.5)  # be polite to the API, don't hammer it

log_df = pd.DataFrame(log)
log_df.to_csv("data/raw/pull_log.csv", index=False)
print("\nDone. See data/raw/pull_log.csv for a summary.")
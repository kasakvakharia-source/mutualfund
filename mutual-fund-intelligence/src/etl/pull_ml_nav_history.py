from mftool import Mftool
import pandas as pd
import time
import os

mf = Mftool()
os.makedirs("data/raw/ml_nav_history", exist_ok=True)

universe = pd.read_csv("data/raw/ml_universe.csv")
log = []

for _, row in universe.iterrows():
    code = str(row['scheme_code'])
    try:
        df = mf.get_scheme_historical_nav(code, as_Dataframe=True).reset_index()
        df['nav'] = df['nav'].astype(float)
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df = df.sort_values('date').reset_index(drop=True)
        df.to_csv(f"data/raw/ml_nav_history/{code}.csv", index=False)
        log.append({"scheme_code": code, "status": "success", "rows": len(df)})
        print(f"OK  {code}")
    except Exception as e:
        log.append({"scheme_code": code, "status": f"FAILED: {e}"})
        print(f"FAIL {code} - {e}")
    time.sleep(0.4)

pd.DataFrame(log).to_csv("data/raw/ml_pull_log.csv", index=False)
print("Done - see ml_pull_log.csv")
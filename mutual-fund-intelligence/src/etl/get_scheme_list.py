from mftool import Mftool
import pandas as pd

mf = Mftool()
all_schemes = mf.get_scheme_codes()

df = pd.DataFrame(list(all_schemes.items()), columns=["scheme_code", "scheme_name"])
df.to_csv("data/raw/all_scheme_codes.csv", index=False)
print(f"Total schemes pulled: {len(df)}")
print(df.head(10))
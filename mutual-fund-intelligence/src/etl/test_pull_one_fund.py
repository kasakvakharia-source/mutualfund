from mftool import Mftool
import pandas as pd
import matplotlib.pyplot as plt

mf = Mftool()

test_code = "139201"

# Pull as a DataFrame directly
df = mf.get_scheme_historical_nav(test_code, as_Dataframe=True).reset_index()
df['nav'] = df['nav'].astype(float)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df = df.sort_values('date').reset_index(drop=True)

print(df.head())
print(df.tail())
print(f"Total rows: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Quick visual sanity check
df.plot(x='date', y='nav', title=f"NAV history - {test_code}")
plt.savefig("data/raw/test_plot.png")
print("Saved test_plot.png - open it to eyeball the trend")
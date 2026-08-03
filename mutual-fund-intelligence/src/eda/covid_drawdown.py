import pandas as pd

nav = pd.read_csv("data/processed/nav_master.csv", parse_dates=['date'])
nav['scheme_code'] = nav['scheme_code'].astype(str)
details = pd.read_csv("data/processed/fund_metrics_enriched.csv")
details['scheme_code'] = details['scheme_code'].astype(str)

covid_start = pd.Timestamp("2020-02-01")
crash_bottom = pd.Timestamp("2020-03-23")

results = []
for code, group in nav.groupby('scheme_code'):
    group = group.sort_values('date')
    pre = group[group['date'] <= covid_start]
    window = group[(group['date'] >= covid_start) & (group['date'] <= crash_bottom)]
    if pre.empty or window.empty:
        continue
    pre_nav = pre.iloc[-1]['nav']
    lowest_nav = window['nav'].min()
    drawdown_pct = (lowest_nav - pre_nav) / pre_nav
    results.append({"scheme_code": code, "covid_drawdown_pct": drawdown_pct})

drawdown_df = pd.DataFrame(results).merge(
    details[['scheme_code', 'scheme_name']], on='scheme_code', how='left'
)
drawdown_df = drawdown_df.sort_values('covid_drawdown_pct', ascending=False)
drawdown_df.to_csv("data/processed/covid_drawdown.csv", index=False)
print(drawdown_df)
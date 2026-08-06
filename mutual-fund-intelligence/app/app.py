import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Mutual Fund Intelligence", layout="wide")

@st.cache_data
def load_data():
    metrics = pd.read_csv("data/processed/fund_metrics_enriched.csv")
    nav = pd.read_csv("data/processed/nav_master.csv", parse_dates=['date'])
    nav['scheme_code'] = nav['scheme_code'].astype(str)
    metrics['scheme_code'] = metrics['scheme_code'].astype(str)
    return metrics, nav

metrics, nav = load_data()

st.title("Mutual Fund Intelligence Platform")
st.caption("Analysis of Flexicap funds — built as the foundation for designing a new fund")

st.write(f"Loaded {len(metrics)} funds, {len(nav)} NAV data points")
st.header("1. Fund Explorer")

col1, col2, col3 = st.columns(3)
with col1:
    amc_filter = st.multiselect("Fund House", options=sorted(metrics['fund_house'].dropna().unique()))
with col2:
    min_cagr = st.slider("Minimum 5Y CAGR (%)", -20.0, 40.0, -20.0)
with col3:
    max_vol = st.slider("Maximum Volatility", 0.0, 0.5, 0.5)

filtered = metrics.copy()
if amc_filter:
    filtered = filtered[filtered['fund_house'].isin(amc_filter)]
filtered = filtered[
    (filtered['cagr_5y'] * 100 >= min_cagr) &
    (filtered['annual_volatility'] <= max_vol)
]

st.write(f"Showing {len(filtered)} of {len(metrics)} funds")
st.dataframe(
    filtered[['scheme_name', 'fund_house', 'cagr_1y', 'cagr_3y', 'cagr_5y',
              'annual_volatility', 'sharpe_3y', 'max_drawdown', 'expense_ratio']]
    .sort_values('sharpe_3y', ascending=False),
    use_container_width=True
)


st.header("2. Risk vs Return Landscape")

fig = px.scatter(
    filtered, x='annual_volatility', y='cagr_5y',
    hover_name='scheme_name', color='fund_house',
    labels={'annual_volatility': 'Volatility (Risk)', 'cagr_5y': '5-Year CAGR'},
    title="Risk vs Return (respects filters above)"
)
st.plotly_chart(fig, use_container_width=True)
st.header("3. Compare Funds")

selected_funds = st.multiselect(
    "Pick 2-4 funds to compare",
    options=metrics['scheme_name'].tolist(),
    default=metrics['scheme_name'].tolist()[:2]
)

if len(selected_funds) >= 2:
    selected_codes = metrics[metrics['scheme_name'].isin(selected_funds)]['scheme_code'].tolist()

    # Metrics side by side
    compare_table = metrics[metrics['scheme_code'].isin(selected_codes)][
        ['scheme_name', 'cagr_1y', 'cagr_3y', 'cagr_5y', 'annual_volatility',
         'sharpe_3y', 'max_drawdown', 'expense_ratio']
    ].set_index('scheme_name').T
    st.dataframe(compare_table, use_container_width=True)

    # Indexed NAV growth chart (normalize all to 100 at earliest common date)
    compare_nav = nav[nav['scheme_code'].isin(selected_codes)].copy()
    compare_nav = compare_nav.merge(metrics[['scheme_code', 'scheme_name']], on='scheme_code')

    indexed_frames = []
    for code, group in compare_nav.groupby('scheme_code'):
        group = group.sort_values('date').copy()
        group['indexed_nav'] = group['nav'] / group['nav'].iloc[0] * 100
        indexed_frames.append(group)
    indexed_df = pd.concat(indexed_frames)

    fig2 = px.line(indexed_df, x='date', y='indexed_nav', color='scheme_name',
                    title="Growth of ₹100 invested (indexed comparison)")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Select at least 2 funds to compare")
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_data(show_spinner=False)
def get_ai_summary(scheme_name, category, score, sharpe):
    prompt = f"""Write a 3-4 sentence factual, neutral summary of this fund for a retail investor.
Fund: {scheme_name} | Category: {category} | Quality Score: {score}/100 | 3Y Sharpe: {sharpe:.2f}
End with a note that this is historical data, not a guarantee. No investment advice."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=200
    )
    return response.choices[0].message.content

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
    st.header("4. What-If: Expense Ratio Sensitivity")

wf_fund = st.selectbox("Pick a fund", options=metrics['scheme_name'].tolist())
wf_row = metrics[metrics['scheme_name'] == wf_fund].iloc[0]

current_expense = wf_row['expense_ratio']
current_cagr = wf_row['cagr_5y']

new_expense = st.slider("Simulated expense ratio (%)", 0.0, 3.0, float(current_expense), 0.05)

# Approximate: add back current expense drag to estimate "gross" return, then apply new expense
approx_gross_return = current_cagr + (current_expense / 100)
approx_net_return_new = approx_gross_return - (new_expense / 100)

investment = st.number_input("Initial investment (₹)", value=100000, step=10000)
years = st.slider("Investment horizon (years)", 1, 20, 5)

fv_current = investment * (1 + current_cagr) ** years
fv_new = investment * (1 + approx_net_return_new) ** years

col1, col2 = st.columns(2)
col1.metric("Value at current expense ratio", f"₹{fv_current:,.0f}")
col2.metric(f"Value at {new_expense}% expense ratio", f"₹{fv_new:,.0f}",
            delta=f"₹{fv_new - fv_current:,.0f}")

st.caption("Approximation: assumes expense drag is linear and additive to CAGR — illustrative, not exact fund accounting.")
st.header("5. SIP Simulator")

sip_fund = st.selectbox("Pick a fund for SIP simulation", options=metrics['scheme_name'].tolist(), key="sip_fund")
sip_code = metrics[metrics['scheme_name'] == sip_fund]['scheme_code'].iloc[0]

sip_amount = st.number_input("Monthly SIP amount (₹)", value=5000, step=500)
sip_years = st.slider("SIP duration (years)", 1, 10, 5, key="sip_years")

fund_nav = nav[nav['scheme_code'] == sip_code].sort_values('date')
end_date = fund_nav['date'].max()
start_date = end_date - pd.DateOffset(years=sip_years)
sip_window = fund_nav[fund_nav['date'] >= start_date].copy()

if not sip_window.empty:
    sip_window['month'] = sip_window['date'].dt.to_period('M')
    monthly_nav = sip_window.groupby('month').first().reset_index()

    total_units = 0
    total_invested = 0
    for _, row in monthly_nav.iterrows():
        total_units += sip_amount / row['nav']
        total_invested += sip_amount

    final_value = total_units * fund_nav.iloc[-1]['nav']

    col1, col2, col3 = st.columns(3)
    col1.metric("Total invested", f"₹{total_invested:,.0f}")
    col2.metric("Final value", f"₹{final_value:,.0f}")
    col3.metric("Gain", f"₹{final_value - total_invested:,.0f}",
                delta=f"{((final_value/total_invested)-1)*100:.1f}%")
else:
    st.warning("Not enough history for this duration")

st.header("6. Portfolio Allocation What-If")

portfolio_funds = st.multiselect(
    "Build a hypothetical portfolio (2-5 funds)",
    options=metrics['scheme_name'].tolist(),
    default=metrics['scheme_name'].tolist()[:3],
    key="portfolio_funds"
)

total_weight = 0
if len(portfolio_funds) >= 2:
    portfolio_codes = metrics[metrics['scheme_name'].isin(portfolio_funds)]['scheme_code'].tolist()

    st.write("Set allocation weights (%) — should sum to 100")
    weights = {}
    cols = st.columns(len(portfolio_funds))
    default_weight = round(100 / len(portfolio_funds))
    for i, fund_name in enumerate(portfolio_funds):
        weights[fund_name] = cols[i].number_input(fund_name[:15], min_value=0, max_value=100,
                                                     value=default_weight, key=f"w_{i}")

    total_weight = sum(weights.values())
    st.write(f"Total allocation: {total_weight}%")

    if total_weight == 100:
        wide_nav = nav[nav['scheme_code'].isin(portfolio_codes)].pivot(index='date', columns='scheme_code', values='nav')
        daily_returns = wide_nav.pct_change().dropna()

        code_to_name = dict(zip(metrics['scheme_code'], metrics['scheme_name']))
        w_array = np.array([weights[code_to_name[c]] / 100 for c in daily_returns.columns])

        cagr_map = dict(zip(metrics['scheme_code'], metrics['cagr_5y']))
        expected_return = sum(w_array[i] * cagr_map[c] for i, c in enumerate(daily_returns.columns))

        cov_matrix = daily_returns.cov() * 252
        portfolio_vol = np.sqrt(w_array.T @ cov_matrix.values @ w_array)

        col1, col2 = st.columns(2)
        col1.metric("Expected Portfolio Return (annualized)", f"{expected_return*100:.2f}%")
        col2.metric("Expected Portfolio Volatility", f"{portfolio_vol*100:.2f}%")
        st.caption("Return: weighted average of historical 5Y CAGR. Volatility: accounts for diversification via historical covariance. Both backward-looking, not guarantees.")
    else:
        st.warning("Adjust weights to sum to exactly 100%")
else:
    st.info("Select at least 2 funds to build a portfolio")

st.header("7. Stress Test: COVID Crash Scenario")

covid_dd = pd.read_csv("data/processed/covid_drawdown.csv")
covid_dd['scheme_code'] = covid_dd['scheme_code'].astype(str)

if len(portfolio_funds) >= 2 and total_weight == 100:
    dd_map = dict(zip(covid_dd['scheme_code'], covid_dd['covid_drawdown_pct']))
    portfolio_dd = sum(w_array[i] * dd_map.get(c, 0) for i, c in enumerate(daily_returns.columns))
    st.metric("Estimated portfolio drawdown if a COVID-like crash repeated", f"{portfolio_dd*100:.1f}%")
    st.caption("Based on how each fund actually behaved Feb-Mar 2020, weighted by your allocation above. A historical scenario, not a prediction.")
else:
    st.info("Build a portfolio in Section 6 (weights summing to 100%) to see its stress-test result")

st.header("8. Proposed New Fund")

alloc = pd.read_csv("data/processed/proposed_allocations.csv")
portfolio_choice = st.radio("View allocation", ["Max Sharpe", "Min Volatility"])
show_alloc = alloc[alloc['portfolio'] == portfolio_choice]

fig3 = px.pie(show_alloc, names='scheme_name', values='weight_pct',
              title=f"Proposed New Fund Allocation ({portfolio_choice})")
st.plotly_chart(fig3, use_container_width=True)
st.dataframe(show_alloc[['scheme_name', 'weight_pct']], use_container_width=True)

st.image("data/processed/efficient_frontier.png", caption="Efficient frontier of candidate pool")

st.header("9. AI Fund Insights")
scores = pd.read_csv("data/processed/fund_scores.csv")
ai_fund = st.selectbox("Get an AI summary for", options=scores['scheme_name'].tolist(), key="ai_fund")
row = scores[scores['scheme_name'] == ai_fund].iloc[0]

if st.button("Generate Summary"):
    with st.spinner("Generating..."):
        summary = get_ai_summary(row['scheme_name'], row['category_group'], row['quality_score'], row['sharpe_3y'])
        st.info(summary)

from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_fund(fund_row):
    prompt = f"""You are a financial analyst writing a short, plain-English fund summary for a retail investor.

Fund: {fund_row['scheme_name']}
Category: {fund_row['category_group']}
Quality Score: {fund_row['quality_score']}/100
3-Year Sharpe Ratio: {fund_row['sharpe_3y']:.2f}

Write a 3-4 sentence summary of this fund's risk-return profile. Be factual and neutral,
not promotional. Do not give investment advice or tell the reader to buy/sell.
End with one sentence noting this is historical data, not a guarantee of future performance."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # lower = more consistent/factual, less creative variation
        max_tokens=200
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    scores = pd.read_csv("mutual-fund-intelligence/data/processed/fund_scores.csv")
    test_fund = scores.iloc[0]
    print(f"Testing on: {test_fund['scheme_name']}\n")
    print(summarize_fund(test_fund))
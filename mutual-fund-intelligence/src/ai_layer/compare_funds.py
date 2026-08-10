from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def compare_funds(fund_a, fund_b):
    prompt = f"""Compare these two mutual funds factually for a retail investor, in 4-5 sentences.
Cover both risk and return. Do not recommend one over the other or give advice — just describe the differences.

Fund A: {fund_a['scheme_name']} | Category: {fund_a['category_group']} | Score: {fund_a['quality_score']}/100 | Sharpe: {fund_a['sharpe_3y']:.2f}
Fund B: {fund_b['scheme_name']} | Category: {fund_b['category_group']} | Score: {fund_b['quality_score']}/100 | Sharpe: {fund_b['sharpe_3y']:.2f}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    scores = pd.read_csv("mutual-fund-intelligence/data/processed/fund_scores.csv")
    print(compare_funds(scores.iloc[0], scores.iloc[1]))
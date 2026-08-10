from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def explain_score(fund_row, top_features):
    features_text = ", ".join([f"{f}" for f in top_features])
    prompt = f"""You are explaining a machine learning fund score to a non-technical reader.

Fund: {fund_row['scheme_name']}
Score: {fund_row['quality_score']}/100
Category: {fund_row['category_group']}

The scoring model's most influential factors overall are (in order): {features_text}.

In 2-3 sentences, explain in plain English what kinds of things this score generally
reflects, based on those factors. Do not claim to know this specific fund's individual
factor values beyond what's given — speak generally about what drives scores in this model."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    scores = pd.read_csv("mutual-fund-intelligence/data/processed/fund_scores.csv")
    top_features = ["5-year historical return consistency", "1-year recent performance",
                     "fund age/track record length", "maximum historical drawdown"]
    test_fund = scores.iloc[0]
    print(explain_score(test_fund, top_features))

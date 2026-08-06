import json
import os

import pandas as pd
from mftool import Mftool


def parse_payload(payload):
    """Normalize the mftool response into a dictionary keyed by category."""
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            return {}
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return payload if isinstance(payload, dict) else {}


def extract_rows(data):
    """Flatten the category -> funds structure into a row-oriented list."""
    rows = []
    for category, funds in data.items():
        if not isinstance(funds, list):
            continue
        for fund in funds:
            if not isinstance(fund, dict):
                continue
            rows.append({
                "category": category,
                "scheme_name": fund.get("scheme_name"),
                "benchmark": fund.get("benchmark"),
                "return_1y_direct": fund.get("1-Year Return(%)- Direct"),
                "return_3y_direct": fund.get("3-Year Return(%)- Direct"),
                "return_5y_direct": fund.get("5-Year Return(%)- Direct"),
            })
    return rows


mf = Mftool()
data = mf.get_open_ended_equity_scheme_performance(as_json=True)
parsed_data = parse_payload(data)
rows = extract_rows(parsed_data)

df = pd.DataFrame(rows)
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/broad_universe_performance.csv", index=False)
print(f"Pulled {len(df)} funds across {df['category'].nunique()} categories")
print(df['category'].value_counts())
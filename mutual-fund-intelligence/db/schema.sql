CREATE TABLE IF NOT EXISTS funds (
    scheme_code TEXT PRIMARY KEY,
    scheme_name TEXT,
    fund_category TEXT,
    fund_house TEXT
);

CREATE TABLE IF NOT EXISTS nav_history (
    scheme_code TEXT,
    nav_date TEXT,
    nav REAL,
    PRIMARY KEY (scheme_code, nav_date)
);

New Fund Investment Thesis
Mandate

A Flexicap-style mutual fund that combines ML-selected equity funds (growth engine) with debt funds (stability sleeve) to maximize long-term risk-adjusted returns. Fund selection is driven by an ensemble machine learning quality score, while portfolio allocation is optimized using Modern Portfolio Theory (Mean-Variance Optimization).

Fund Selection Methodology

The proposed portfolio was constructed in two stages:

Machine Learning Screening
74 mutual funds were evaluated using an ensemble ML scoring model developed during this project.
Each fund received a Quality Score based on multiple performance and risk metrics.
The highest-scoring equity and debt funds were shortlisted.
Portfolio Optimization
The shortlisted funds were optimized using Mean-Variance Optimization (Maximum Sharpe Ratio).
Historical 3-year returns were used to estimate expected returns and the covariance matrix.
Two optimized portfolios were generated:
Maximum Sharpe Portfolio
Minimum Volatility Portfolio
Proposed Allocation (Maximum Sharpe Portfolio)
Fund	Weight
Aditya Birla Sun Life Fixed Term Plan – Series TQ (Direct Growth)	55.53%
HDFC FMP 1269D March 2023 – Growth Option	34.48%
HDFC Defence Fund – Direct Growth	9.99%
Expected Performance
Metric	Value
Expected Annual Return	10.91%
Expected Volatility	3.03%
Expected Sharpe Ratio	1.62

The optimizer allocated nearly 90% of the portfolio to fixed-income funds and approximately 10% to an equity fund, indicating that the selected debt funds offered superior risk-adjusted performance over the historical period while the equity allocation contributed additional return potential.

Efficient Frontier

Figure: Efficient Frontier with individual assets, Maximum Sharpe portfolio (red star), and Minimum Volatility portfolio (blue star).

Interpretation

Individual mutual funds are shown as black dots.
The blue curve represents the Efficient Frontier, illustrating the best achievable return for each level of portfolio risk.
The Maximum Sharpe portfolio lies on the frontier and provides the highest return per unit of risk.
The Minimum Volatility portfolio offers the lowest portfolio risk, making it suitable for conservative investors.
Most individual funds lie below the frontier, demonstrating that diversification improves portfolio efficiency.
Why ML-Based Selection Instead of Naive Fund Picking

Three portfolio construction strategies were compared.

Strategy	Return	Volatility	Sharpe Ratio
ML-selected portfolio	10.91%	3.03%	1.62
Naive Top-15 by 1-Year CAGR	33.13%	18.58%	1.46
Naive Same Equity/Debt Mix	9.78%	1.64%	2.31
Interpretation
The unrestricted top-performing funds generated the highest historical return but also carried substantially higher volatility, resulting in a lower risk-adjusted performance than the ML portfolio.
The ML-based portfolio achieved a higher Sharpe ratio than the unrestricted naive strategy, indicating more efficient return generation for the level of risk taken.
Although the "Naive Same Mix" portfolio produced a higher Sharpe ratio in this historical sample, its expected return (9.78%) was lower than the ML portfolio (10.91%). This suggests that the ML model prioritized higher-quality funds capable of delivering stronger expected returns while maintaining controlled risk.
Overall, the ML approach provides a more balanced and systematic investment process than simply selecting recent top performers, reducing dependence on short-term performance persistence.
Expense Ratio Strategy

Earlier exploratory data analysis showed only a weak relationship between expense ratio and long-term returns, indicating that higher management fees were not consistently associated with better performance.

Therefore, the proposed fund aims to maintain a relatively competitive expense ratio while emphasizing portfolio quality and risk-adjusted performance rather than assuming higher-cost funds will outperform.

Alternative Portfolio

A second allocation was generated for conservative investors.

Minimum Volatility Portfolio
Fund	Weight
Aditya Birla Sun Life Fixed Term Plan – Series TQ (Direct Growth)	49.77%
HDFC FMP 1269D March 2023 – Growth Option	49.79%

Expected Return: 7.41%

Expected Volatility: 1.60%

Sharpe Ratio: 0.88

This allocation prioritizes capital preservation and stable returns over maximizing expected performance.

Limitations
Portfolio optimization relies on historical 3-year return data and may not reflect future market conditions.
The study considers a relatively small universe of 74 funds.
Transaction costs, taxes, exit loads, and liquidity constraints are not incorporated.
Expected returns and covariance estimates are sensitive to the selected historical period.
The optimizer assumes stable statistical relationships, whereas real markets are dynamic and subject to structural changes.

Overall, the proposed ML-driven portfolio demonstrates how combining machine learning-based fund selection with Modern Portfolio Theory can produce a disciplined investment strategy that emphasizes superior risk-adjusted performance rather than relying solely on recent historical returns.
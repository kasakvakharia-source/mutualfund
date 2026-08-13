# Mutual Fund Intelligence

Project structure for a mutual fund analysis and recommendation workflow.
# Project Theory Reference

Everything below maps directly to a decision made somewhere in your project — this isn't
generic textbook material, it's the reasoning behind your specific choices. Good for
interview prep: if someone asks "why did you do X," the answer is here.

---

## 1. Financial Metrics — What You Actually Computed

### CAGR (Compound Annual Growth Rate)
```
CAGR = (End Value / Start Value)^(1 / years) - 1
```
The single annual growth rate that, compounded every year, gets you from the starting
NAV to the ending NAV. It smooths out year-to-year noise into one number — which is
also its weakness: a fund that fell 50% then rose 100% has the same CAGR as one that
grew steadily, but a very different experience for an investor. This is *why* CAGR
alone is a weak signal, and why you paired it with volatility and drawdown rather than
ranking funds by CAGR alone.

### Volatility (Annualized Standard Deviation of Returns)
```
daily_returns = pct_change(NAV)
annual_volatility = std(daily_returns) x sqrt(252)
```
Standard deviation measures how spread out daily returns are around their average.
Multiplying by sqrt(252) (trading days in a year) scales a daily measure up to an annual
one — this works because variance scales linearly with time, so standard deviation
(its square root) scales with the square root of time. This is a **risk** measure, not
a return measure — two funds can have identical CAGR with very different volatility,
and volatility is what tells them apart.

### Sharpe Ratio (Risk-Adjusted Return)
```
Sharpe = (CAGR - risk_free_rate) / annual_volatility
```
The central metric of the whole project. It answers: *how much return did this fund
earn per unit of risk taken?* A fund with 15% CAGR and high volatility can have a worse
Sharpe than a fund with 10% CAGR and low volatility — Sharpe is what let you rank funds
fairly across very different risk profiles, and it's the actual target your ML model
was trained to predict (as a category-relative rank, not a raw prediction).

**Why subtract the risk-free rate?** Because earning 8% when a risk-free government
bond also pays 6% only really "earned" you 2% of compensated risk-taking. The risk-free
rate is the return you'd get for *no* risk, so subtracting it isolates the reward for
actually being in the market.

### Maximum Drawdown
```
running_max = cummax(NAV)
drawdown = NAV / running_max - 1
max_drawdown = min(drawdown)
```
The worst peak-to-trough decline in the fund's history. This is the metric that
answers "what's the worst it's ever gotten?" — which matters to real investors far
more than volatility does, because volatility treats a good day and a bad day
symmetrically, while drawdown captures the specific pain of watching an investment
fall and stay down. This is why your COVID stress test used actual historical
drawdown behavior rather than volatility alone.

### Why Category-Relative, Not Absolute
Debt funds structurally have lower Sharpe ratios than Equity funds — comparing a Debt
fund's raw Sharpe to an Equity fund's raw Sharpe would make every Debt fund look bad
regardless of how well-managed it is. Computing the median Sharpe *within each category*
and asking "is this fund above or below its own category's median" is a fairness
correction — the statistical term for this is **relative ranking** or **peer-group
normalization**, and it's standard practice in real fund-rating methodologies
(Morningstar's star ratings work the same way, rated within category).

---

## 2. Modern Portfolio Theory — Designing the New Fund

### The Core Idea (Markowitz, 1952)
You don't have to accept a fund's average risk-return profile in isolation — when you
combine multiple assets whose returns don't move in perfect lockstep, the portfolio's
overall volatility can be *lower* than the volatility of any individual asset in it.
This is **diversification**, and it's not just "don't put all your eggs in one basket"
folk wisdom — it's a mathematical consequence of how variance combines across
correlated assets. Your Day 6 what-if allocation tool demonstrated this directly: mixing
funds dropped portfolio volatility below any single fund's volatility.

### Covariance Matrix
```
cov_matrix = daily_returns.cov() x 252
portfolio_volatility = sqrt(w^T . Sigma . w)
```
This is *why* diversification works mathematically. The covariance matrix (Sigma) captures
how every pair of funds' returns move together. When you combine funds with low or
negative covariance, their individual ups and downs partially cancel out at the
portfolio level — the weighted-average return stays the same, but the combined
volatility drops. This matrix, not the individual fund volatilities, is what your
optimizer actually optimized against.

### The Efficient Frontier
For any target return, there's one combination of assets that achieves it with the
*least* possible risk — and the curve connecting all these optimal combinations across
every possible return level is the efficient frontier. Any portfolio *below* the
frontier is inefficient (you could get the same return with less risk, or more return
with the same risk). Your Day 9 chart plotted this frontier and marked where your
proposed fund landed on it.

### Max Sharpe vs. Min Volatility Portfolios
Two different points on the same efficient frontier:
- **Max Sharpe** finds the portfolio with the best risk-adjusted return — the point
  where a line from the risk-free rate is *tangent* to the frontier (this is why it's
  sometimes called the "tangency portfolio")
- **Min Volatility** finds the single lowest-risk point on the frontier, regardless of
  return — the most conservative efficient option

You computed both so your new fund proposal could offer an aggressive and a
conservative variant, both still efficient (neither wastes risk for no reason).

### Covariance Shrinkage (Ledoit-Wolf)
With only ~15 candidate funds and a few years of daily data, a raw sample covariance
matrix is noisy — it overfits to coincidental co-movements that won't repeat. Shrinkage
blends the raw sample covariance toward a more structured, stable estimate (like a
scaled identity matrix), trading a small amount of bias for a meaningful reduction in
estimation error. This is *why* your optimizer script used
`CovarianceShrinkage(...).ledoit_wolf()` instead of the plain sample covariance — it's
the standard fix for exactly the small-sample problem your candidate pool has.

---

## 3. Machine Learning — The Scoring Model

### Why Classification, Not Regression
You framed the task as "is this fund above or below its category's median Sharpe"
(binary) rather than "predict the exact Sharpe ratio" (regression). With only 74 rows,
predicting an exact continuous number reliably is much harder than predicting which
side of a threshold a fund falls on — classification is a lower-variance, more
achievable target at this sample size, and it directly answers the practical question
("is this fund good or not") without pretending to more precision than the data
supports.

### Feature Leakage
A feature "leaks" when it's mathematically derived from (or nearly identical to) the
target you're predicting — the model doesn't learn a real pattern, it just rediscovers
the formula, and looks artificially perfect. Since Sharpe = (CAGR3y - risk_free) /
volatility, including `annual_volatility` or `cagr_3y` as *features* while predicting a
Sharpe-based target would be leakage — the model could reconstruct the label almost
exactly without learning anything generalizable. This is why those two columns were
deliberately excluded, even though they're genuinely informative about the fund.

### Ensemble Learning — Why Combine Models
Different model types make different *kinds* of mistakes:
- **Logistic Regression** — a linear model, fast and interpretable, but can only draw
  straight-line decision boundaries in feature space
- **Random Forest** — a **bagging** ensemble: trains many decision trees on random
  subsets of data and features, then averages them. Reduces *variance* (overfitting to
  noise in any one tree) without adding much bias.
- **XGBoost** — a **boosting** ensemble: trains trees sequentially, where each new tree
  focuses on correcting the previous trees' mistakes. Reduces *bias* (systematic
  underfitting) but can overfit if not regularized.

Combining models with different error patterns via **soft voting** (averaging their
predicted *probabilities*, not just their final yes/no votes) tends to cancel out each
model's individual mistakes — this is the actual mechanism behind why your ensemble
outperformed every individual model, not just an assumption that "more models = better."

### Cross-Validation — Why a Single Train/Test Split Wasn't Enough
With 74 rows, one 80/20 split leaves only ~15 rows in the test set — small enough that
which 15 rows happen to land in the test set can swing your accuracy score by 10+
percentage points purely by chance, telling you more about the lucky split than about
the model. **Repeated Stratified K-Fold** cross-validation splits the data into 5 folds
(stratified = each fold keeps the same proportion of each class), trains and tests 5
times rotating which fold is held out, then repeats that whole process 10 times with
different random splits — giving you 50 total evaluations to average, which is far
more stable than trusting any single split.

### Accuracy vs. AUC-ROC
- **Accuracy** — the percentage of correct predictions. Simple, but can be misleading
  if the model is only confident on easy cases.
- **AUC-ROC** — measures how well the model *ranks* funds by predicted probability of
  being top-half, across every possible decision threshold, not just a fixed 50% cutoff.
  An AUC of 0.92 means: if you randomly pick one top-half fund and one bottom-half fund,
  the model assigns the top-half fund a higher score 92% of the time. This is a more
  complete picture of the model's skill than accuracy alone, which is why both were
  reported.

### The Baseline (Dummy Classifier)
Always predicting the majority class got 47.3% accuracy — close to 50% because your
target was roughly balanced by construction (median split). This baseline exists so
"82% accuracy" has meaning: it says the model beat random/naive guessing by ~35 points,
not that 82% is some universally impressive number in isolation. **Any model result is
meaningless without a baseline to compare it to** — this is one of the most common gaps
in less rigorous ML projects.

---

## 4. AI Layer — What the LLM Integration Actually Is

### Prompt Engineering, Not Model Training
You didn't train or fine-tune any language model — you sent a pre-trained model
(Llama 3.1) a carefully structured prompt containing your *own* computed numbers, and
asked it to phrase them in plain English. The model's job was narrow: turn structured
data into fluent prose, not to independently "know" anything about mutual funds. This
is why the prompts explicitly said "only use the numbers provided" — constraining the
model to your data rather than letting it draw on (and potentially hallucinate from)
its general training knowledge.

### Temperature
```
temperature = 0.3
```
Controls how deterministic vs. varied the model's word choices are. Low temperature
(closer to 0) makes the model consistently pick its most likely next word — better for
factual, repeatable summaries. High temperature (closer to 1) introduces more
randomness — better for creative writing, worse for a tool that should describe the
same fund the same way each time. 0.3 was chosen to keep summaries factual and stable
while still reading naturally rather than robotically.

### Grounding (Why This Isn't Hallucination-Prone)
Every number in your prompts (`quality_score`, `sharpe_3y`, `category`) came directly
from your own pipeline — the model was never asked to recall or estimate a fund's
performance from its own training data, only to phrase numbers you already computed
and verified. This is the core discipline that prevents an LLM feature from becoming a
liability: the model narrates verified numbers, it doesn't generate them.

### Structured-Data Injection vs. Full RAG
"RAG" (Retrieval-Augmented Generation) usually means: search a large document
collection for relevant passages, then feed those passages to the model as context.
Your AI layer is a lightweight version of the same idea — instead of searching
unstructured documents, you looked up a specific fund's row in a structured table
(`fund_scores.csv`) and injected *that* into the prompt. Same underlying principle
(ground the model in retrieved, verified data rather than its own memory), simpler
implementation because your data is already structured and small enough not to need a
vector database or semantic search.

---

## 5. Data Handling Decisions

### Missingness as a Signal
Rather than silently filling missing `cagr_5y` values with the category median, you
also added a `cagr_5y_missing` flag column. A fund missing 5-year data is (almost
always) a newer fund — and "is this fund new" can itself be predictive, so encoding
*that a value was missing* preserves information that a silent fill would have erased.

### Median Split vs. Quartile Split
A top-quartile (75th percentile) split would have left very few "positive" examples
per category at this sample size (roughly 13 out of 52 Equity funds, 5 out of 22 Debt
funds) — too few for a classifier to learn from reliably. A median split guarantees a
roughly balanced 50/50 target by construction, trading some precision (top-half is a
less exclusive bar than top-quartile) for a target the model can actually learn well
at this sample size.

### Outlier Handling (The IDCW Bug)
This wasn't standard outlier trimming (like clipping the top/bottom 1% of values) — it
was root-cause diagnosis. Instead of blindly capping extreme volatility values, the
actual cause (dividend-payout NAV resets in IDCW plans) was identified and those rows
were excluded specifically, which is a materially stronger data-cleaning approach than
statistical outlier removal alone, because it fixes the *reason* the data was wrong
rather than just hiding the symptom.

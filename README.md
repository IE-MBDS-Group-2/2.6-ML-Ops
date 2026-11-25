# Portfolio optimization with reinforcement learning

> The firm should fund an MLOps pipeline because its pre-set, reproducible gate picks the agent, a cheaper rule or today's process, replacing one-off backtests.

## 1. Business case

### Context
Client: hypothetical firm (mid-sized independent wealth manager, about €2bn in client assets) in wealth management, Portfolio Management team (model portfolios desk). Basis: realistic stand-in. Firm size, quarterly rebalancing, ±5% drift rule, Investment Committee approval and human approval of every trade are assumptions, not confirmed facts.

Maturity: data MEDIUM, tools LOW, processes MEDIUM, people LOW.
- Data: daily ETF prices come from a market data vendor and holdings sit in the portfolio management system, but analysts keep their history in Excel exports.
- Tools: Excel and the portfolio management system only. No ML platform, model tracking or cloud analytics environment.
- Processes: quarterly calendar rebalancing plus a manual check when a weight drifts more than ±5% from target. Documented, run by hand.
- People: portfolio managers have no coding background. One quant analyst uses Python for reporting, with no ML engineering experience.

After delivery: the quant analyst runs the weekly batch and monitors the model. Portfolio managers review each recommendation and decide. The Investment Committee approves large changes. IT maintains the infrastructure.

### Problem and baseline
The Portfolio Management team struggles with avoidable client losses in market downturns because model portfolios are rebalanced on a fixed calendar with a wide drift band (cause is a hypothesis), so risk is cut late and some trades add only cost. Today the team rebalances quarterly, plus a check when any weight drifts more than ±5% from target. Large changes go to the monthly Investment Committee.

Balanced model portfolio, target weights (ASSUMPTION): SPY 40%, EFA 20%, IEF 30%, GLD 5%, BIL 5%. "60/40" names this mix: 60% equities (SPY, EFA), 40% defensive (IEF, GLD, BIL).

Baseline: the current rule (quarterly rebalance plus ±5% drift check, back to target, 0.1% trading cost) run on this portfolio. Max drawdown: To be measured, from a backtest on frozen daily prices, run and recorded before any model is trained.

### Goal and success metrics
Move the balanced portfolio's max drawdown from the baseline by a reduction set after the feasibility check (aspiration: at least 20% smaller, for example -20% to -16%) within 12 months of shadow operation alongside the current process.

| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | max drawdown | Today's rule: -18.7% (validation 2020 to 2021). Stage 2 model: -19.6%; `max_drawdown_reduction` -0.047, gate not met | Set after the feasibility check (aspiration: at least 20% smaller) |
| ML metric | `max_drawdown_reduction` (higher is better): 1 - agent max drawdown / baseline max drawdown, both as losses, so 0.20 means a 20% smaller worst loss. Quality gate, measured on validation 2020 to 2021 against today's rule | 0 (today's rule) | `>=` 0.20 |
| Guard | annual net return | To be measured | No more than 0.25 percentage points below baseline |
| Guard | annual trading cost | To be measured | No more than 2 x baseline |
| Guard | `tracking_error` to target weights (annualised) | To be measured | No more than X, set from the baseline on 2020 to 2021 |
| Challenger | `max_drawdown` vs selected simple rule | To be measured | Smaller than the selected rule's |

Stage 2: `main.py` checks only the quality gate (`max_drawdown_reduction >= 0.20` on validation 2020-2021); guards are reported, not gated. The release decision on 2022-2025, including the guards, runs once, in Stage 3.

Do not release if, on 2022 to 2025, any of these holds: max drawdown misses the KPI target; annual net return trails the baseline by more than 0.25 percentage points; annual trading cost exceeds 2 x the baseline's; tracking error exceeds X; or max drawdown is not smaller than the selected simple rule's. Then recommend the selected simple rule if it passes the gate against the baseline, otherwise keep the current process.

Reported, not gated: Sharpe ratio (headline), Sortino ratio, volatility, CVaR (expected shortfall) at 95%, turnover, tracking error. Calmar ratio may be shown for reference.

Why tracking error is a guard: rebalancing exists to keep the portfolio close to the allocation the Investment Committee approved. An agent that cuts drawdown by letting the portfolio drift away from target would take it outside its mandate.

Limits (ESTIMATES): weekly decision run under 60 s; compute cost under €0.01 per decision.

### Solution
Every Friday after the close, the portfolio manager receives a one-line recommendation for the balanced portfolio: "do nothing", "rebalance halfway" or "rebalance fully back to target". It shows the current drift, the estimated trading cost and a confidence level, to decide whether to rebalance next week.

The model predicts which of three actions (0%, 50% or 100% of the way back to target) gives the best next-week result after costs. Inputs, all known at Friday's close: current weights and drift per ETF, drop since the portfolio's last high, 4-week and 12-week returns, 20-day volatility per ETF, weeks since last rebalance, estimated cost of a full rebalance. Weekly batch. Target weights stay fixed, so the agent only decides when and how much to rebalance.

Timing rule: decision at the close of Friday t, trade at the close of the next trading day e. The new weights earn returns from the day after e to the next execution day, inclusive. The agent and the baseline share one execution function.

Fallback: if the model is unavailable or unsure, the current process applies unchanged. "Unsure" means the preferred action has less than 60% probability (ESTIMATE), or any input is missing or stale. The portfolio manager approves every trade. A large change (total buys plus sells above 10% of portfolio value in one week, ESTIMATE) also goes to the monthly Investment Committee.

### Data
Source: Yahoo Finance daily prices for SPY, EFA, IEF, GLD and BIL via `yfinance` (listed in `environment.yml`; versions frozen in Stage 3). Yahoo owns the data. Ricardo Arciniegas owns the frozen snapshot. Size: about 4,700 trading days x 5 ETFs, May 2007 to December 2025, with the first ~60 trading days used only to warm up the inputs. First decision date: 2007-09-07. Split by time: train 2007-09-07 to 2019-12-31, validation 2020 to 2021, test 2022 to 2025. Starting in 2007 puts the 2008 crash in the training data.

Label: `best_action`, one of 0.0, 0.5 or 1.0 (the share of the way back to target), the action with the best reward for the coming week. The reward is the next window's portfolio return minus 0.1% of the value traded, minus a penalty for new drops below the previous high. The Stage 2 model is a classifier that predicts this label; the reinforcement learning agent described above is a Stage 3 challenger. Refresh: weekly in operation. For the project, one frozen snapshot, shared directly with the team and not committed to Git (yfinance's documentation notes the Yahoo Finance API is intended for personal use only). Its SHA-256 checksum is recorded in `TEAM_BRIEF.md`. Training never downloads data.

Personal or sensitive data: NO. Public prices only. The client and portfolio are hypothetical. Yahoo Finance is a stand-in for this course project. Operation at a firm would require a licensed market data source.

Tests: every input date is on or before the decision day; the first reward date is after the execution date; reward windows join up with no gaps or overlaps; the agent and baseline use the same execution function.

### Value, cost and risks
Benefit over a non-AI approach: smaller losses in downturns, with net return no more than 0.25 points below the baseline per year, trading cost no more than 2 x the baseline's, and tracking error within the guard. Measured by max drawdown, Sharpe and tracking error against the baseline and the best simple rule on 2022 to 2025.

Simple-rule challengers, checked weekly on Friday. The first four go fully back to target: current rule (quarterly, ±5%); band 2.5 points; band 1.5 points; persistent drift (at least 2 ETFs beyond 2.5 points on each of the last 5 trading days). The last two do not:
- Asymmetric rule: rebalance fully as soon as drift or drawdown passes a threshold, then move back toward target only gradually, with a minimum wait between steps (threshold and wait to be defined, ESTIMATES).
- Matched-exposure fixed mix: constant weights holding the agent's own average equity share over the same dates, rebalanced quarterly. The agent cannot pass just by holding fewer stocks.

The best is selected on 2020 to 2021 by max drawdown, among rules that pass the same guards. Parameters are the team's own assumptions, inspired by public descriptions checked on 9 Nov 2026: Indexa's drift bands of 1.5 and 1.25 percentage points as reported by Finect (secondary source), and Finizens' multi-asset, multi-day trigger as described by Finect on 27 Oct 2026 (company-reported, no numeric parameters given).

Benchmarks (reported, not part of the gate): traditional portfolio optimisation portfolios (mean-variance, maximum Sharpe, minimum variance, risk parity), fitted on 2007 to 2019 only, run through the same execution function and costs.

Robustness checks:
- Holm correction across the challenger comparisons. Log how many agent versions were tried before the final one.
- Cost sensitivity: rerun with higher costs, with exits costing more than entries plus a market impact term that grows with trade size.
- Show the 2022 to 2025 result again with the largest drawdown episode removed. A pass that depends on one episode is reported as fragile.
- A 12-month shadow period with no market drawdown is reported as "inconclusive", not "passed".
- Optional input: a public market-stress indicator (VIX vs VIX3M, or recent volatility vs its normal level), used as a model input or to trigger the fallback. It also gives the drift monitor alert a concrete number.

Rough cost (ALL ESTIMATES): course build by the 4 to 5 person team, part-time, about 6 weeks. At a firm: 1 ML engineer (half-time, 12 weeks) plus the quant analyst (a quarter of their time), then about a tenth of one person's time to run and monitor. Infrastructure: laptop CPU for training, GitHub Actions, W&B, a small self-hosted container, under €50 per month.

| Risk | Mitigation |
| --- | --- |
| Data quality: Yahoo errors (missing days, price spikes, adjustments that change on re-download) | Validation tests on every snapshot (no missing trading days, no extreme daily moves, all 5 tickers present); frozen DVC snapshot with checksum |
| Model error: overfitting, since about 640 training decisions (ESTIMATE) is little for RL | Median of 5 seeds; feasibility ceiling check before training; simple-rule challengers; small model with only 3 actions |
| Model error: regime change (for example 2022, when stocks and bonds fell together) | Monitor input drift and alert above a set limit (limit to be defined); fall back to the current rule when inputs are out of range |
| Operational: Friday batch fails or data arrives late | Automatic fallback to the current rule; alert to the quant analyst; no recommendation without fresh data |
| Operational: portfolio managers trust the agent blindly | Human approval on every trade; log every recommendation, decision and outcome; monthly review of agreement vs overrides |
| Operational: a new model version performs worse | Model registry with versions; one-step rollback to the previous approved version, or to the current rule if no approved version exists |

Scales to (ideas, not findings): the firm's conservative and growth model portfolios, UCITS ETF versions, indexed pension plans, other rule-based rebalancing jobs.

### Open gaps
1. Baseline max drawdown, return, trading cost and tracking error are To be measured over 2007 to 2021 (feasibility check and challenger selection) and 2022 to 2025, before any model is trained.
2. Define `tracking_error` in this README and in `config.yaml` (proposed: annualised standard deviation of the difference between portfolio and target-portfolio weekly returns) and set X from the baseline on 2020 to 2021.
3. Not stated: which weights each action moves toward target (Friday or Monday pre-trade), and which weights the Friday cost estimate uses.
4. Not stated: starting weights on the first decision date and at the start of 2022, and the scope of the "every day in exactly one window" test.
5. `config.yaml` needs fixed meanings for: "±5%" and "±2.5" (percentage points proposed), the quarterly date, daily weights for the persistent rule, simple or compounded weekly reward, the drawdown penalty reference, and a minimum trade size (Finizens uses one).
6. "Confidence" is an action probability, not a calibrated confidence. The 60% and 10% thresholds are estimates. The weekly vs monthly committee cadence is undefined for large changes.
7. Before any training: a timing-only agent with perfect foresight on 2007 to 2021 sets the ceiling for the drawdown reduction. The KPI target is set from it. Rough estimate: timing alone may not reach 20%, because the agent cannot hold fewer stocks than drift already gives it, and in 2022 stocks and bonds fell together so drift was small.
8. Evidence limits: 2020 to 2021 includes the COVID crash, and there are only about 950 weekly decisions in total (ESTIMATE). Not yet in the risk table.
9. "Wide drift band" as the cause is a hypothesis. The band research is not done.
10. The snapshot owner is named (Ricardo Arciniegas). The personal-use claim comes from yfinance's documentation, not from Yahoo's own terms. Still open: name who read Yahoo's terms, with the date.
11. Check the `yfinance` notes for the pinned version (the cited pages were for older versions), confirm the tickers. Confirmed: all five ETFs have Yahoo data from 2007-05-30 (BIL is the latest). Verified by the snapshot download on 11 Nov 2026: 4,679 trading days per ticker, no gaps, no missing values.
12. The drift monitor alert limit has no number. The €0.01 cost limit is compute only. The scaling claims are ideas.
13. Challengers check weekly, whereas Indexa monitors daily. The project name is not chosen.
14. The label is only ever 0.0 (about 55% of decisions) or 1.0 (about 45%), never 0.5, because the reward is nearly linear in the action. Decide whether to keep three actions.

### Stage 3 update
Added in Stage 3. The business case above is unchanged, as submitted. This part says what the live service changes for the desk, what it costs to run, and what became of each commitment. Results come from one run of `python -m src.main` on the official snapshot before the v1.1.0 release (`dev` at `cc40b45` plus #125), with the shipped settings; the RL agent and the grid search are reported from the same run with `rl.enabled` and `selection.enabled` switched on. Validation is 2020 to 2021.

**What changes for the desk.** Before, the Friday batch wrote a CSV. Now the portfolio manager sends their positions (ticker and market value), the portfolio's drawdown, the weeks since the last rebalance and recent prices to a web address. They get back the recommendation (0.0, 0.5 or 1.0), a proposed portfolio, a trades list and one plain sentence saying why. They make two choices, in their own words: the **target weights** (their own policy mix, or a calculated one: minimum variance, maximum Sharpe, mean-variance or risk parity) and the **rebalancing method** (smart, where our model decides; calendar; tolerance band; or calendar and band). The review frequency (daily, weekly or monthly) is a setting, weekly by default. The portfolio manager still approves every trade.

**What it changes in the decisions today: nothing.** No model has passed the gate, so the service answers with today's rule (quarterly, plus the 5-point drift check) and says so in its reply. The desk gets the same answer as before, with a ready trades list instead of a CSV.

**Cost of running it.** GitHub Actions on GitHub-hosted runners (the repository is public), a Weights & Biases team account for tracking and the model registry, the GitHub container registry for the image, and a self-hosted always-on machine behind a Cloudflare tunnel for the service. No paid component is used. The estimate in "Value, cost and risks" (under €50 per month for infrastructure) is unchanged. Time per request, measured locally in rule-only mode: 41 ms, with about 190 MB of memory, against the limits above (under 60 s and under €0.01 per decision); the latency on the host: PENDING, measured at the v1.1.0 release.

**What became of each commitment** (status on 18 Nov 2026):

| Commitment in this section | Where it lives | Result |
| --- | --- | --- |
| Quality gate, `max_drawdown_reduction >= 0.20` on validation 2020 to 2021 | `evaluate.check_gate`, run by `main.py` | Stage 2 classifier: -0.047 (-19.6% against -18.7%). **Not met.** Nothing is saved or shipped |
| Reinforcement learning agent through the same gate, median of 5 seeds | `rl_agent.py`, run by `main.py` (#102) | Median of 5 seeds: +0.019 (seeds 0.019, 0.019, 0.019, 0.024, 0.019). **Not met.** The 60% fallback would hand the decision to today's rule on 22% of validation weeks |
| Simple-rule challengers: current rule, bands of 2.5 and 1.5 points, persistent drift, asymmetric rule, matched-exposure mix; the best is selected on validation | `rules.py`, reported by `main.py` | Persistent drift selected: -18.40% max drawdown and 14.34% annual return against today's rule's -18.74% and 13.72%. The two bands and matched exposure fail the return guard |
| Traditional benchmarks (mean-variance, maximum Sharpe, minimum variance, risk parity), reported and never gated | `benchmarks.py`, reported by `main.py` | Fitted on the train years with a 60% asset cap, covariance shrinkage and zero cash weight. Minimum variance: 25.3% SPY, 60.0% IEF, 14.7% GLD (max drawdown -8.3%); maximum Sharpe: 27.5% SPY, 60.0% IEF, 12.5% GLD (max drawdown -8.5%); mean-variance: 50.3% SPY, 27.8% IEF, 22.0% GLD (max drawdown -16.7%); risk parity: 14.8% SPY, 11.2% EFA, 59.1% IEF, 14.9% GLD (max drawdown -8.9%). These allocations change the policy mix, so they are reported separately from the gate. |
| Guards: annual net return, annual trading cost, tracking error | `evaluate.guard_report`, reported by `main.py` | The tracking-error limit is 0.012, twice today's rule (0.0059). Classifier: return shortfall 0.0063 against a limit of 0.0025 (**failed**), cost ratio 1.12 against 2.0 (passed), tracking error 0.0025 against 0.012 (passed) |
| Release decision on 2022 to 2025, run once | `evaluate.release_decision`, the test audit in `main.py` | **Not run:** no model passed the gate, so the test years stay sealed |
| Robustness checks: Holm correction, cost sensitivity, the largest drawdown episode removed | `robustness.py`, reported by `main.py` | Feasibility ceiling: a perfect-hindsight agent on 2007 to 2021 reduces max drawdown by only 0.0044, so **the 0.20 gate cannot be reached with these three actions**. Cost sensitivity (classifier): -0.047 with flat costs, -0.047 with realistic costs. One version scored on validation with the shipped settings (six with the RL seeds). Holm correction and episode removal run only in the test audit, which stays sealed |
| Walk-forward check and a small grid search on the train years, every try logged | `selection.py`; `selection` in `config.yaml` (off by default) | Seven yearly blocks 2013 to 2019 (table below): median -0.0013. Grid of six trials: best C = 1 without class weights (median -0.0010), -0.031 on validation, gate not met. Review frequency net of costs: daily -0.0094, weekly -0.0010, monthly +0.0015 |
| Drift monitor alert | `monitoring.py`, `scripts/monitor_report.py` (#122) | Largest PSI of SPY, EFA, IEF and GLD daily returns (last 180 days against the train years) above 0.2. On the official snapshot it fires for the week of 26 Dec 2025 (EFA 0.254) |
| 12 months of shadow operation | operation at the firm | **Not done.** It needs a year of running alongside the current process, which is outside the course |
| Optional market-stress input (VIX) | none | **Not built** |

**What became of each open gap** (status on 18 Nov 2026, checked against `dev`). The list above is the submitted text and is not edited.

| Gap | Status |
| --- | --- |
| 1. Baseline measures | **Closed.** Today's rule on 2020 to 2021: max drawdown -18.74%, annual return 13.72%, volatility 13.81%, Sharpe 0.99, trading cost 0.025% a year, tracking error 0.0059 |
| 2. Tracking error | **Closed** (#64): the annualised standard deviation of the difference between portfolio and target-portfolio weekly returns (`evaluate.tracking_error`); X = 0.012 (`guards.max_tracking_error`), twice today's rule on 2020 to 2021 (0.0059) |
| 3. Which weights the actions and costs use | Decision timing **closed**: the state uses the pre-trade Friday weights, and the trade happens at the next trading day's close (`docs/MODULE_SPECS.md`, portfolio). The weights behind the cost estimate are not restated there |
| 4. Starting weights | The simulation starts at target weights on the first decision date (2007-09-07). The start of 2022 and the "every day in one window" test are not checked here |
| 5. Meanings in `config.yaml` | Mostly **closed**: "±5%" is `drift_band: 0.05` in absolute points, the quarterly date is the first Friday of January, April, July and October, `penalty_lambda` is 1.0, `min_trade` is 0.0. Not restated: daily weights for the persistent rule and simple or compounded reward |
| 6. "Confidence" | **Open**, stated as a known limit in the model card |
| 7. Feasibility ceiling | **Closed** (#101): perfect hindsight on 2007 to 2021 reduces max drawdown by 0.0044, far below the 0.20 gate. That window differs from validation, so it is not directly comparable with the validation results |
| 8. Evidence limits | **Open**, stated as a known limit in the model card |
| 9. Wide drift band as the cause | **Open**: the band research is not done |
| 10. Yahoo's terms | **Open**: the snapshot owner is named, who read the terms and when is not |
| 11. `yfinance` and tickers | Tickers and data **confirmed** on 11 Nov. The notes for the pinned `yfinance` version are not rechecked |
| 12. Drift alert limit | **Closed** (#122): PSI above 0.2, see Monitoring |
| 13. Daily or weekly challengers; project name | **Open** |
| 14. Three actions or two | **Answered:** on the train years (643 weeks) the best action is no trade 55.4% and full rebalance 44.6%; the half step is never best (0.0%). Validation (104 weeks): 51.0% / 49.0% / 0.0% |

---

# 2.6-ML-Ops

## Project
Weekly rebalancing advice for a balanced 5-ETF portfolio (SPY, EFA, IEF, GLD, BIL). A classifier predicts the best action each Friday: move 0%, 50% or 100% of the way back to the target weights. If the model is unsure, today's rule applies instead.

The Stage 2 quality gate is `max_drawdown_reduction >= 0.20` on validation (2020 to 2021). On the current data the gate is not met (see section 1).

## Model card

### Purpose and users
At each review (weekly by default) the model recommends one of three actions for the balanced 5-ETF portfolio: leave it (0.0), rebalance halfway back to target (0.5) or rebalance fully (1.0). Portfolio managers decide and approve every trade. The quant analyst runs the weekly batch and watches the model. The Investment Committee approves large changes. These roles are assumptions of the business case.

The model only decides *when and how much* to return to targets that the fund sets. It does not choose the targets or the assets, and it is not tactical allocation. It is not meant for other tickers or portfolios, for use without human approval, or for real money: the client is hypothetical.

### Data
| Item | Value |
| --- | --- |
| Source | Yahoo Finance daily prices (close and volume) for SPY, EFA, IEF, GLD and BIL, through `yfinance`. The yfinance documentation says the Yahoo Finance API is for personal use only, so a real firm would need a licensed source |
| Snapshot | 4,679 trading days per ticker, 2007-05-30 to 2025-12-31, downloaded on 11 Nov 2026. Shared directly with the team, never committed to Git |
| Raw file checksum | `data/raw/prices.csv`: SHA-256 `3cce101c20eb1bcd5f453176affaa3c7f4a3f470a89973de47617134d6705526` |
| Inference file checksum | `data/inference/prices_latest.csv`, the last 300 trading days of the same snapshot (2024-10-21 to 2025-12-31): SHA-256 `b5adf3d40afbf760afcb1527c07821064099a7587573b67a8d5ee3f02515e0f3` |
| Split | By date. Train 2007-09-07 to 2019-12-31 (includes the 2008 crash), validation 2020 to 2021, test 2022 to 2025 (sealed: read once, at the release decision) |
| Checks on every run | All five tickers present, the same dates for every ticker, close above 0, no missing values, no daily move above 25%. Any failure stops the run |

A new download gives slightly different prices and so a different checksum. Use the shared file and do not re-run `scripts/fetch_prices.py` unless you mean to make a new snapshot.

### Model and features
- **Default model** (`model.type: classifier`): a multinomial logistic regression with balanced class weights and a fixed seed (42). It is fast, repeatable and easy to explain to a portfolio manager. Balanced weights stop it defaulting to the most common action.
- **Challenger** (`model.type: fitted_q`): a reinforcement learning agent (fitted Q iteration, scikit-learn only). It is meant to go through the same gate as the classifier. `main.py` runs it when `model.type` is `fitted_q`, or reports it next to the classifier when `rl.enabled` is true (#102). The shipped setting is `classifier`.
- **Inputs**, all known at the close of the review day: for each ETF the current weight and its drift from target, plus the 4-week and 12-week return and the 20-day volatility; for the portfolio the drop since its last high, the weeks since the last rebalance and the estimated cost of a full rebalance.
- **What it learns**: the action with the best reward for the coming window. Reward = the window's portfolio return, minus 0.1% of the value traded, minus a penalty for new drops below the previous high.
- **Fallback**: if the model's probability for its preferred action is below 60%, or an input is missing, today's rule applies instead.

### Calculated allocation targets

The four calculated targets are fitted at every retrain on prices through `split.train_end`.
Minimum variance, maximum Sharpe, mean-variance and risk parity share the benchmark optimizer,
Ledoit-Wolf covariance shrinkage and a configurable cap of 60% per risky asset.
The BIL cash ticker receives zero weight. Maximum Sharpe subtracts BIL returns as the risk-free rate.
Changing the cap changes the investment mandate and requires review.

The API and pipeline read the reviewed weights in `target_weights.calculated`.
For a calculated target, `smart` serving falls back to calendar-and-band and explains the reason.
The API reports the training-period tracking-error estimate stored with the fitted weights.
The estimate stays fixed across requests until the next reviewed refit.
Models evaluated with calculated targets are research candidates. Serving calculated targets uses the
rule even if a model is present. The existing policy target keeps its normal smart behavior.
During grid search, calculated targets are refitted inside each training window before its scoring
block so later training prices cannot influence earlier allocation decisions.

Run `python -m scripts.fit_targets` at every retrain and submit the updated configuration for review.
The generated block records the snapshot checksum, return dates, training cut-off, optimizer settings
and training-period tracking errors as machine-readable data. Config loading rejects stale split/settings
metadata, and calculated-target training verifies the actual snapshot checksum before it starts.

Every saved model includes its training target method and weights inside its joblib artifact and W&B
metadata. The API uses a model only when those values match the configured target. Older artifacts
without target metadata fall back to the rule until retrained.

### The gate and the honest result
A model is only saved, registered and served if `max_drawdown_reduction >= 0.20` on validation (2020 to 2021). That is a worst loss at least 20% smaller than today's rule (quarterly rebalance plus a 5-point drift check).

| | Max drawdown, validation 2020 to 2021 |
| --- | --- |
| Today's rule | -18.7% |
| Stage 2 classifier | -19.6% |
| `max_drawdown_reduction` | **-0.047, gate NOT met** (needs 0.20 or more) |

So the Stage 2 classifier is not saved, not registered and not shipped. Until a model passes, the live service answers with today's rule and says so in its reply. We report this result as it is and do not tune the model on the validation years until it passes. Stage 3 adds the reinforcement learning agent, simple-rule challengers and benchmark portfolios, and a release decision on 2022 to 2025 that runs once.

### Why the label serves the gate
The gate asks for one thing: on validation (2020 to 2021) the model's maximum drawdown must be at least 20% smaller than today's rule (`max_drawdown_reduction >= 0.20`). The label is chosen to push in that direction, and it is worth saying how far it can.

- **The link.** Maximum drawdown is the deepest fall from a running peak. A week can only deepen it by ending below the previous high, that is, by making a new low. The reward charges exactly that: it subtracts `penalty_lambda` times the new drop below the previous high. So the label prefers the actions that avoid the steps that build the worst trough.
- **The gap.** The label is picked one week at a time, while maximum drawdown belongs to the whole path. A week that makes no new low gives no drawdown signal at all, and the label there is decided by return and cost alone. The label is therefore a proxy for the gate's measure, not the measure itself.
- **How the proxy was tested.** First, the perfect-hindsight ceiling: an agent that knows the future reduces maximum drawdown by only 0.0044 on 2007 to 2021, so no model that only chooses among the three rebalancing actions can reach 0.20. The limit is the action space, not the model. Second, the reinforcement learning agent, which optimises several weeks at once instead of one: median of 5 seeds +0.019 on validation, best seed +0.024. Third, the gate itself blocks every model, so today's rule serves (decision D8). That is the design working as intended, and we report it as it is.
- **What would close the gap.** Actions that can change the mix itself (dynamic weights within a mandate band), and a gate chosen on the training years instead of fixed in advance. More than three actions is the first step under "A stronger agent" in [Future work](#future-work); the other two come after it.

The metric (`max_drawdown_reduction`), the threshold (0.20) and the direction (at least, on validation) are the same here, in `TEAM_BRIEF.md` and in `config.yaml` (`quality_gate`).

### Walk-forward results per year
Train years only, shipped settings; each year scored by a model trained on the years before it.

| Year | Today's rule max drawdown | Model max drawdown | `max_drawdown_reduction` |
| --- | --- | --- | --- |
| 2013 | -6.11% | -6.07% | 0.0058 |
| 2014 | -4.50% | -4.54% | -0.0092 |
| 2015 | -7.63% | -7.49% | 0.0193 |
| 2016 | -3.38% | -3.41% | -0.0079 |
| 2017 | -1.42% | -1.42% | -0.0001 |
| 2018 | -10.60% | -10.96% | -0.0334 |
| 2019 | -2.68% | -2.69% | -0.0013 |
| Median | | | -0.0013 |

### Known limits
- **Little evidence.** Validation contains the COVID crash, one stress episode, and there are only about 950 weekly decisions in total (estimate).
- **Flat costs.** Trading costs are a flat 0.1% of the value traded. Commission, slippage and market impact are a sensitivity check, not the default.
- **The half step never wins.** On the train years the best action is no trade 55.4% and full rebalance 44.6% of the time, and the half step 0.0%; on validation 51.0%, 49.0% and 0.0%.
- **"Confidence" is a probability**, the model's probability for its preferred action. It is not a calibrated confidence.
- **Narrow scope.** Five ETFs, long only, fixed target weights. Positions in any other ticker are rejected.
- **Hypothetical client, public prices.** Nothing here has been tested with real money or real client data.
- **Small trades.** `portfolio.min_trade` is 0.0 today, so every trade is kept and buys and sells net to zero. If it is raised, trades below it are dropped and the remaining buys and sells may no longer net exactly to zero.
- **Validation is not the release decision.** A model that passes the gate can still fail the guards on 2022 to 2025 (return, cost, tracking error, the best simple rule).

### What triggers a retrain
One list, used by the monitoring (section "Monitoring") and by the people who run the service. Retrain, and run the quality gate again, when any of these happens:
- **Monitoring:** the fallback rate is above 50% for 4 weeks in a row (the model is rarely confident enough to override today's rule), measured from the predictions the API logs to Weights & Biases; or the drawdown reduction against today's rule is below 0 over a quarter (the model does worse than the rule it replaces), measured from the realised portfolio values in the monthly review. The API logs no portfolio values, so it cannot measure this one.
- **Data:** a new price snapshot is adopted (a new checksum), or the weekly data check fails.
- **Inputs:** the input drift alert fires. Proposed: it has no numeric limit yet (open gap 12 in section 1).
- **Use:** the monthly review shows the portfolio managers overriding the recommendation much more than they follow it.
- **Settings:** the target weights or the rebalancing method change, because the labels depend on them.

A retrained model must pass the gate again before it takes the `prod` alias. If a new version performs worse, roll back to the previous approved version, or to today's rule when there is none.

### Who owns it
At the firm: the quant analyst runs and monitors it, portfolio managers decide, IT keeps the infrastructure running. In this project: the module owners in the table at the end of this file. Ricardo Arciniegas owns the data snapshot and Miguel Palanca owns the release.

## Install
```bash
conda env create -f environment.yml
conda activate mlops-group2
```
This works on macOS, Windows and Linux. CI, Docker and the live service use the exact versions in `conda-lock.yml`. That file is made for Linux only (`linux-64`), so on a Mac or a Windows PC use the command above, or run the Docker image below. On Linux: `conda-lock install -n mlops-group2 conda-lock.yml`.

## Get the data
The pipeline never downloads data. Use the `prices.csv` that Ricardo shares with the team. It is not in Git, because Yahoo's API is for personal use only.

1. Create the folders (Git does not keep empty folders): `mkdir -p data/raw data/inference` (Windows: `mkdir data\raw data\inference`). Put the file at `data/raw/prices.csv`.
2. Check its SHA-256 against the Data table in the Model card above. On Windows: `certutil -hashfile data\raw\prices.csv SHA256`. On Mac or Linux: `shasum -a 256 data/raw/prices.csv`.
3. The run also reads `data/inference/prices_latest.csv`, the last 300 trading days of the same snapshot. Ask Ricardo for it too, and check its SHA-256 in the same table.

Run `python scripts/fetch_prices.py` only to make a new snapshot. Downloads differ over time, so a new file has a different checksum and gives different results.

## Refit calculated allocation targets

At each retrain, run `python -m scripts.fit_targets` against the frozen training snapshot.
The command fits minimum variance, maximum Sharpe, mean-variance and risk parity with the
same optimizer as the benchmark report. It uses prices through `split.train_end` only.
Covariance shrinkage stabilizes the estimates, `benchmarks.max_weight` caps each asset,
and the cash ticker has zero weight in the four risky allocations. Maximum Sharpe uses
cash returns as its risk-free return.

The command updates `target_weights.calculated` in `config.yaml` and records the source
checksum, return dates and machine-readable fit provenance. Submit the changes through a reviewed PR when
retraining so the API reads the same versioned weights.
No access to the snapshot? Make sample data instead: `python -m scripts.make_sample_prices`. The prices are random, every run logs that they are SYNTHETIC, and CI uses them. The script refuses to overwrite a file that already exists unless you pass `--force`, so it cannot replace the real snapshot by accident. Results on sample data mean nothing.

## Run
```bash
python -m src.main
```
One command runs every step and writes these files. Git ignores them.

| File | What it is |
| --- | --- |
| `data/processed/clean.csv` | cleaned prices |
| `models/model.joblib` | the fitted model |
| `reports/predictions.csv` | one recommendation per decision date, with a fallback flag |
| `reports/challengers.csv` | one row per simple rule over the validation years: max drawdown, return, cost, tracking error and guard results |
| `reports/benchmarks.csv` | the four traditional portfolios over the validation years: weights, max drawdown, return, volatility, Sharpe, cost (reported, never gated) |
| `reports/metrics.json` | validation scores, model against today's rule, the gate result, the guards, the chosen challenger and the benchmarks |
| `reports/selection.csv` | parameter trials and median training-period walk-forward scores when `selection.enabled` is true |
| `reports/frequency_comparison.csv` | daily, weekly and monthly review compared on the train years, net of costs, when `selection.enabled` is true |
| `reports/robustness.json` | historical hindsight diagnostic, validation under flat and realistic costs, number of versions scored on validation, and test-audit status |
| `reports/proposed_portfolio.csv`, `reports/trades.csv` | only when `data/inference/positions.csv` exists: the recommendation for that book, its proposed portfolio and the trades |

The robustness report records episode exclusion and Holm-adjusted challenger comparisons when the model passes the validation gate and reaches the test audit. Otherwise the report marks the test period as sealed. With W&B tracking active, the run also logs a robustness table.

The log (`logs/pipeline.log`) says "Quality gate met" or "NOT met". If the data or config is wrong, the run stops with exit code 1 and says what to fix.

The predictions use the inference file, which ends in 2025. They are a demo of the output, not out-of-sample evidence.

## Serve the API
The service has two endpoints: `GET /health` (is it up, which model and release) and `POST /predict` (a portfolio's positions and recent prices in; a recommendation, the reason, a proposed portfolio and a trades list out).

```bash
cp .env.example .env     # first time only (Windows: copy .env.example .env)
uvicorn src.api:app      # from the repository root, with mlops-group2 active
```
Then open `http://127.0.0.1:8000/docs` for the interactive page, where you can try both endpoints.

**Settings** live in `.env` and never go in Git. Running locally needs no key.

| Name | Meaning |
| --- | --- |
| `MODEL_SOURCE` | `local` reads `models/model.joblib`; `wandb` downloads the model with the alias `prod` |
| `WANDB_MODE` | `online`, `offline` or `disabled` |
| `WANDB_API_KEY` | your **own** key, only for `wandb` or `online` (team `ie-mbds-group-2`, ask Miguel for an invite) |
| `WANDB_ENTITY`, `WANDB_MODEL_ALIAS`, `APP_VERSION` | the W&B team, the model alias (`prod`) and the release name shown by `/health` |
| `APP_COMMIT` | the deployed commit, shown by `/health` as `commit`. The release image has it baked in; leave it as `local` on your machine |

**No model yet?** No model has passed the gate, so none is saved. The service still starts and answers with today's rule, with `fallback: true` and the reason in plain words.

**Call it with `curl`**, using the sample request (`scripts/sample_request.json`, built from sample prices):
```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @scripts/sample_request.json
```

| You send (`/predict`) | |
| --- | --- |
| `as_of` | the review date |
| `positions` | a list of `{ticker, market_value}`; only the configured tickers, values 0 or more, total above 0 |
| `drawdown` | 0 at the portfolio's peak, negative below it |
| `weeks_since_rebalance` | reviews since the last rebalance, 0 or more |
| `prices` | at least 61 trading days of the five ETFs (`api.min_history_days`) |
| `target_method`, `rebalancing_method`, `portfolio_id` | optional; the defaults are the policy mix and our model (`config.yaml`) |

You get back `recommendation` (0.0, 0.5 or 1.0), `fallback` (true when today's rule decided), `reason`, `proposed_portfolio` (ticker, weight, value), `trades` (ticker, side, value, weight before and after) and `model_version`. A field that breaks the request gives HTTP 422 with the allowed values. Data the pipeline cannot use gives HTTP 400 with the pipeline's own message. Prices and positions are never logged.

Check a running service end to end (works on a local run, a Docker run or the live URL):
```bash
python scripts/smoke_test.py --url http://127.0.0.1:8000
```
It exits with 0 when `/health` is ok and `/predict` answers the sample request.

**Live service:** https://etf-api.mpholdings.co (`/health`, `/predict`, `/docs`). Runs, data versions and models are in Weights & Biases: `https://wandb.ai/ie-mbds-group-2/etf-rebalancing`.

## Test
```bash
pytest --cov=src --cov-branch
flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests
```
GitHub Actions runs both on every pull request and requires at least 90% coverage. Keep code lines at 100 characters or fewer.

## Run in Docker
The API runs in one image with the exact package versions from `conda-lock.yml`. Install Docker Desktop first.
```bash
docker build -t etf-advisor .
docker run --rm -p 8000:8000 -e MODEL_SOURCE=local etf-advisor
```
Then open `http://localhost:8000/health` (or `/docs`). Stop it with Ctrl+C.

- **Contents:** only `config.yaml`, `conda-lock.yml` and `src/` (`.dockerignore` is an allow-list). `.env`, `data/`, `models/` and `tests/` never enter the image. Pass settings as environment variables (`-e NAME=value`), as the host does.
- **Platform:** the image is built for `linux/amd64`, the platform the lock file and the host use. On an Apple Silicon Mac Docker emulates it, which is slower but gives the same result.
- **Size:** about 1.5 GB (the scientific stack, W&B, and the dev tools that share the lock file).
- **Changing packages:** edit `environment.yml`, then run `conda-lock -f environment.yml -p linux-64` and commit both files.

## Monitoring
The live service sends its answers to Weights & Biases so the team can see when the model stops helping.

- **What is logged:** for each `/predict` answer, `as_of`, `recommendation`, `probability`, `fallback`, `model_version` and the latency in ms. The API keeps the last `api.monitoring_batch_size` answers (20) in memory and logs them as one table (`serve/predictions`) under a W&B run with job type `serve`. The last partial batch is sent when the service shuts down.
- **What is not logged:** the request itself. Positions, prices and `portfolio_id` never leave the service, and the log has one line per request (as_of, recommendation, fallback, latency). No personal data is involved: the inputs are ETF prices and portfolio weights.
- **Never breaks serving:** with `WANDB_MODE=disabled` or `tracking.enabled: false` nothing is kept or sent. If W&B cannot be reached, at start-up or later, the error is logged and the request still gets its answer.
- **Weekly monitoring run:** `python -m scripts.monitor_report`, once a week after Friday's close. It writes `reports/monitoring.json` and, with tracking on, logs a W&B run of job type `monitor` with two charts, each with its retrain limit drawn in: the fallback rate per week (from the `serve/predictions` tables) and the input drift per week. W&B view: [etf-rebalancing project](https://wandb.ai/ie-mbds-group-2/etf-rebalancing), runs of job type `monitor`. With W&B off, pass `--answers <csv>` (columns `as_of`, `fallback`) to measure the fallback rate from a file.
- Monitoring view in Weights & Biases: https://wandb.ai/ie-mbds-group-2/etf-rebalancing/runs/f80resps (weekly drift PSI and fallback rate, with the retrain limits drawn in).
- **Drift measure and limit:** the population stability index (PSI) of the daily returns of SPY, EFA, IEF and GLD, comparing the last 180 trading days of the inference data with the train years (2007-09-07 to 2019-12-31), in 10 bins cut at the train years' deciles. The weekly number is the largest PSI over the four tickers; the alert fires above 0.2. Why PSI and 0.2: it is the standard drift measure in credit scoring, read as below 0.1 stable, 0.1 to 0.2 watch, above 0.2 a real shift. Why daily returns: every market feature is built from them, and they are close to independent from day to day. The 4- and 12-week returns and the 20-day volatility overlap, so their PSI stays far above 0.2 even when nothing has changed (on the synthetic prices, it fired in every week tested, with a median of 3.5 over 120 days). Why 180 days: noise alone gives a PSI near (10 - 1) / 180 = 0.05; on the synthetic prices (three seeds) it never fired with no change, and it caught daily moves 1.5 times larger after 90 to 110 trading days. Why not BIL: its daily return is the T-bill rate, near 0 in most train years and about 5% since 2022, so it would keep the alert on permanently. Settings: `monitoring` in `config.yaml`. On the official snapshot the alert fires for the week of 26 Dec 2025 (EFA daily returns, PSI 0.254).
- **Retrain triggers:** the full list is in one place, "What triggers a retrain" in the model card. This monitoring measures two of them: the fallback rate (above 50% for 4 weeks in a row) and the input drift (largest daily-return PSI above 0.2 in the latest week). The live drawdown reduction against today's rule (below 0 over a quarter) needs realised portfolio values, which the API does not log; it is checked in the monthly review.

## Future work
Five ideas, in the order I would take them. Each says what it is, why it is not done now and the first step.

**Any universe, or a whole book.** *What:* accept any tickers, or a whole book of holdings, not only the five ETFs. *Why not now:* the model's inputs are per ticker (`w_SPY`, `drift_SPY`, `ret_4w_SPY` and so on), so its input has a fixed length and fixed column names. A model trained on these five cannot take a sixth ETF or a different one, and `data.tickers`, the target weights and the label would all change with the list. The asset-agnostic design removes this: describe each holding by features that do not carry its name (its own weight, drift, 4-week and 12-week return and volatility), add a few portfolio-level features (largest drift, drawdown, weeks since the last rebalance), and train one shared model on many assets. *First step:* write the feature table for this design on the same five ETFs, train it, and check that it reproduces the current gate result before any new ticker is added.

**Continuous training.** *What:* retrain on new data on a schedule and replace the served model only when the new one passes the gate. *Why not now:* there is one frozen snapshot, Yahoo's data is for personal use only so a firm needs a licensed feed, and no model has passed the gate yet, so there is nothing to replace. *First step:* with a licensed feed in place, add a scheduled workflow that fetches and validates new prices, trains, runs the gate and registers the model in Weights & Biases. The `prod` alias must stay on the previous version when the gate fails.

**Rolling target weights.** *What:* when the target weights are calculated (minimum variance, maximum Sharpe, mean-variance, risk parity), refit them on a moving window at each review instead of once. *Why not now:* today they are fitted once on the train years only, which keeps the test years sealed. A rolling fit must use only data known at each review date or it leaks the future, and because the labels depend on the target, every change of target means a retrain. *First step:* refit on the trailing years inside the walk-forward check, and report how much the targets move from one review to the next.

**A stronger agent.** *What:* a richer reinforcement learning agent, for example more than three actions or a better reward. *Why not now:* there are only about 950 weekly decisions, the half step is never the best action (0% of the train weeks, gap 14), and even perfect foresight with these three actions reduces drawdown by only 0.0044 (gap 7), so a bigger model on the same actions has little to gain. *First step:* change the action set first, for example rebalancing back to the edge of the tolerance band instead of halfway, and rerun the feasibility check before training anything larger.

**Named portfolios.** *What:* several model portfolios, such as conservative, balanced and growth, each with its own targets, thresholds and gate. *Why not now:* each one needs its own target weights, labels, model and gate result, and the `prod` alias holds one model. The API request already has an optional `portfolio_id`, but nothing uses it to choose a model yet. *First step:* make a second configuration (target weights and gate) for one more portfolio, run the pipeline for it, and give its model its own alias.

## Repo structure
```text
.
├── README.md             # business case (section 1) and how to run
├── TEAM_BRIEF.md         # agreed settings, decisions, owners
├── config.yaml           # every non-secret setting
├── environment.yml       # conda environment
├── docs/                 # MODULE_SPECS.md: the module contract
├── scripts/              # fetch_prices.py: makes the price snapshot; monitor_report.py: weekly drift and fallback run
├── src/                  # one module per job, main.py runs them all
├── tests/                # automated tests
└── .github/workflows/    # ci.yml: lint-and-test
```

## Team and module owners
| Owner | Modules |
| --- | --- |
| Ricardo Arciniegas (@rarciniegascueva) | `utils`, `load_data`, `scripts/fetch_prices.py`, README section 1 |
| Tate Bothner (@tatebothner) | `clean_data`, `validate` |
| Farah (@faalnuseiratie) | `features` |
| Riccardo Bellini (@riccbell) | `portfolio`, `train` |
| Raja Akel (@raja-akel) | `evaluate`, `infer` |
| Miguel Palanca (@Smiggus) | `main`, `costs`, CI, release |

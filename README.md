# [Project name: to be chosen by the team]

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
Move the balanced portfolio's max drawdown from the baseline to at least 20% smaller (for example -20% to -16%) within 12 months of shadow operation alongside the current process.

| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | max drawdown | To be measured | At least 20% smaller |
| ML metric | `calmar_ratio` (higher is better), median of 5 training runs, 2022 to 2025 | To be measured | At least 1.20 x baseline |
| Guard | annual net return | To be measured | No more than 1 percentage point below baseline |
| Guard | annual trading cost | To be measured | No more than 2 x baseline |
| Challenger | `calmar_ratio` vs selected simple rule | To be measured | At least 1.05 x selected rule (ESTIMATE) |

Do not release if, on 2022 to 2025, any of these holds: `calmar_ratio` below 1.20 x baseline; annual net return trails the baseline by more than 1 percentage point; annual trading cost exceeds 2 x the baseline's; `calmar_ratio` below 1.05 x the selected simple rule. Then recommend the selected simple rule if it passes the gate against the baseline, otherwise keep the current process.

Limits (ESTIMATES): weekly decision run under 60 s; compute cost under €0.01 per decision.

### Solution
Every Friday after the close, the portfolio manager receives a one-line recommendation for the balanced portfolio: "do nothing", "rebalance halfway" or "rebalance fully back to target". It shows the current drift, the estimated trading cost and a confidence level, to decide whether to rebalance next week.

The model predicts which of three actions (0%, 50% or 100% of the way back to target) gives the best next-week result after costs. Inputs, all known at Friday's close: current weights and drift per ETF, drop since the portfolio's last high, 4-week and 12-week returns, 20-day volatility per ETF, weeks since last rebalance, estimated cost of a full rebalance. Weekly batch. Target weights stay fixed, so the agent only decides when and how much to rebalance.

Timing rule: decision at the close of Friday t, trade at the close of the next trading day e. The new weights earn returns from the day after e to the next execution day, inclusive. The agent and the baseline share one execution function.

Fallback: if the model is unavailable or unsure, the current process applies unchanged. "Unsure" means the preferred action has less than 60% probability (ESTIMATE), or any input is missing or stale. The portfolio manager approves every trade. A large change (total buys plus sells above 10% of portfolio value in one week, ESTIMATE) also goes to the monthly Investment Committee.

### Data
Source: Yahoo Finance daily prices for SPY, EFA, IEF, GLD and BIL via `yfinance` (version pinned in `environment.yml`). Yahoo owns the data. [TEAM MEMBER] owns the frozen snapshot. Size: about 4,000 trading days x 5 ETFs, January 2010 to December 2025, plus 2009 used only to warm up the inputs. First decision date: 2010-01-08. Split by time: train 2010 to 2019, validation 2020 to 2021, test 2022 to 2025.

Label: no target column. The reward is the next window's portfolio return minus 0.1% of the value traded, minus a penalty for new drops below the previous high. Refresh: weekly in operation. For the project, one frozen snapshot versioned with DVC and a checksum. Training never downloads data.

Personal or sensitive data: NO. Public prices only. The client and portfolio are hypothetical. Yahoo Finance is a stand-in for this course project. Operation at a firm would require a licensed market data source.

Tests: every input date is on or before the decision day; the first reward date is after the execution date; reward windows join up with no gaps or overlaps; the agent and baseline use the same execution function.

### Value, cost and risks
Benefit over a non-AI approach: smaller losses in downturns, with net return no more than 1 point below the baseline per year and trading cost no more than 2 x the baseline's. Measured by max drawdown and `calmar_ratio` against the baseline and the best simple rule on 2022 to 2025.

Simple-rule challengers, checked weekly on Friday and always fully back to target: current rule (quarterly, ±5%); band 2.5 points; band 1.5 points; persistent drift (at least 2 ETFs beyond 2.5 points on each of the last 5 trading days). The best is selected on 2020 to 2021 by `calmar_ratio`, among rules that pass the same guards. Parameters are the team's own assumptions, inspired by public descriptions checked on 9 Nov 2026: Indexa's drift bands of 1.5 and 1.25 percentage points as reported by Finect (secondary source), and Finizens' multi-asset, multi-day trigger as described by Finect on 27 Oct 2026 (company-reported, no numeric parameters given).

Rough cost (ALL ESTIMATES): course build by the 4 to 5 person team, part-time, about 6 weeks. At a firm: 1 ML engineer (half-time, 12 weeks) plus the quant analyst (a quarter of their time), then about a tenth of one person's time to run and monitor. Infrastructure: laptop CPU for training, GitHub Actions, W&B, Render free or lowest tier, under €50 per month.

| Risk | Mitigation |
| --- | --- |
| Data quality: Yahoo errors (missing days, price spikes, adjustments that change on re-download) | Validation tests on every snapshot (no missing trading days, no extreme daily moves, all 5 tickers present); frozen DVC snapshot with checksum |
| Model error: overfitting, since about 520 training decisions (ESTIMATE) is little for RL | Median of 5 seeds; feasibility ceiling check before training; simple-rule challengers; small model with only 3 actions |
| Model error: regime change (for example 2022, when stocks and bonds fell together) | Monitor input drift and alert above a set limit (limit to be defined); fall back to the current rule when inputs are out of range |
| Operational: Friday batch fails or data arrives late | Automatic fallback to the current rule; alert to the quant analyst; no recommendation without fresh data |
| Operational: portfolio managers trust the agent blindly | Human approval on every trade; log every recommendation, decision and outcome; monthly review of agreement vs overrides |
| Operational: a new model version performs worse | Model registry with versions; one-step rollback to the previous approved version, or to the current rule if no approved version exists |

Scales to (ideas, not findings): the firm's conservative and growth model portfolios, UCITS ETF versions, indexed pension plans, other rule-based rebalancing jobs.

### Open gaps
1. Baseline max drawdown, `calmar_ratio`, return and cost are To be measured over 2010 to 2021 (feasibility check and challenger selection) and 2022 to 2025, before any model is trained.
2. No release rule ties the gate to max drawdown. A model can pass `calmar_ratio` without moving the KPI, and only the return guard blocks the "sit in cash" case.
3. `calmar_ratio` needs one definition in this README and in `config.yaml` (proposed: annualised net return divided by absolute max drawdown).
4. Not stated: which weights each action moves toward target (Friday or Monday pre-trade), and which weights the Friday cost estimate uses.
5. Not stated: starting weights on the first decision date and at the start of 2022, and the scope of the "every day in exactly one window" test.
6. `config.yaml` needs fixed meanings for: "±5%" and "±2.5" (percentage points proposed), the quarterly date, daily weights for the persistent rule, simple or compounded weekly reward, the drawdown penalty reference, and a minimum trade size (Finizens uses one).
7. "Confidence" is an action probability, not a calibrated confidence. The 60% and 10% thresholds are estimates. The weekly vs monthly committee cadence is undefined for large changes.
8. A feasibility check before training: can a timing-only agent with perfect foresight reach 1.20 x baseline on 2010 to 2021?
9. Evidence limits: the 2022 to 2025 test period holds essentially one drawdown, 2020 to 2021 includes the COVID crash, there are only about 830 weekly decisions in total (ESTIMATE), and the 12-month shadow period may contain no drawdown. Not yet in the risk table.
10. "Wide drift band" as the cause is a hypothesis. The band research is not done.
11. `[TEAM MEMBER]` placeholder: name the snapshot owner and who read Yahoo's terms, with the date.
12. Check the `yfinance` notes for the pinned version (the cited pages were for older versions), confirm the tickers, and confirm all five ETFs have data from 2009.
13. The drift monitor alert limit has no number. The €0.01 cost limit is compute only. The scaling claims are ideas.
14. Challengers check weekly, whereas Indexa monitors daily. The project name is not chosen.

---

# 2.6-ML-Ops
2.6 - Machine Learning Ops Team Repository for IE MBDS April 26 Intake.

---

# MLOps Project Skeleton (reference only)

This repo is a **map, not a codebase**. Every file is a placeholder: no code, only a short note on
what the file does, where it sits in the pipeline and which files it talks to.

Create the same folders and files in **your own** project repo, then write the code for **your** use case.

## Stages = branches

| Branch | Stage | Sessions | What it adds |
|---|---|---|---|
| `main` (you are here) | 1. Kickoff | 1 to 3 | Folder structure, business case, one placeholder per pipeline step |
| `stage-2-modular` | 2. Modular code | 4 to 9 | Tests, style and test settings, first CI (Continuous Integration) workflow |
| `stage-3-cicd` | 3. CI/CD | 10 to 14 | Config loader, logging, experiment tracking, API, Docker, deployment |

Tip: on GitHub, open **Compare** (`main...stage-2-modular`) to see exactly what each stage adds.

## The pipeline at a glance

```text
                           main.py  (orchestrator, one command)
                              |
 data/raw/ -> load_data -> clean_data -> validate -> split -> features -> train -> evaluate
                                                (train/val/test) (recipe)   (fit)   (val split)
                                                                              |
                     reports/predictions.csv <- infer <- models/model.joblib (save)
```

## Stage 1 structure

```text
.
├── README.md                 # business case, success metrics, how to run
├── config.yaml               # non-secret settings (paths, target, split sizes)
├── environment.yml           # dependencies (conda)
├── .env.example              # names of secrets, never real values
├── .gitignore                # keeps data, secrets and generated files out of Git
├── LICENSE
├── .github/
│   └── pull_request_template.md
├── notebooks/                # the "lab": exploration only
├── src/                      # the "factory": one job per module
│   ├── load_data.py          # 1. read raw data
│   ├── clean_data.py         # 2. stateless cleaning
│   ├── validate.py           # 3. data quality gate
│   ├── features.py           # 4. unfitted feature recipe
│   ├── train.py              # 5. fit the pipeline
│   ├── evaluate.py           # 6. metrics on unseen data
│   ├── infer.py              # 7. predictions on new data
│   ├── utils.py              # shared read and write helpers
│   └── main.py               # 0. orchestrator: python -m src.main
├── data/  raw/ processed/ inference/
├── models/                   # saved model (generated)
├── reports/                  # predictions, plots (generated)
└── tests/                    # automated tests (stage 2)
```

## How to use this skeleton

1. Open a file and read its note
2. Create the equivalent file in your repo
3. Write your own code: the note describes the job, not the solution

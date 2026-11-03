# 2.6-ML-Ops
2.6 - Machine Learning Ops Team Repository for IE MBDS April 26 Intake.

# Margin Call Early Warning

> SET should fund a tested, monitored early-warning pipeline that gives a day's head start, protecting clients and deal flow, instead of fixed ladders.

## 1. Business case

### Context

Client: a European bank, Equity Derivatives, Strategic Equity Transactions (SET) desk. SET lends against listed shares through margin loans, total return swaps and collars to founders and families, corporates monetizing stakes in other listed companies, and executives pledging vested ESOP shares. Pledges of a borrower's own or treasury shares are excluded.

Maturity: data HIGH (Bloomberg on the desk, daily LTV monitoring), tools LOW (LGD and haircut work in an Excel model owned in London, with a steep learning curve), processes MEDIUM (committee-based credit governance), people to be confirmed.

Owner after delivery: the SET desk uses the warnings; Risk validates and approves the model.

### Problem and baseline

The SET desk struggles with margin calls that arrive too late on slow-to-sell stakes, because the contractual call level is fixed at deal inception and ignores current volatility and liquidity. Today they monitor LTV daily (BOD or EOD) against the fixed contractual ladder.

Baseline: in the MDA II March 2020 replay (7 fictional loans), a dynamic volatility and volume rule called up to 4 days earlier than the fixed ladder and caught 2 loans it missed, at a cost of 11 calls instead of 8 and USD 57MM more cash asked of borrowers. Source: MDA II notebook outputs. Median warning lead time for that rule on 2018-2025 data: to be measured, by running it on the same simulated book as the model.

### Goal and success metrics

Move median warning lead time before the contractual margin call from the baseline (MDA II dynamic rule, to be measured on the 2018-2025 backtest) to baseline + 1 trading day, by the final presentation (19 Nov 2026).

| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | Median warning lead time (trading days) | To be measured (MDA II dynamic rule) | Baseline + 1 trading day |
| ML metric, quality gate | `recall` | To be measured | >= 0.80 on unseen data |
| ML metric, floor | `precision` | To be measured | >= 0.50 on unseen data |

Link to the KPI: catching more calls earlier gives the desk more lead time, and a missed warning leaves the bank one day less to secure a cure. The contractual call still fires, so warnings must be right often enough for the desk to act. If either metric fails on unseen data, the model is not released.

Limits: the full pipeline (train and score the book) runs on a standard laptop CPU within 1 hour, with no GPU or paid cloud. Daily batch, so response time per prediction is not critical.

### Solution

The SET desk receives, each day at BOD, the probability that each loan is called within 5 trading days, to decide whether to call the client and warn them ahead of the contractual call.

The model predicts that probability from yesterday's closing price, historical volatility, ADV and days to liquidate the stake, compared with the loan's current LTV, its contractual ladder and the MDA II haircut. Daily batch at BOD, so the desk can act during the day.

If the model is unavailable or unsure: the contractual LTV ladder still triggers margin calls as today.

### Data

Source: yfinance daily prices and volumes for the 7 MDA II loan tickers, 2018-01-01 to 2025-12-31. Owner: Miguel (fetch script and raw file); data provided by Yahoo Finance via yfinance.

Label: y = 1 if a simulated loan's LTV reaches its contractual call level within the next 5 trading days, else 0. Loans are simulated per ticker with random origination dates (about one per year, seeded in `config.yaml`), a 2 to 3 year term, the ticker's MDA II ladder, and every borrower curing back to the opening LTV. One row per loan per day.

Refresh: daily.

Personal or sensitive data: No. Prices and volumes only; ownership data is left out of this stage.

### Value, cost and risks

Benefit over a non-AI approach: earlier, ranked warnings than the MDA II dynamic rule, so the desk can contact clients before the contractual call, especially on slow-to-sell stakes; measured by median warning lead time at `recall` >= 0.80 and `precision` >= 0.50.

Rough cost (estimate): 6 team members over about 3 weeks (to 19 Nov 2026); runs on a standard laptop CPU, free public data, no paid compute.

| Risk | Mitigation |
| --- | --- |
| Data quality: yfinance gaps, survivorship bias and personal-use terms | `validate.py` stops the run on missing or invalid prices; risk stated here; broader licensed or PIT data pending professor approval |
| Simulation bias: random origination may understate call frequency (real loans cluster after rallies) | Sensitivity run with rally-tilted origination; conservative warning threshold in `config.yaml` |
| Model error: too many false alarms and the desk ignores warnings | `precision` floor of 0.50 in the quality gate; no release if it fails |
| Drift: market regime shifts degrade the model | Monitor `recall` and `precision` against realised calls; retrain when either falls below the gate |

Scales to: Russell 2000 universe (IWM N-PORT holdings or Miguel's datalake), Miguel's GPU cluster or AWS for larger books, ownership data (SEC EDGAR, Companies House) for wrong-way risk, and failed-cure scenarios to measure uncovered loss.

### Open gaps

- People maturity rating for the SET desk
- Client is a realistic but unnamed European bank; decide whether to name a realistic example
- Baseline lead time, recall and precision for the MDA II dynamic rule: to be measured on the same 2018-2025 simulated book
- Train, validation and test split must be by date, not random, so no future prices leak into training; agree with the kick-off helper
- Professor approval for a broader universe (Russell 2000) or Miguel's datalake

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

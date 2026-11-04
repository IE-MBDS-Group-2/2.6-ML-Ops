# Day-ahead electricity demand forecasting

> The trading desk should fund an automated, monitored forecasting pipeline to cut correction losses, instead of checking models by hand every morning.

## 1. Business case

### Context

- **Client:** an energy trading desk that takes day-ahead positions based on national (peninsular) electricity demand
- **Maturity:** data high, people high (a small quant team already builds forecasting models), processes medium (assumption: models are run and checked by hand)
- **Owner after delivery:** the desk's quant team runs and maintains the system

### Problem and baseline

The quant team struggles with costly position corrections after the day-ahead auction because its demand forecast is produced manually under the 12:00 deadline. Today they run their models and check the output by hand each morning.

**Baseline:** forecast error (MAPE) of the current process: to be measured from the desk's records. Seasonal-naive MAPE and REE's day-ahead forecast MAPE: to be measured from REData over the last 12 months.

### Goal and success metrics

| Type | Metric | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | Euros lost per month on position corrections | To be measured | 10% reduction within 6 months (estimate) |
| ML metric (quality gate) | MAPE on an unseen test period | Seasonal-naive MAPE: to be measured | Lower than seasonal-naive |
| ML metric (reported) | MAE on an unseen test period | To be measured | Reported alongside MAPE |
| Benchmark (reported, not a gate) | MAPE compared with REE's day-ahead forecast (version available before 11:00) | To be measured | Reported |
| Operational | Forecast delivery time | Manual process | Before 11:00 every day |
| Operational | Data validation pass rate | To be measured | To be set |
| Operational | Time from drift alert to redeployed model | Not tracked today | To be set |

**Do not release if:** the model does not beat seasonal-naive on the test period.

### Solution

The desk's traders receive a 24-hour national demand forecast at 15-minute grain each morning before 11:00 to decide their day-ahead positions. The model (LightGBM) predicts peninsular demand from weather forecasts, calendar features and recent demand, in a daily batch. If the model is unavailable or unsure, the desk uses the seasonal-naive forecast. A trader signs off on the final position.

### Data

- **Demand (target):** REE, via the REData API and ESIOS. Public, refreshed daily. Actual demand is published by REE and used as the label
- **Weather (features):** Open-Meteo historical and forecast APIs (free for non-commercial use), same parameters for training and serving. Hourly values are spread across four 15-minute intervals
- **Calendar (features):** `holidays` Python package (Spanish national and regional holidays)
- **Market prices (for euro impact):** OMIE day-ahead prices, ESIOS imbalance prices
- **Personal or sensitive data:** none, demand is aggregated at national level. API tokens are kept out of the repository

### Value, cost and risks

- **Benefit over a non-AI approach:** open gap
- **Rough cost:** open gap
- **Scales to:** open gap

| Risk | Mitigation |
| --- | --- |
| Bad or missing input data | Data validation checks before each run. If they fail, the desk uses the seasonal-naive forecast |
| Model makes poor forecasts | Release gate (must beat seasonal-naive), and a trader signs off on the final position |
| Drift over time | Forecast compared with actual demand every day, with retraining when accuracy degrades |
| API unavailable | Seasonal-naive fallback is delivered before 11:00 |

### Open gaps

- Euro cost of one MWh of forecast error for a trading desk
- All baseline values: desk's current MAPE, seasonal-naive MAPE, REE MAPE, euros lost per month
- Validate the 10% target once the baseline is measured
- Number of years of training history
- Weather: training on actual values but serving on forecasts (training and serving mismatch)
- Recent-demand features limited to data published before 11:00
- Definition of "unsure" that triggers the seasonal-naive fallback
- Targets for data validation pass rate and drift-to-redeploy time
- Mitigation for weather interpolation at 15-minute grain
- Benefit over seasonal-naive and the current hand-checked process
- Rough cost (people, infrastructure, weeks)
- Where the solution could scale

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

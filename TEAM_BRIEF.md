!!! This is the scaffold, filled in after our business case coach prompt is run !!!

# Team brief
Business case: README.md, section 1 (merged into dev on [date])
Use case: [who] decides [what] using [the prediction]
Notebook: [file name]
Target: [column] | Problem type: [classification / regression] | ID column: [column / none]
Feature groups: [group]: [columns] -> [transformation] (why)   <- one line per group
Excluded columns: [column] (why)
Model: [family] with [key parameters]
Metric: [name], key in evaluate_model output: ["key"], quality gate: [>= or <=] [threshold]
Split: test [share] and validation [share], both of the full dataset; random_state [number]
Validation rules: [rule] (why it should stop the pipeline)
README follow-ups: [change or gap] -> [owner]
features.py signature: get_feature_preprocessor([params]), settings per group: [e.g. n_bins=4]
Agreed extra parameters: [function(parameter=default)]
utils.py helpers: load_csv, save_csv, save_model, load_model
Environment: [packages beyond pandas, numpy, scikit-learn, joblib, pytest, flake8]
Decisions: [decision] -> [choice] (why)
Owners (team): [module] -> [name] (@[GitHub username]), branch feature/[module-name]   <- one per module, utils first, main last

## Roles and contributions

Assumes we go with the RL rebalancing proposal (PR #7). Paths are where each piece will live; new files are marked (new).

| Role | Name | Contribution | Where it lives |
|---|---|---|---|
| Admin | Miguel Palanca (@Smiggus) | Repo admin: branch protection, reviews, merges, this brief | Repo settings, `TEAM_BRIEF.md`, `.github/CODEOWNERS` (new) |
| | | Traditional portfolio baselines (mean-variance, minimum variance, risk parity) to compare the ML result against. Fit on 2010-2019 only | `src/baselines.py` (new), `config.yaml` (`baselines:`), `tests/test_baselines.py` (new) |
| | | Trading cost model: exits cost more than entries, plus a market impact term that grows with trade size | `src/costs.py` (new), used by the shared execution function; `config.yaml` (`costs:`); `tests/test_costs.py` (new) |
| | | "Is the market stressed?" input from public data (VIX vs VIX3M, recent volatility vs normal) | `src/load_data.py` (extra tickers), `src/features.py` (stress feature), `config.yaml` (`features:`), `tests/test_features.py` |
| | | CI/CD on self-hosted runners, locked down so only approved people can trigger runs | `.github/workflows/ci.yml` (new, stage 2), `.github/workflows/cd.yml` (new, stage 3), runner notes in `README.md` |
| Collaborator 1 | Ricardo Arciniegas (@rarciniegascueva) | Business case owner (PR #7). Other modules TBD | `README.md` section 1 |
| Collaborator 2 | Tate Bothner (@tatebothner) | TBD | TBD |
| Collaborator 3 | Open | | |
| Collaborator 4 | Open | | |
| Collaborator 5 | Open | | |

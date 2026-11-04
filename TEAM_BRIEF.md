## 1. Business case

### Context

Client: a Premier League club (unnamed), performance analysis department. The department prepares each fixture with the coaching staff: opponent briefing, squad selection and rotation, and match-day plan.

Maturity: data MEDIUM (public historical results; no tracking data in this stage), tools LOW (spreadsheets and manual form tables), processes MEDIUM (weekly fixture-prep cycle), people to be confirmed.

Owner after delivery: the head of performance analysis uses the output; the coaching staff decide how to act on it.

### Problem and baseline

Fixture preparation relies on manual form tables and gut feel, so a home match that is likely not to be won is flagged late, if at all. Today the analyst eyeballs recent results and the league table.

Baseline: always predict a home win (the majority class), plus a "last-5-matches points" rule. Both to be measured on the same held-out seasons.

### Goal and success metrics

Give the analyst a ranked, repeatable pre-match probability that the home team wins, produced before each matchweek, by the final presentation.

| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | Share of home matches correctly flagged as win / not-win | To be measured (majority-class rule) | Baseline + 3 points |
| ML metric, quality gate | `roc_auc` | 0.50 (random ranking) | >= 0.60 on unseen seasons |
| ML metric, floor | `accuracy` | To be measured (majority class) | > majority-class baseline on unseen seasons |

Thresholds are a first proposal for the team to confirm. If either metric fails on unseen data, the model is not released.

Limits: trains and scores on a standard laptop CPU in minutes, no GPU or paid cloud. Weekly batch, so response time per prediction is not critical.

### Solution

Before each matchweek, the analyst receives the probability that each home team wins, to decide which fixtures need extra preparation or squad-rotation attention.

The model predicts it from information available before kick-off only: each team's recent form (points, goals scored and conceded over the previous matches), home and away record, and rest days since the last match. Weekly batch.

If the model is unavailable or unsure: the analyst keeps using the current manual form tables.

### Data

Source: match results CSVs per season from football-data.co.uk (Premier League file). Multiple seasons downloaded as raw files. Owner: [name].

Label: y = 1 if the home team wins (full-time result = H), else 0. One row per match.

Refresh: weekly during the season.

Personal or sensitive data: No. Team-level match results only.

### Value, cost and risks

Benefit over a non-AI approach: a consistent, ranked pre-match probability instead of ad hoc judgment, measured by `roc_auc` and `accuracy` against the baselines above.

Rough cost (estimate): [team size] members over [duration]; laptop CPU, free public data, no paid compute.

| Risk | Mitigation |
| --- | --- |
| Data leakage: using match statistics (shots, goals, cards) from the match being predicted | Features built only from previous matches; `validate.py` checks no post-match columns reach the model |
| Football is noisy: modest signal means the model may barely beat the baseline | Gate on `roc_auc` and `accuracy` vs baseline; no release if it fails |
| Drift: squads, managers and promoted/relegated teams change season to season | Evaluate on a later season than training; retrain when metrics fall below the gate |
| Data quality: missing or inconsistent rows, team name changes across seasons | `validate.py` stops the run on missing dates, scores or unknown team names |

Scales to: other leagues from the same source, bookmaker-free feature sets, lineup and injury data, and draw/away outcomes (3-class).

### Open gaps

- Final owner names and team size
- Confirm metric thresholds with the team
- Train, validation and test split must be by date (whole seasons held out), not random, so future matches do not leak into training
- Which seasons to include
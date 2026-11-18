"""
rules.py  [Stage 3: SIMPLE-RULE CHALLENGERS]
============================================
Purpose   The business case's simple rebalancing rules, each as a policy through the same simulator and costs,
          and the selection of the best one on validation among the rules that pass the guards
Input     close_wide, volume_wide, market features (as for portfolio.simulate), target weights, the config
          sections challengers, portfolio.current_rule, costs and guards; optionally the agent's daily run
Output    make_rule_policy: a policy (state row -> action); run_challengers: one row per rule with risk, return,
          cost, tracking error and guard results; select_challenger: the selected rule's name
Called by main.py (Stage 3)          Calls   portfolio.simulate, evaluate.portfolio_metrics / tracking_error /
                                              guard_report

Why it matters
- The agent must beat not only today's rule but the best simple rule (README "Do not release if")
- Every rule goes through portfolio.simulate, so they are charged by exactly the same timing rule and costs
- Every number (bands, days, thresholds, wait, equity tickers) comes from config: the README's assumptions,
  never tuned on validation

Tests     tests/test_rules.py
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from src import evaluate, portfolio

logger = logging.getLogger(__name__)

RULES: tuple[str, ...] = ("current", "band_2_5", "band_1_5", "persistent_drift", "asymmetric", "matched_exposure")
MATCHED_EXPOSURE = "matched_exposure"  # needs the agent's daily run
_NO_DRIFT = 1e-9                       # weights this close to target count as on target

Policy = Callable[[pd.Series], float]


def _weights(state_row: pd.Series, tickers: list[str]) -> pd.Series:
    """The pre-trade weights in a state row.

    Inputs:  state_row, as built by portfolio.simulate (columns w_<T>); tickers, in target order.
    Outputs: the weights as a Series indexed by ticker.
    Why:     every rule reads the weights the same way the model's state does.
    """
    return pd.Series([float(state_row[f"w_{ticker}"]) for ticker in tickers], index=tickers)


def _max_drift(state_row: pd.Series, target: pd.Series) -> float:
    """The largest absolute distance of any weight from target, in fractions (0.03 = 3 points).

    Inputs:  state_row (w_<T> columns); target weights.
    Outputs: max |w - target| over the tickers.
    Why:     the tolerance bands and the asymmetric rule all trigger on the worst-drifted ETF.
    """
    return float((_weights(state_row, list(target.index)) - target).abs().max())


def _quarter_starts(decisions: pd.DatetimeIndex, months: list[int]) -> set[pd.Timestamp]:
    """The first decision date of each listed month (the calendar dates of a quarterly rebalance).

    Inputs:  decisions, the run's decision dates; months, e.g. [1, 4, 7, 10] (portfolio.current_rule).
    Outputs: the set of decision dates that open one of those months.
    Why:     the matched-exposure mix rebalances on the same calendar as today's rule.
    """
    decisions = pd.DatetimeIndex(decisions)
    periods = decisions.to_period("M")
    first_in_month = np.r_[True, periods[1:] != periods[:-1]]
    in_months = decisions.to_series().dt.month.isin(months).to_numpy()
    return set(decisions[first_in_month & in_months])


def _band_name(band: float) -> str:
    """The rule name a band gives: 0.025 -> "band_2_5" (points, with the decimal point as an underscore).

    Inputs:  band, a fraction from challengers.bands.
    Outputs: the rule name.
    Why:     ties each band rule in RULES to its number in config, so neither is hard-coded twice.
    """
    return "band_" + f"{band * 100:g}".replace(".", "_")


def _band_policy(name: str, target: pd.Series, cfg: dict) -> Policy:
    """Tolerance band: rebalance fully when any ETF drifts beyond the band, at any review.

    Inputs:  name, "band_2_5" or "band_1_5"; target weights; cfg (reads challengers.bands).
    Outputs: the policy.
    Why:     the README's band rules (2.5 and 1.5 points), with no calendar.
    Raises:  ValueError if no band in challengers.bands gives this name.
    """
    bands = {_band_name(band): float(band) for band in cfg["challengers"]["bands"]}
    if name not in bands:
        raise ValueError(f"Rule {name!r} needs a band in challengers.bands (config.yaml) that gives its name "
                         f"(0.025 gives band_2_5); the configured bands give {sorted(bands)}.")
    band = bands[name]

    def policy(state_row: pd.Series) -> float:
        """Inputs: a state row. Outputs: 1.0 beyond the band, else 0.0. Why: the band rule's whole logic."""
        return 1.0 if _max_drift(state_row, target) > band else 0.0

    return policy


def _persistent_policy(target: pd.Series, close_wide: pd.DataFrame, cfg: dict) -> Policy:
    """Persistent drift: rebalance fully when enough ETFs sat beyond the band on each of the last days.

    Inputs:  target weights; close_wide (date x ticker); cfg (reads challengers.persistent_drift: min_assets,
             band, days).
    Outputs: the policy.
    Why:     the README's multi-asset, multi-day trigger: ignore a drift that lasts one day only.
             Past weights are rebuilt from the decision day's weights and prices (w_d ~ w_t x P_d / P_t), which
             is exact for weekly and monthly reviews: the last trade is at the close of the day after the
             previous decision, before the window starts. With daily reviews the window holds earlier trades, so
             the rebuilt weights are an approximation. In a holiday week (fewer trading days) the window can
             reach back before that trade, so its first day uses the rebuilt weights, not the ones held then.
    """
    rule = cfg["challengers"]["persistent_drift"]
    tickers = list(target.index)

    def policy(state_row: pd.Series) -> float:
        """Inputs: a state row. Outputs: 1.0 if min_assets ETFs were beyond the band on every one of the last
        days, else 0.0 (also with too few days of prices). Why: the persistent rule's whole logic."""
        window = close_wide.loc[:state_row.name, tickers].tail(rule["days"])
        if len(window) < rule["days"]:
            return 0.0
        held = (window / window.iloc[-1]) * _weights(state_row, tickers)
        drift = held.div(held.sum(axis=1), axis=0) - target
        beyond = (drift.abs() > rule["band"]).sum(axis=1)
        return 1.0 if bool((beyond >= rule["min_assets"]).all()) else 0.0

    return policy


def _asymmetric_policy(target: pd.Series, cfg: dict) -> Policy:
    """Asymmetric: rebalance fully on a drift or drawdown breach, otherwise step back slowly with a minimum wait.

    Inputs:  target weights; cfg (reads challengers.asymmetric: drift_threshold, drawdown_threshold, step,
             min_wait).
    Outputs: the policy.
    Why:     the README's asymmetric rule: act fast when risk is high, trade little when it is not. The wait uses
             the simulator's weeks_since_rebalance, which resets after every trade, so the policy needs no memory.
    """
    rule = cfg["challengers"]["asymmetric"]

    def policy(state_row: pd.Series) -> float:
        """Inputs: a state row. Outputs: 1.0 on a breach, the step after min_wait reviews with some drift, else
        0.0. Why: the asymmetric rule's whole logic."""
        drift = _max_drift(state_row, target)
        if drift > rule["drift_threshold"] or state_row["drawdown"] <= rule["drawdown_threshold"]:
            return 1.0
        if state_row["weeks_since_rebalance"] >= rule["min_wait"] and drift > _NO_DRIFT:
            return float(rule["step"])
        return 0.0

    return policy


def _calendar_policy(decisions: pd.DatetimeIndex, cfg: dict) -> Policy:
    """Calendar only: rebalance fully on the first decision of each quarter month, never on drift.

    Inputs:  decisions, the run's decision dates; cfg (reads portfolio.current_rule.quarter_months).
    Outputs: the policy.
    Why:     the matched-exposure fixed mix is rebalanced quarterly, like today's rule but with no band.
    """
    starts = _quarter_starts(decisions, cfg["portfolio"]["current_rule"]["quarter_months"])

    def policy(state_row: pd.Series) -> float:
        """Inputs: a state row (named by its date). Outputs: 1.0 on a quarter start, else 0.0. Why: calendar."""
        return 1.0 if pd.Timestamp(state_row.name) in starts else 0.0

    return policy


def make_rule_policy(name: str, target: pd.Series, decisions: pd.DatetimeIndex, close_wide: pd.DataFrame,
                     cfg: dict) -> Policy:
    """One of the README's simple rules as a policy: a function from a state row to an action.

    Inputs:  name, one of RULES; target, the weights the rule rebalances to (for matched_exposure: the mix from
             matched_exposure_weights); decisions, the run's decision dates; close_wide (date x ticker);
             cfg, the full config (reads challengers and portfolio.current_rule).
    Outputs: policy(state_row) -> action in [0, 1], ready for portfolio.simulate.
    Why:     every rule then goes through the same simulate and costs as today's rule and the model.
    Raises:  ValueError for an unknown name, or a band rule whose band is not in challengers.bands.
    """
    if name not in RULES:
        raise ValueError(f"Unknown rule {name!r}. Choose one of: {', '.join(RULES)}.")
    if name == "current":
        return portfolio.make_current_rule_policy(target, decisions, cfg)
    if name.startswith("band_"):
        return _band_policy(name, target, cfg)
    if name == "persistent_drift":
        return _persistent_policy(target, close_wide, cfg)
    if name == "asymmetric":
        return _asymmetric_policy(target, cfg)
    return _calendar_policy(decisions, cfg)


def matched_exposure_weights(target: pd.Series, agent_daily: pd.DataFrame, equity_tickers: list[str]) -> pd.Series:
    """Fixed-mix weights that hold the agent's own average equity share, split like the target on each side.

    Inputs:  target weights; agent_daily, the agent's daily frame from portfolio.simulate (w_<T> columns);
             equity_tickers, from challengers.equity_tickers (e.g. SPY, EFA).
    Outputs: the mix, indexed like target, summing to 1.
    Why:     README: "the agent cannot pass just by holding fewer stocks"; this rule holds the same equity share.
    Raises:  ValueError if agent_daily lacks a weight column.
    """
    missing = [f"w_{ticker}" for ticker in target.index if f"w_{ticker}" not in agent_daily.columns]
    if missing:
        raise ValueError(f"agent_daily has no weight columns {missing}. Pass the agent's daily frame from "
                         "portfolio.simulate.")
    share = float(agent_daily[[f"w_{ticker}" for ticker in equity_tickers]].sum(axis=1).mean())
    is_equity = target.index.isin(equity_tickers)
    target_share = float(target[is_equity].sum())
    mix = target * np.where(is_equity, share / target_share, (1.0 - share) / (1.0 - target_share))
    logger.info("Matched-exposure mix: equity share %.4f (target %.4f)", share, target_share)
    return mix


def review_dates(close_wide: pd.DataFrame, start: str, end: str, cfg: dict) -> pd.DatetimeIndex:
    """The decision dates of a run, exactly as portfolio.simulate picks them.

    Inputs:  close_wide (its index is the calendar); start and end (YYYY-MM-DD); cfg (reads rebalancing.review).
    Outputs: the decision dates inside [start, end].
    Why:     the rules' calendar and the tracking-error dates must be the simulator's own.
    """
    days = close_wide.index[close_wide.index <= pd.Timestamp(end)]
    return portfolio.decision_dates(days, start, end, portfolio.review_frequency(cfg))


def target_path(close_wide: pd.DataFrame, target: pd.Series, decisions: pd.DatetimeIndex) -> pd.Series:
    """Values of a portfolio held exactly at target every day, cost-free, on the decision dates.

    Inputs:  close_wide; target weights; decisions.
    Outputs: a value path starting at 1.0 on the first decision date, one value per decision date.
    Why:     the reference for tracking error (README guard), the same for every rule.
    """
    returns = close_wide[target.index].pct_change(fill_method=None).loc[decisions[0]:].iloc[1:]
    daily = pd.concat([pd.Series([1.0], index=[decisions[0]]), (1.0 + returns @ target).cumprod()])
    return daily.reindex(decisions)


def _simulate_rule(name: str, market_data: tuple, target: pd.Series, decisions: pd.DatetimeIndex, cfg: dict,
                   period: tuple[str, str], agent_daily: pd.DataFrame | None) -> pd.DataFrame:
    """Run one rule through portfolio.simulate and return its daily frame.

    Inputs:  name; market_data (close_wide, volume_wide, market features); target; decisions; cfg;
             period (start, end); agent_daily (only read for matched_exposure).
    Outputs: the daily frame (value, cost_paid, weights).
    Why:     one place where a rule meets the simulator, so every rule is run the same way.
    """
    rule_target = target
    if name == MATCHED_EXPOSURE:
        rule_target = matched_exposure_weights(target, agent_daily, cfg["challengers"]["equity_tickers"])
    policy = make_rule_policy(name, rule_target, decisions, market_data[0], cfg)
    daily, _ = portfolio.simulate(*market_data, rule_target, policy, *period, cfg)
    return daily


def _result_row(metrics: dict, guards: dict) -> dict:
    """One results row: the risk, return and cost numbers plus one passed flag per guard.

    Inputs:  metrics, portfolio_metrics plus tracking_error; guards, guard_report for this rule.
    Outputs: a flat dict, ready for a DataFrame and reports/challengers.csv.
    Why:     a CSV cannot hold nested guard dicts; the full reports stay in results.attrs["guards"].
    """
    row = {key: metrics[key] for key in ("max_drawdown", "annual_return", "annual_volatility", "sharpe",
                                         "annual_trading_cost", "tracking_error")}
    row.update({f"guard_{name}": entry["passed"] for name, entry in guards.items()})
    row["guards_passed"] = all(entry["passed"] for entry in guards.values())
    return row


def run_challengers(market_data: tuple, target: pd.Series, cfg: dict, start: str, end: str,
                    agent_daily: pd.DataFrame | None = None,
                    daily_values: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Run every simple rule over [start, end] and check each against today's rule with the README's guards.

    Inputs:  market_data, (close_wide, volume_wide, market features) as for portfolio.simulate; target weights;
             cfg, the full config (reads challengers, portfolio, costs, guards, rebalancing.review); start and
             end (YYYY-MM-DD);
             agent_daily, the agent's daily frame over the same dates (matched_exposure is skipped without it).
             daily_values, an optional mapping to receive daily values for the robustness audit.
    Outputs: one row per rule (index "rule", in RULES order): max_drawdown, annual_return, annual_volatility,
             sharpe, annual_trading_cost, tracking_error, guard_<name> per guard and guards_passed.
             results.attrs["guards"] holds the full guard_report per rule, for select_challenger.
    Why:     the agent must beat the best simple rule, and only rules within the mandate count (README).
    """
    decisions = review_dates(market_data[0], start, end, cfg)
    # tracking error reference: the target's values on the review dates, and reviews per year to annualise
    reference = (target_path(market_data[0], target, decisions),
                 portfolio.periods_per_year(portfolio.review_frequency(cfg)))
    names = [name for name in RULES if name != MATCHED_EXPOSURE or agent_daily is not None]
    if agent_daily is None:
        logger.info("Rule %s skipped: it needs the agent's daily run (agent_daily)", MATCHED_EXPOSURE)
    metrics = {}
    for name in names:
        daily = _simulate_rule(name, market_data, target, decisions, cfg, (start, end), agent_daily)
        if daily_values is not None:
            daily_values[name] = daily["value"]
        metrics[name] = evaluate.portfolio_metrics(daily["value"], daily["cost_paid"])
        metrics[name]["tracking_error"] = evaluate.tracking_error(daily["value"].reindex(decisions), *reference)
    guards = {name: evaluate.guard_report(metrics[name], metrics["current"], cfg["guards"]) for name in names}
    results = pd.DataFrame([_result_row(metrics[name], guards[name]) for name in names],
                           index=pd.Index(names, name="rule"))
    results.attrs["guards"] = guards
    logger.info("Ran %d rule challengers from %s to %s; %d pass the guards", len(results), start, end,
                int(results["guards_passed"].sum()))
    return results


def select_challenger(results: pd.DataFrame, guards: dict) -> str:
    """The rule with the smallest worst loss (max drawdown closest to 0) among the rules that pass every guard.

    Inputs:  results, from run_challengers on validation 2020-2021 (column max_drawdown, index = rule names);
             guards, {rule: guard_report(...)}, normally results.attrs["guards"].
    Outputs: the selected rule's name, for metrics.json ("challenger") and the release decision.
    Why:     README: "The best is selected on 2020 to 2021 by max drawdown, among rules that pass the same guards."
    Raises:  ValueError if a rule has no guard report, or no rule passes the guards.
    """
    missing = [name for name in results.index if name not in guards]
    if missing:
        raise ValueError(f"select_challenger: no guard report for {missing}. Pass results.attrs['guards'] "
                         "from run_challengers.")
    passing = [name for name in results.index if all(entry["passed"] for entry in guards[name].values())]
    if not passing:
        raise ValueError("No rule passes the guards on validation, so there is no challenger to select. "
                         "Keep the current process (README) and report it.")
    chosen = str(results.loc[passing, "max_drawdown"].idxmax())
    logger.info("Selected challenger %s: max drawdown %.4f, best of %d rules that pass the guards", chosen,
                results.loc[chosen, "max_drawdown"], len(passing))
    return chosen

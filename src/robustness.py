"""Robustness diagnostics for portfolio comparisons and reporting."""

import json
import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate, portfolio, rules, tracking, utils

logger = logging.getLogger(__name__)


def _drawdown(values: pd.Series) -> float:
    """Return max drawdown from positive, chronological portfolio values.

    Inputs: Daily portfolio value series.
    Outputs: Worst loss as a nonpositive fraction.
    Why: All robustness comparisons measure loss against the running peak.
    Raises: ValueError if the path has fewer than two values, or any is not finite and positive.
    """
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("A portfolio path needs at least two finite, positive values.")
    return float((values / values.cummax() - 1).min())


def _reduction(agent_mdd: float, baseline_mdd: float) -> float | None:
    """Calculate the gate metric when the baseline has experienced a loss.

    Inputs: Candidate and baseline maximum drawdowns.
    Outputs: Relative reduction, or None for a flat baseline.
    Why: A period without baseline drawdown is inconclusive.
    """
    if baseline_mdd == 0:
        return None
    return evaluate.max_drawdown_reduction(agent_mdd, baseline_mdd)


def feasibility_ceiling(market_data: tuple, target: pd.Series, cfg: dict, start: str, end: str) -> dict:
    """Evaluate a hindsight timing policy on the requested period.

    Inputs: Market frames, target weights, project configuration and inclusive dates.
    Outputs: Baseline and hindsight drawdowns, reduction, gate target and status.
    Why: Measure the protection achieved by labels that can see the next reward window.
    Raises: ValueError if a simulated portfolio path is unusable (fewer than two finite, positive values).
    """
    baseline_policy = portfolio.make_current_rule_policy(
        target, rules.review_dates(market_data[0], start, end, cfg), cfg)
    labelled = portfolio.build_decision_table(*market_data, target, start, end, cfg)
    actions = labelled[cfg["problem"]["target_column"]].dropna()

    def hindsight(state: pd.Series) -> float:
        """Choose a labelled action, or today's rule when the window is incomplete.

        Inputs: Current state indexed by decision date.
        Outputs: Rebalance action.
        Why: Incomplete reward windows have no hindsight label.
        """
        if state.name in actions.index:
            return float(actions.at[state.name])
        return baseline_policy(state)

    candidate, _ = portfolio.simulate(*market_data, target, hindsight, start, end, cfg)
    baseline, _ = portfolio.simulate(*market_data, target, baseline_policy, start, end, cfg)
    baseline_mdd, hindsight_mdd = _drawdown(baseline["value"]), _drawdown(candidate["value"])
    hindsight_reduction = _reduction(hindsight_mdd, baseline_mdd)
    ceiling = None if hindsight_reduction is None else max(0.0, hindsight_reduction)
    return {
        "baseline_mdd": baseline_mdd,
        "hindsight_mdd": hindsight_mdd,
        "hindsight_reduction": hindsight_reduction,
        "max_drawdown_reduction": ceiling,
        "gate_target": float(cfg["quality_gate"]["threshold"]),
        "status": "inconclusive" if ceiling is None else "measured",
    }


def cost_sensitivity(run_policy: Callable[[dict], pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Compare the agent and current rule under the flat and realistic cost models.

    Inputs: Runner returning daily agent_value and baseline_value, and full configuration.
    Outputs: A row per cost model with drawdowns and their relative reduction.
    Why: A timing result can depend on the assumptions behind trading costs.
    Raises: ValueError if the runner returns an unusable portfolio path.
    """
    rows = []
    for model in ("flat", "realistic"):
        daily = run_policy({**cfg, "costs": {**cfg["costs"], "model": model}})
        agent_mdd = _drawdown(daily["agent_value"])
        baseline_mdd = _drawdown(daily["baseline_value"])
        rows.append({"cost_model": model, "agent_mdd": agent_mdd, "baseline_mdd": baseline_mdd,
                     "max_drawdown_reduction": _reduction(agent_mdd, baseline_mdd)})
    return pd.DataFrame(rows)


def without_largest_episode(daily_agent: pd.DataFrame, daily_baseline: pd.DataFrame, *,
                            threshold: float, comparison: str) -> dict:
    """Recompute the gate metric without the baseline's largest peak-to-trough episode.

    Inputs: Aligned daily paths with value columns, and the gate settings (quality_gate.threshold and
        quality_gate.comparison in config.yaml; there are no defaults).
    Outputs: Episode bounds, full and reduced scores, and fragility or inconclusive status.
    Why: Removing the same daily returns from both paths tests dependence on one decline.
    Raises: ValueError if the two paths do not share one unique, sorted calendar.
    """
    agent, baseline = daily_agent["value"], daily_baseline["value"]
    dates = baseline.index
    if not dates.equals(agent.index) or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Portfolio paths must share a unique, sorted calendar.")
    full_score = _reduction(_drawdown(agent), _drawdown(baseline))
    result = {"episode_start": None, "episode_end": None, "full_reduction": full_score,
              "without_episode_reduction": None, "fragile": None, "status": "inconclusive"}
    if full_score is None:
        return result
    trough = (baseline / baseline.cummax() - 1).idxmin()
    peak = baseline.loc[:trough][::-1].idxmax()
    keep = (dates <= peak) | (dates > trough)
    result.update(episode_start=pd.Timestamp(dates[dates.get_loc(peak) + 1]).date().isoformat(),
                  episode_end=pd.Timestamp(trough).date().isoformat())
    if keep.sum() < 2:
        return result
    retained_values = pd.concat([agent, baseline], axis=1).pct_change(fill_method=None).fillna(0)
    retained_values = (1 + retained_values.loc[keep]).cumprod()
    reduced = _reduction(_drawdown(retained_values.iloc[:, 0]), _drawdown(retained_values.iloc[:, 1]))
    result["without_episode_reduction"] = reduced
    if reduced is not None:
        result["status"] = "measured"
        result["fragile"] = evaluate.check_gate(full_score, comparison, threshold) and not evaluate.check_gate(
            reduced, comparison, threshold)
    return result


def bootstrap_p_value(agent_returns: pd.Series, other_returns: pd.Series, n: int, block: int, seed: int) -> float:
    """Estimate a one-sided paired block-bootstrap p-value for mean weekly excess return.

    Inputs: Aligned weekly returns, number of draws, block length and random seed.
    Outputs: One-sided tail probability under the zero-mean null, using half weight for ties.
    Why: Centered, paired circular blocks preserve shared market weeks and short-range dependence.
    Raises: ValueError if the two series have different indexes, if the returns are not finite, if n is
        below 1, or if the block is below 1 or longer than the sample.
    """
    if not agent_returns.index.equals(other_returns.index):
        raise ValueError("Weekly return series must have the same length and index.")
    differences = (agent_returns - other_returns).to_numpy(dtype=float)
    size = len(differences)
    if n < 1 or not 1 <= block <= size or not np.isfinite(differences).all():
        raise ValueError("Use finite weekly returns, positive draws and a block within the sample length.")
    generator = np.random.default_rng(seed)
    observed = differences.mean()
    centered = differences - observed
    tail_count = 0.0
    offsets = np.arange(block)
    for _ in range(n):
        starts = generator.integers(size, size=math.ceil(size / block))
        sample = centered[((starts[:, None] + offsets) % size).ravel()[:size]]
        mean = sample.mean()
        tail_count += float(mean > observed) + 0.5 * float(mean == observed)
    return (tail_count + 0.5) / (n + 1)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Apply Holm's step-down correction across a family of challenger comparisons.

    Inputs: Nonempty mapping of comparison names to p-values in [0, 1].
    Outputs: Adjusted probabilities keyed by the original comparison names.
    Why: Account for testing the selected model against multiple challengers.
    Raises: ValueError if the mapping is empty or any p-value is outside [0, 1].
    """
    if not p_values or any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("Supply at least one p-value; every value must be in [0, 1].")
    adjusted = {}
    largest = 0.0
    for rank, (name, value) in enumerate(sorted(p_values.items(), key=lambda item: item[1])):
        largest = min(1.0, max(largest, (len(p_values) - rank) * value))
        adjusted[name] = largest
    return {name: adjusted[name] for name in p_values}


def test_period_diagnostics(agent: pd.DataFrame, baseline: pd.DataFrame,
                            challengers: dict[str, pd.Series], cfg: dict) -> dict:
    """Report fragility and challenger significance during the one-time test audit.

    Inputs: Aligned daily simulator paths, each challenger's values and configuration.
    Outputs: Episode exclusion, raw and adjusted p-values, and bootstrap settings.
    Why: Reuse paths from the release audit without running more candidates on the test years.
    Raises: ValueError from the episode check or the bootstrap when the paths are not aligned or usable.
    """
    gate = cfg["quality_gate"]
    episode = without_largest_episode(agent, baseline, threshold=gate["threshold"], comparison=gate["comparison"])
    weekly = pd.DataFrame({"agent_values": agent["value"], **challengers}).resample("W-FRI").last()
    weekly = weekly.pct_change(fill_method=None).dropna()
    if weekly.empty or not challengers:
        return {"largest_episode": episode, "challenger_significance": {"status": "inconclusive"}}

    draws = cfg["robustness"]["bootstrap_draws"]
    block = min(cfg["robustness"]["bootstrap_block_weeks"], len(weekly))
    seed = cfg["model"]["random_state"]
    comparisons = {name: bootstrap_p_value(weekly["agent_values"], weekly[name], draws, block, seed)
                   for name in challengers}
    return {
        "largest_episode": episode,
        "challenger_significance": {
            "status": "measured", "weekly_returns": len(weekly), "draws": draws,
            "block_weeks": block, "seed": seed, "p_values": comparisons, "holm_adjusted": holm_adjust(comparisons),
        },
    }


def report_robustness(results: dict, versions_tried: int, path: Path, run: object = None) -> None:
    """Persist the robustness diagnostics together with the candidate count.

    Inputs: Named results, total candidate versions evaluated, output path and tracking run.
    Outputs: JSON file and a tracking table with the same results.
    Why: The model card needs both the findings and the number of versions considered.
    Raises: ValueError if versions_tried is below 1; nothing is written then.
    """
    if versions_tried < 1:
        raise ValueError("The robustness report needs at least one evaluated version.")
    report = {**results, "versions_tried": versions_tried}
    utils.save_json(report, path)
    table = pd.DataFrame({"check": list(report), "result": [json.dumps(value) for value in report.values()]})
    tracking.log_table(run, "robustness", table)
    logger.info("Saved robustness report after %d versions", versions_tried)

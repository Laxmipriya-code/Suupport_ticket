"""Rates, comparisons, trends, and target evaluation using shared query primitives."""

from typing import Any

import pandas as pd

from support_ticket_ai.dataset import Dataset
from support_ticket_ai.dates import date_bounds
from support_ticket_ai.query import criteria, measure, records, select
from support_ticket_ai.schemas import Calculation, CompareSpec, RateSpec, TargetSpec, TrendSpec


def rate(dataset: Dataset, spec: RateSpec, as_of: pd.Timestamp) -> Calculation:
    base = select(dataset.frame(as_of), spec.base, as_of)
    matched = select(base, spec.matching, as_of)
    denominator, numerator = len(base), len(matched)
    return Calculation(
        kind="rate",
        summary={
            "numerator": numerator,
            "denominator": denominator,
            "fraction": numerator / denominator if denominator else None,
            "percentage": numerator / denominator * 100 if denominator else None,
        },
        criteria={"base": criteria(spec.base, as_of), "matching": criteria(spec.matching, as_of)},
        warnings=[]
        if denominator
        else ["Percentage is undefined because the base population is empty."],
    )


def compare(dataset: Dataset, spec: CompareSpec, as_of: pd.Timestamp) -> Calculation:
    source = dataset.frame(as_of)
    baseline = select(source, spec.baseline.selection, as_of)
    comparison = select(source, spec.comparison.selection, as_of)
    rows: list[dict[str, Any]] = []
    warnings = []
    for metric in spec.metrics:
        left, right = measure(baseline, metric), measure(comparison, metric)
        a, b = left["value"], right["value"]
        delta = b - a if a is not None and b is not None else None
        change = delta / a * 100 if a not in (None, 0) and delta is not None else None
        if change is None:
            warnings.append(
                f"{metric.key}: percentage change is undefined for missing/zero baseline or missing comparison."
            )
        rows.append(
            {
                "metric": metric.key,
                "baseline": a,
                "comparison": b,
                "baseline_sample_count": left["sample_count"],
                "comparison_sample_count": right["sample_count"],
                "absolute_difference": delta,
                "percentage_change": change,
            }
        )
    return Calculation(
        kind="comparison",
        summary={
            "baseline_name": spec.baseline.name,
            "comparison_name": spec.comparison.name,
            "baseline_tickets": len(baseline),
            "comparison_tickets": len(comparison),
        },
        rows=rows,
        warnings=warnings,
        criteria={
            "baseline": criteria(spec.baseline.selection, as_of),
            "comparison": criteria(spec.comparison.selection, as_of),
        },
    )


def trends(dataset: Dataset, spec: TrendSpec, as_of: pd.Timestamp) -> Calculation:
    source = dataset.frame(as_of)
    frame = select(source, spec.selection, as_of)
    frame = frame[frame[spec.time_field].notna() & frame[spec.time_field].le(as_of)]
    coverage = source[spec.time_field].dropna()
    start = coverage.min() if not coverage.empty else as_of
    end = min(coverage.max(), as_of) if not coverage.empty else as_of
    if spec.selection.date_window:
        if spec.selection.date_window.field != spec.time_field:
            raise ValueError("Trend date window and time_field must use the same timestamp field")
        lower, upper, inclusive = date_bounds(spec.selection.date_window, as_of)
        start = lower if lower is not None else start
        end = (
            min(upper if inclusive else upper - pd.Timedelta(1, unit="ns"), as_of)
            if upper is not None
            else end
        )
    warnings = ["Status-based trends describe recorded snapshot statuses, not historical backlog."]
    if spec.time_field == "resolved_at_estimated":
        warnings.append("Resolution timestamps are estimated from recorded duration.")
    freq = {"day": "D", "week": "W-SUN", "month": "M"}[spec.interval]
    start_period = start.tz_localize(None).to_period(freq)
    end_period = end.tz_localize(None).to_period(freq)
    if end_period.ordinal - start_period.ordinal > 365:
        raise ValueError("Trend requests are limited to 366 buckets; choose a coarser interval")
    periods = list(pd.period_range(start_period, end_period, freq=freq)) if start <= end else []
    bucket_values = frame[spec.time_field].dt.tz_localize(None).dt.to_period(freq)
    rows = []
    for period in periods:
        group = frame[bucket_values.eq(period)]
        computed = measure(group, spec.metric)
        rows.append(
            {
                "period_start": period.start_time.tz_localize(dataset.timezone).isoformat(),
                "value": computed["value"],
                "sample_count": computed["sample_count"],
                "ticket_count": len(group),
            }
        )
    if not coverage.empty and (start < coverage.min() or end > coverage.max()):
        warnings.append(
            "Requested periods extend outside recorded timestamp coverage; empty buckets are not evidence of complete coverage."
        )
    return Calculation(
        kind="trend",
        summary={
            "metric": spec.metric.key,
            "interval": spec.interval,
            "time_field": spec.time_field,
            "matched_tickets": len(frame),
            "total_buckets": len(rows),
            "coverage_start": coverage.min().isoformat() if not coverage.empty else None,
            "coverage_end": coverage.max().isoformat() if not coverage.empty else None,
        },
        rows=rows,
        warnings=warnings,
        criteria=criteria(spec.selection, as_of),
    )


def target_rows(frame: pd.DataFrame, target: str, threshold: float) -> pd.DataFrame:
    result = frame.copy()
    field = "response_time_hrs" if target == "first_response" else "resolution_time_hrs"
    result["target_state"] = "unknown"
    completed = result[field].notna()
    if target == "resolution":
        completed &= result.status.eq("Resolved")
        pending = result.status.isin(["Open", "Escalated"]) & result[field].isna()
        result.loc[pending & result.age_hours.le(threshold), "target_state"] = (
            "pending_within_target"
        )
        result.loc[pending & result.age_hours.gt(threshold), "target_state"] = "pending_overdue"
    result.loc[completed & result[field].le(threshold), "target_state"] = "completed_within_target"
    result.loc[completed & result[field].gt(threshold), "target_state"] = "completed_late"
    result["observed_hours"] = result[field].where(completed)
    if target == "resolution":
        pending_mask = result.target_state.isin(["pending_overdue", "pending_within_target"])
        result.loc[pending_mask, "observed_hours"] = result.loc[pending_mask, "age_hours"]
    result["threshold_hours"] = threshold
    result["excess_hours"] = (result.observed_hours - threshold).clip(lower=0)
    return result


def time_target(dataset: Dataset, spec: TargetSpec, as_of: pd.Timestamp) -> Calculation:
    frame = target_rows(
        select(dataset.frame(as_of), spec.selection, as_of), spec.target, spec.threshold_hours
    )
    counts = {
        state: int(frame.target_state.eq(state).sum())
        for state in [
            "completed_within_target",
            "completed_late",
            "pending_within_target",
            "pending_overdue",
            "unknown",
        ]
    }
    breaches = counts["completed_late"] + counts["pending_overdue"]
    if spec.breaches_only:
        frame = frame[frame.target_state.isin(["completed_late", "pending_overdue"])]
    return Calculation(
        kind="time_target",
        summary={
            "target": spec.target,
            "threshold_hours": spec.threshold_hours,
            "states": counts,
            "breach_count": breaches,
            "matched_tickets": sum(counts.values()),
        },
        rows=records(frame.sort_values(["excess_hours", "ticket_id"], ascending=[False, True])),
        criteria=criteria(spec.selection, as_of),
        warnings=["This target is an analytical threshold, not a contractual SLA."],
    )

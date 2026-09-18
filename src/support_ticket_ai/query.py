"""Allowlisted pandas operations; no expressions, generated code, or SQL execution."""

import json
import math
from typing import Any

import pandas as pd

from support_ticket_ai.dataset import Dataset
from support_ticket_ai.dates import date_bounds, timestamp
from support_ticket_ai.schemas import (
    TIME_COLUMNS,
    AggregateSpec,
    Calculation,
    ListSpec,
    Metric,
    Predicate,
    QualitySpec,
    Selection,
)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def condition(frame: pd.DataFrame, predicate: Predicate, timezone: str) -> pd.Series:
    column = frame[predicate.field]
    value: Any = predicate.value  # Predicate validates compatibility before this pandas boundary.
    if predicate.field in TIME_COLUMNS and value is not None:
        value = (
            [timestamp(str(v), timezone) for v in value]
            if isinstance(value, list)
            else timestamp(str(value), timezone)
        )
    match predicate.op:
        case "is_null":
            mask = column.isna()
        case "not_null":
            mask = column.notna()
        case "eq":
            mask = column.eq(value)
        case "ne":
            mask = column.ne(value) & column.notna()
        case "gt":
            mask = column.gt(value)
        case "gte":
            mask = column.ge(value)
        case "lt":
            mask = column.lt(value)
        case "lte":
            mask = column.le(value)
        case "in":
            mask = column.isin(value)
        case "not_in":
            mask = ~column.isin(value) & column.notna()
        case "contains":
            mask = column.astype("string").str.contains(
                str(value), case=False, regex=False, na=False
            )
    return mask.fillna(False)


def select(frame: pd.DataFrame, selection: Selection, as_of: pd.Timestamp) -> pd.DataFrame:
    mask = frame.created_at.le(as_of)
    for predicate in selection.where:
        mask &= condition(frame, predicate, str(as_of.tz))
    if selection.any_of:
        union = pd.Series(False, index=frame.index)
        for group in selection.any_of:
            group_mask = pd.Series(True, index=frame.index)
            for predicate in group:
                group_mask &= condition(frame, predicate, str(as_of.tz))
            union |= group_mask
        mask &= union
    if selection.date_window:
        window = selection.date_window
        start, end, inclusive = date_bounds(window, as_of)
        column = frame[window.field]
        if start is not None:
            mask &= column.ge(start)
        if end is not None:
            mask &= column.le(end) if inclusive else column.lt(end)
        mask &= column.notna()
    return frame.loc[mask].copy()


def criteria(selection: Selection, as_of: pd.Timestamp) -> dict[str, Any]:
    result = selection.model_dump(mode="json")
    if selection.date_window:
        start, end, inclusive = date_bounds(selection.date_window, as_of)
        result["resolved_date_window"] = {
            "field": selection.date_window.field,
            "start": start.isoformat() if start is not None else None,
            "end": end.isoformat() if end is not None else None,
            "end_inclusive": inclusive,
        }
    return result


def measure(frame: pd.DataFrame, metric: Metric) -> dict[str, Any]:
    if metric.operation == "count":
        return {"value": len(frame), "sample_count": len(frame)}
    values = frame[str(metric.field)].dropna()
    count = len(values)
    if metric.operation == "distinct_count":
        value: int | float | None = int(values.nunique())
    elif not count:
        value = None
    else:
        match metric.operation:
            case "average":
                value = float(values.mean())
            case "sum":
                value = float(values.sum())
            case "min":
                value = float(values.min())
            case "max":
                value = float(values.max())
            case "median":
                value = float(values.median())
            case "percentile":
                assert metric.percentile is not None
                value = float(
                    values.quantile(float(metric.percentile) / 100, interpolation="linear")
                )
        if value is not None and not math.isfinite(value):
            raise ValueError("Calculation produced a non-finite result")
    return {"value": value, "sample_count": count}


def aggregate(dataset: Dataset, spec: AggregateSpec, as_of: pd.Timestamp) -> Calculation:
    frame = select(dataset.frame(as_of), spec.selection, as_of)
    result = Calculation(
        kind="aggregate",
        summary={"matched_tickets": len(frame)},
        criteria=criteria(spec.selection, as_of),
    )
    if not spec.group_by:
        result.summary["metrics"] = {metric.key: measure(frame, metric) for metric in spec.metrics}
        return result
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(spec.group_by, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = dict(zip(spec.group_by, keys, strict=True))
        row["ticket_count"] = len(group)
        for metric in spec.metrics:
            computed = measure(group, metric)
            row[metric.key] = computed["value"]
            row[f"{metric.key}_sample_count"] = computed["sample_count"]
        rows.append(row)
    result.summary["total_groups"] = len(rows)
    if rows:
        table = pd.DataFrame(rows)
        sort_metric = spec.metrics[spec.rank_metric].key
        table = table.sort_values(
            [sort_metric, *spec.group_by],
            ascending=[not spec.descending, *([True] * len(spec.group_by))],
            na_position="last",
            kind="stable",
        )
        if spec.top_n is not None:
            table = table[
                table[sort_metric].rank(method="min", ascending=not spec.descending).le(spec.top_n)
            ]
        result.rows = records(table)
    result.summary["metrics"] = [m.key for m in spec.metrics]
    result.summary["ranking"] = {
        "metric": spec.metrics[spec.rank_metric].key,
        "descending": spec.descending,
        "top_n": spec.top_n,
        "includes_ties": True,
    }
    return result


def list_tickets(dataset: Dataset, spec: ListSpec, as_of: pd.Timestamp) -> Calculation:
    frame = select(dataset.frame(as_of), spec.selection, as_of)
    if spec.search:
        frame = frame[
            frame.issue_summary.str.contains(spec.search, case=False, regex=False, na=False)
        ]
    ordering = [spec.sort_by] + ([] if spec.sort_by == "ticket_id" else ["ticket_id"])
    frame = frame.sort_values(
        ordering,
        ascending=[not spec.descending] + ([True] if len(ordering) == 2 else []),
        kind="stable",
    )
    columns = list(dict.fromkeys(["ticket_id", *spec.columns]))
    return Calculation(
        kind="tickets",
        summary={"matched_tickets": len(frame)},
        rows=records(frame[columns]),
        criteria={**criteria(spec.selection, as_of), "search_spec": spec.model_dump(mode="json")},
    )


def quality_report(dataset: Dataset, spec: QualitySpec, as_of: pd.Timestamp) -> Calculation:
    frame = select(dataset.frame(as_of), spec.selection, as_of)
    before = frame.resolution_time_hrs.lt(frame.response_time_hrs)
    conflict = (frame.status.eq("Resolved") & frame.resolution_time_hrs.isna()) | (
        frame.status.ne("Resolved")
        & (frame.resolution_time_hrs.notna() | frame.customer_rating.notna())
    )
    rows: list[dict[str, Any]] = []
    if spec.warning_type in {"all", "resolution_before_response"}:
        rows.extend(
            {"ticket_id": t, "warning": "resolution_before_response"}
            for t in frame.loc[before, "ticket_id"]
        )
    if spec.warning_type in {"all", "status_conflict"}:
        rows.extend(
            {"ticket_id": t, "warning": "status_conflict"} for t in frame.loc[conflict, "ticket_id"]
        )
    if spec.warning_type in {"all", "missing_values"}:
        for field in ["response_time_hrs", "resolution_time_hrs", "customer_rating"]:
            rows.extend(
                {"ticket_id": t, "warning": "missing_value", "field": field}
                for t in frame.loc[frame[field].isna(), "ticket_id"]
            )
    return Calculation(
        kind="quality",
        summary={
            "matched_tickets": len(frame),
            "resolution_before_response": int(before.sum()),
            "status_conflicts": int(conflict.sum()),
            "missing_values": {
                c: int(frame[c].isna().sum())
                for c in ["response_time_hrs", "resolution_time_hrs", "customer_rating"]
            },
            "note": "Missing resolution/rating values on unresolved tickets are expected, not corruption.",
        },
        rows=rows,
        criteria=criteria(spec.selection, as_of),
    )

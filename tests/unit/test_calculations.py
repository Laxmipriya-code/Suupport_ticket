import hashlib
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from support_ticket_ai import analytics, anomalies, query
from support_ticket_ai.dataset import Dataset
from support_ticket_ai.dates import date_bounds, reference_time
from support_ticket_ai.schemas import (
    AggregateSpec,
    AnomalySpec,
    Cohort,
    CompareSpec,
    DateWindow,
    ListSpec,
    Metric,
    Predicate,
    QualitySpec,
    RateSpec,
    Selection,
    TargetSpec,
    TrendSpec,
)


def selection(**values):
    return Selection(
        where=[
            Predicate(field=k, op="in" if isinstance(v, list) else "eq", value=v)
            for k, v in values.items()
        ]
    )


def test_source_unchanged(dataset):
    original = Path(__file__).resolve().parents[3] / "AI Intern - Assessment/support_tickets.csv"
    if original.exists():
        assert dataset.fingerprint == hashlib.sha256(original.read_bytes()).hexdigest()
    assert dataset.info().summary["total_tickets"] == 500


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"status": "Open"}, 111),
        ({"status": ["Open", "Escalated"]}, 173),
        ({"priority": "Critical", "status": ["Open", "Escalated"]}, 31),
        ({"category": "Technical"}, 152),
        ({"status": "Resolved"}, 327),
    ],
)
def test_reference_counts(dataset, as_of, filters, expected):
    result = query.aggregate(dataset, AggregateSpec(selection=selection(**filters)), as_of)
    assert result.summary["metrics"]["count"]["value"] == expected


def test_rating_and_group_ranking(dataset, as_of):
    metric = Metric(operation="average", field="customer_rating")
    result = query.aggregate(
        dataset, AggregateSpec(selection=selection(category="Technical"), metrics=[metric]), as_of
    )
    assert result.summary["metrics"][metric.key]["value"] == pytest.approx(3.7403846153846154)
    ranked = query.aggregate(
        dataset,
        AggregateSpec(metrics=[metric], group_by=["agent_id"], descending=False, top_n=1),
        as_of,
    )
    assert ranked.rows[0]["agent_id"] == "AGT-08"
    assert ranked.rows[0][metric.key] == pytest.approx(3.48)
    assert ranked.rows[0][metric.key + "_sample_count"] == 25


def test_tied_monthly_winners_and_estimated_resolution(dataset, as_of):
    sel = selection(status="Resolved")
    sel.date_window = DateWindow(field="resolved_at_estimated", period="this_month")
    result = query.aggregate(
        dataset, AggregateSpec(selection=sel, group_by=["agent_id"], top_n=1), as_of
    )
    assert [(r["agent_id"], r["count"]) for r in result.rows] == [("AGT-01", 15), ("AGT-12", 15)]


def test_or_filters_null_and_literal_search(dataset, as_of):
    sel = Selection(
        where=[Predicate(field="priority", op="eq", value="Critical")],
        any_of=[
            [Predicate(field="status", op="eq", value="Open")],
            [Predicate(field="status", op="eq", value="Escalated")],
        ],
    )
    assert (
        query.aggregate(dataset, AggregateSpec(selection=sel), as_of).summary["matched_tickets"]
        == 31
    )
    nulls = Selection(where=[Predicate(field="resolution_time_hrs", op="is_null")])
    assert (
        query.aggregate(dataset, AggregateSpec(selection=nulls), as_of).summary["matched_tickets"]
        == 173
    )
    assert query.list_tickets(dataset, ListSpec(search=".*"), as_of).rows == []
    assert (
        query.list_tickets(dataset, ListSpec(search="refund"), as_of).summary["matched_tickets"]
        == 20
    )


def test_rates_and_empty_denominator(dataset, as_of):
    result = analytics.rate(
        dataset,
        RateSpec(
            base=selection(priority="Critical"), matching=selection(status=["Open", "Escalated"])
        ),
        as_of,
    )
    assert result.summary == {
        "numerator": 31,
        "denominator": 55,
        "fraction": 31 / 55,
        "percentage": 31 / 55 * 100,
    }
    empty = analytics.rate(
        dataset, RateSpec(base=selection(ticket_id="NO-TICKET"), matching=Selection()), as_of
    )
    assert empty.summary["percentage"] is None
    assert empty.warnings


def test_comparison_and_zero_baseline(dataset, as_of):
    result = analytics.compare(
        dataset,
        CompareSpec(
            baseline=Cohort(name="Open", selection=selection(status="Open")),
            comparison=Cohort(name="Resolved", selection=selection(status="Resolved")),
        ),
        as_of,
    )
    assert result.rows[0]["absolute_difference"] == 216
    assert result.rows[0]["percentage_change"] == pytest.approx(216 / 111 * 100)
    zero = analytics.compare(
        dataset,
        CompareSpec(
            baseline=Cohort(name="Empty", selection=selection(ticket_id="none")),
            comparison=Cohort(name="All"),
        ),
        as_of,
    )
    assert zero.rows[0]["percentage_change"] is None


def test_reference_targets_anomalies_and_quality(dataset, as_of):
    target = analytics.time_target(
        dataset, TargetSpec(selection=selection(priority="Critical"), threshold_hours=12), as_of
    )
    assert target.summary["breach_count"] == 34
    assert len(target.rows) == 34
    found = anomalies.detect(dataset, AnomalySpec(), as_of)
    assert found.summary["long_resolution_count"] == 21
    assert found.summary["resolution_threshold_hours"] == pytest.approx(48.15)
    assert found.summary["overdue_high_priority_count"] == 80
    filtered = anomalies.detect(
        dataset, AnomalySpec(selection=selection(category="Billing")), as_of
    )
    assert (
        filtered.summary["resolution_threshold_hours"]
        == found.summary["resolution_threshold_hours"]
    )
    quality = query.quality_report(dataset, QualitySpec(), as_of)
    assert quality.summary["resolution_before_response"] == 28
    assert quality.summary["missing_values"]["customer_rating"] == 173


def test_targets_exact_boundary_pending_and_unknown():
    frame = pd.DataFrame(
        {
            "status": ["Resolved", "Resolved", "Open", "Escalated", "Resolved"],
            "resolution_time_hrs": [24, 24.1, None, None, None],
            "response_time_hrs": [1, None, 2, 3, 4],
            "age_hours": [30, 30, 24, 24.1, 30],
        }
    )
    result = analytics.target_rows(frame, "resolution", 24)
    assert result.target_state.tolist() == [
        "completed_within_target",
        "completed_late",
        "pending_within_target",
        "pending_overdue",
        "unknown",
    ]
    assert analytics.target_rows(frame, "first_response", 2).target_state.iloc[1] == "unknown"


def test_metrics_empty_percentile_and_no_mutation(dataset, as_of):
    before = dataset.frame(as_of)
    result = query.aggregate(
        dataset,
        AggregateSpec(
            selection=selection(ticket_id="none"),
            metrics=[Metric(operation="average", field="customer_rating")],
        ),
        as_of,
    )
    assert result.summary["metrics"]["average_customer_rating"] == {
        "value": None,
        "sample_count": 0,
    }
    toy = pd.DataFrame({"response_time_hrs": [0.0, 10.0, 20.0, 30.0]})
    assert (
        query.measure(
            toy, Metric(operation="percentile", field="response_time_hrs", percentile=25)
        )["value"]
        == 7.5
    )
    query.list_tickets(dataset, ListSpec(), as_of)
    pd.testing.assert_frame_equal(before, dataset.frame(as_of))


def test_trends_zero_buckets_and_coverage(dataset, as_of):
    result = analytics.trends(dataset, TrendSpec(interval="week"), as_of)
    assert sum(row["value"] for row in result.rows) == 500
    assert all(pd.Timestamp(row["period_start"]).weekday() == 0 for row in result.rows)
    spec = TrendSpec(
        selection=Selection(date_window=DateWindow(start="2024-04-01", end="2024-04-04")),
        interval="day",
    )
    later = pd.Timestamp("2024-04-05T00:00:00Z")
    empty = analytics.trends(dataset, spec, later)
    assert [r["value"] for r in empty.rows] == [0, 0, 0]
    assert any("coverage" in w for w in empty.warnings)
    spec.metric = Metric(operation="average", field="customer_rating")
    assert all(r["value"] is None for r in analytics.trends(dataset, spec, later).rows)


def test_date_ranges_timezone_and_week():
    ref = pd.Timestamp("2024-04-01T10:00:00+05:30")
    start, end, inclusive = date_bounds(DateWindow(period="last_week"), ref)
    assert start.isoformat() == "2024-03-25T00:00:00+05:30"
    assert end.isoformat() == "2024-04-01T00:00:00+05:30"
    assert not inclusive
    start, end, inclusive = date_bounds(DateWindow(period="this_month"), ref)
    assert start.day == 1 and end == ref and inclusive
    assert reference_time(pd.Timestamp("2024-01-01T00:00:00Z"), "Asia/Kolkata").hour == 5
    with pytest.raises(ValueError):
        date_bounds(DateWindow(start="2024-02-01", end="2024-01-01"), ref)


@pytest.mark.parametrize(
    "bad",
    [
        {"field": "password", "op": "eq", "value": "x"},
        {"field": "status", "op": "eq", "value": "unresolved"},
        {"field": "customer_rating", "op": "gt", "value": "a"},
        {"field": "status", "op": "in", "value": "Open"},
        {"field": "status", "op": "gt", "value": "Open"},
        {"field": "customer_rating", "op": "eq", "value": float("inf")},
        {"field": "customer_rating", "op": "eq", "value": True},
    ],
)
def test_invalid_filters(bad):
    with pytest.raises(ValidationError):
        Predicate.model_validate(bad)


@pytest.mark.parametrize(
    "bad",
    [
        {"operation": "average", "field": "issue_summary"},
        {"operation": "sum", "field": "customer_rating"},
        {"operation": "percentile", "field": "resolution_time_hrs"},
        {"operation": "count", "field": "ticket_id"},
    ],
)
def test_invalid_metrics(bad):
    with pytest.raises(ValidationError):
        Metric.model_validate(bad)


def test_filter_budget_includes_both_populations():
    many = Selection(where=[Predicate(field="status", op="eq", value="Open")] * 11)
    with pytest.raises(ValidationError):
        RateSpec(base=many, matching=many)
    with pytest.raises(ValidationError):
        CompareSpec(
            baseline=Cohort(name="a", selection=many), comparison=Cohort(name="b", selection=many)
        )


def test_structural_data_errors(tmp_path):
    source = Path(__file__).resolve().parents[2] / "data/support_tickets.csv"
    frame = pd.read_csv(source, dtype=str, keep_default_na=False).head(5)
    for column, value in [
        ("created_at", "bad date"),
        ("priority", "Urgent"),
        ("response_time_hrs", "-1"),
        ("customer_rating", "2.5"),
        ("resolution_time_hrs", "inf"),
    ]:
        invalid = frame.copy()
        invalid.loc[0, column] = value
        path = tmp_path / "bad.csv"
        invalid.to_csv(path, index=False)
        with pytest.raises(ValueError, match=column):
            Dataset(path)
    duplicate = frame.copy()
    duplicate.loc[1, "ticket_id"] = duplicate.loc[0, "ticket_id"]
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        Dataset(path)
    frame.drop(columns=["status"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns"):
        Dataset(path)


@pytest.mark.parametrize("size", [3, 5])
def test_insufficient_and_zero_iqr_baselines(tmp_path, size):
    source = Path(__file__).resolve().parents[2] / "data/support_tickets.csv"
    frame = pd.read_csv(source).head(size)
    frame["status"] = "Resolved"
    frame["resolution_time_hrs"] = 10
    path = tmp_path / "flat.csv"
    frame.to_csv(path, index=False)
    result = anomalies.detect(
        Dataset(path), AnomalySpec(rule="long_resolution"), pd.Timestamp("2025-01-01T00:00Z")
    )
    assert result.summary["resolution_threshold_hours"] is None
    assert result.warnings

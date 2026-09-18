"""Closed, bounded contracts shared by the API and calculation tools."""

import math
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

Column = Literal[
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
    "resolved_at_estimated",
    "age_hours",
]
GroupColumn = Literal["agent_id", "category", "priority", "status"]
TimeColumn = Literal["created_at", "resolved_at_estimated"]
NUMERIC_COLUMNS = {"response_time_hrs", "resolution_time_hrs", "customer_rating", "age_hours"}
TIME_COLUMNS = {"created_at", "resolved_at_estimated"}
ENUM_VALUES = {
    "category": ["Billing", "Technical", "General"],
    "priority": ["Low", "Medium", "High", "Critical"],
    "status": ["Open", "Resolved", "Escalated"],
}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Predicate(Contract):
    field: Column
    op: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains", "is_null", "not_null"
    ]
    value: str | StrictFloat | list[str | StrictFloat] | None = None

    @model_validator(mode="after")
    def compatible(self) -> Self:
        if self.op in {"is_null", "not_null"}:
            if self.value is not None:
                raise ValueError("Null checks must not include a value")
            return self
        if self.value is None:
            raise ValueError("A filter value is required")
        if (self.op in {"in", "not_in"}) != isinstance(self.value, list):
            raise ValueError("Membership filters require a list; other filters require a scalar")
        values = self.value if isinstance(self.value, list) else [self.value]
        if not values or len(values) > 50:
            raise ValueError("Provide 1 to 50 filter values")
        if any(isinstance(v, str) and len(v) > 200 for v in values):
            raise ValueError("Filter strings are limited to 200 characters")
        if self.field in NUMERIC_COLUMNS:
            if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
                raise ValueError("Numeric fields require finite numeric values")
            if self.op == "contains":
                raise ValueError("Contains is a text operation")
        elif any(not isinstance(v, str) for v in values):
            raise ValueError("Text and timestamp fields require strings")
        if (
            self.op in {"gt", "gte", "lt", "lte"}
            and self.field not in NUMERIC_COLUMNS | TIME_COLUMNS
        ):
            raise ValueError("Ordering comparisons require a number or timestamp")
        if self.field in TIME_COLUMNS and self.op == "contains":
            raise ValueError("Contains is not a timestamp operation")
        if self.field in ENUM_VALUES:
            canonical = {v.lower(): v for v in ENUM_VALUES[self.field]}
            if any(str(v).lower() not in canonical for v in values):
                raise ValueError(f"Invalid value for {self.field}")
            normalized: list[str | float] = [canonical[str(v).lower()] for v in values]
            self.value = normalized if isinstance(self.value, list) else normalized[0]
        return self


class DateWindow(Contract):
    field: TimeColumn = "created_at"
    period: (
        Literal[
            "today", "this_week", "last_week", "this_month", "last_month", "this_year", "last_year"
        ]
        | None
    ) = None
    start: datetime | None = None
    end: datetime | None = None

    @model_validator(mode="after")
    def exclusive_modes(self) -> Self:
        if self.period and (self.start or self.end):
            raise ValueError("Use a relative period OR explicit timestamps")
        if not self.period and not (self.start or self.end):
            raise ValueError("Specify a period or at least one timestamp")
        return self


class Selection(Contract):
    """All `where` predicates AND at least one `any_of` group (when supplied)."""

    where: list[Predicate] = Field(default_factory=list, max_length=20)
    any_of: list[list[Predicate]] = Field(default_factory=list, max_length=10)
    date_window: DateWindow | None = None

    @property
    def condition_count(self) -> int:
        return len(self.where) + sum(map(len, self.any_of))

    @model_validator(mode="after")
    def bounded_filters(self) -> Self:
        if self.condition_count > 20:
            raise ValueError("At most 20 conditions are allowed")
        if any(not group for group in self.any_of):
            raise ValueError("OR groups must not be empty")
        return self


class Metric(Contract):
    operation: Literal[
        "count", "distinct_count", "average", "sum", "min", "max", "median", "percentile"
    ]
    field: Column | None = None
    percentile: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def valid_metric(self) -> Self:
        if self.operation == "count" and self.field is not None:
            raise ValueError("Count measures tickets and must not specify a field")
        if self.operation != "count" and self.field is None:
            raise ValueError("This metric requires a field")
        if self.operation not in {"count", "distinct_count"} and self.field not in NUMERIC_COLUMNS:
            raise ValueError("This metric requires a numeric field")
        if self.operation == "sum" and self.field not in {
            "response_time_hrs",
            "resolution_time_hrs",
        }:
            raise ValueError("Only recorded durations can be summed")
        if (self.operation == "percentile") != (self.percentile is not None):
            raise ValueError("Provide percentile only for a percentile metric")
        return self

    @property
    def key(self) -> str:
        parts: list[str | None] = [self.operation, self.field]
        if self.percentile is not None:
            parts.append(str(self.percentile))
        return "_".join(p for p in parts if p)


class AggregateSpec(Contract):
    selection: Selection = Field(default_factory=Selection)
    metrics: list[Metric] = Field(
        default_factory=lambda: [Metric(operation="count")], min_length=1, max_length=5
    )
    group_by: list[GroupColumn] = Field(default_factory=list, max_length=2)
    rank_metric: int = Field(default=0, ge=0)
    descending: bool = True
    top_n: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def unique_metrics(self) -> Self:
        if self.rank_metric >= len(self.metrics):
            raise ValueError("rank_metric must identify an existing metric")
        if len({m.key for m in self.metrics}) != len(self.metrics):
            raise ValueError("Metrics must be unique")
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("Grouping fields must be unique")
        return self


def default_columns() -> list[Column]:
    return [
        "ticket_id",
        "created_at",
        "category",
        "priority",
        "status",
        "agent_id",
        "issue_summary",
    ]


class ListSpec(Contract):
    selection: Selection = Field(default_factory=Selection)
    search: str | None = Field(default=None, min_length=1, max_length=200)
    columns: list[Column] = Field(
        default_factory=default_columns,
        min_length=1,
        max_length=12,
    )
    sort_by: Column = "created_at"
    descending: bool = True


class RateSpec(Contract):
    base: Selection = Field(default_factory=Selection)
    matching: Selection

    @model_validator(mode="after")
    def bounded_total(self) -> Self:
        if self.base.condition_count + self.matching.condition_count > 20:
            raise ValueError("At most 20 conditions across both populations are allowed")
        return self


class TrendSpec(Contract):
    selection: Selection = Field(default_factory=Selection)
    time_field: TimeColumn = "created_at"
    interval: Literal["day", "week", "month"] = "week"
    metric: Metric = Field(default_factory=lambda: Metric(operation="count"))


class Cohort(Contract):
    name: str = Field(min_length=1, max_length=50)
    selection: Selection = Field(default_factory=Selection)


class CompareSpec(Contract):
    baseline: Cohort
    comparison: Cohort
    metrics: list[Metric] = Field(
        default_factory=lambda: [Metric(operation="count")], min_length=1, max_length=5
    )

    @model_validator(mode="after")
    def bounded_total(self) -> Self:
        if self.baseline.selection.condition_count + self.comparison.selection.condition_count > 20:
            raise ValueError("At most 20 conditions across both populations are allowed")
        if len({m.key for m in self.metrics}) != len(self.metrics):
            raise ValueError("Metrics must be unique")
        return self


class TargetSpec(Contract):
    selection: Selection = Field(default_factory=Selection)
    target: Literal["first_response", "resolution"] = "resolution"
    threshold_hours: float = Field(gt=0, le=87600)
    breaches_only: bool = True


class AnomalySpec(Contract):
    selection: Selection = Field(default_factory=Selection)
    rule: Literal["all", "long_resolution", "overdue_high_priority"] = "all"


class QualitySpec(Contract):
    selection: Selection = Field(default_factory=Selection)
    warning_type: Literal[
        "all", "missing_values", "resolution_before_response", "status_conflict"
    ] = "all"


class DatasetInfoSpec(Contract):
    section: Literal["overview", "schema", "definitions"] = "overview"


class AgentOutcome(Contract):
    """Select successful tool results; never write or calculate the answer yourself."""

    status: Literal["success", "clarification", "unsupported"]
    result_ids: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Copy top-level result_id values from successful tool responses. NEVER ticket_id or agent_id values from preview rows. Each calculation has ONE result_id even when it contains many tickets.",
    )
    message: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def evidence_required(self) -> Self:
        if self.status == "success" and not self.result_ids:
            raise ValueError("Success requires at least one tool result ID")
        if self.status != "success" and (self.result_ids or not self.message.strip()):
            raise ValueError("Clarification/unsupported requires a message and no result IDs")
        if len(set(self.result_ids)) != len(self.result_ids):
            raise ValueError("Result IDs must be unique")
        return self


class Page(Contract):
    offset: int = Field(default=0, ge=0, le=1000000)
    limit: int = Field(default=50, ge=1, le=500)


class QueryRequest(Page):
    question: str = Field(min_length=1, max_length=2000)
    as_of: datetime | None = None

    @model_validator(mode="after")
    def nonblank(self) -> Self:
        self.question = self.question.strip()
        if not self.question:
            raise ValueError("Question must not be blank")
        return self


class SearchRequest(Page):
    spec: ListSpec = Field(default_factory=ListSpec)
    as_of: datetime | None = None


class Calculation(Contract):
    kind: str
    summary: dict[str, Any]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    criteria: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ResultView(Calculation):
    result_id: str
    total_rows: int
    offset: int
    limit: int
    truncated: bool


class QueryResponse(Contract):
    request_id: str
    status: Literal["success", "clarification", "unsupported"]
    answer: str
    results: list[ResultView]
    as_of: datetime
    dataset_fingerprint: str
    warnings: list[str]
    tools_used: list[dict[str, Any]]

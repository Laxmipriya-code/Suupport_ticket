"""Small, typed tools sharing the same calculation services as the REST API."""

from typing import Any

from langchain.tools import ToolRuntime, tool

from support_ticket_ai import analytics, anomalies, query
from support_ticket_ai.agent.context import RequestContext
from support_ticket_ai.schemas import (
    AggregateSpec,
    AnomalySpec,
    CompareSpec,
    DatasetInfoSpec,
    ListSpec,
    QualitySpec,
    RateSpec,
    TargetSpec,
    TrendSpec,
)


@tool
async def get_dataset_info(
    spec: DatasetInfoSpec, runtime: ToolRuntime[RequestContext]
) -> dict[str, Any]:
    """Describe dataset coverage, schema/allowed values, or definitions. Returns no ticket rows."""
    return runtime.context.publish(runtime.context.dataset.info(spec.section))


@tool
async def get_data_quality_report(
    spec: QualitySpec, runtime: ToolRuntime[RequestContext]
) -> dict[str, Any]:
    """Report missing values and inconsistent records. Missing unresolved durations/ratings are expected."""
    ctx = runtime.context
    return ctx.publish(query.quality_report(ctx.dataset, spec, ctx.as_of))


@tool
async def aggregate_tickets(
    spec: AggregateSpec, runtime: ToolRuntime[RequestContext]
) -> dict[str, Any]:
    """Compute metrics, distributions, or agent rankings. Count uses no field; averages report sample sizes. top_n retains ties."""
    ctx = runtime.context
    return ctx.publish(query.aggregate(ctx.dataset, spec, ctx.as_of))


@tool
async def list_tickets(spec: ListSpec, runtime: ToolRuntime[RequestContext]) -> dict[str, Any]:
    """Find ticket records by filters, ID, or literal issue-summary keyword search. Full rows go to UI, only a small preview to you."""
    ctx = runtime.context
    return ctx.publish(query.list_tickets(ctx.dataset, spec, ctx.as_of))


@tool
async def calculate_ticket_rate(
    spec: RateSpec, runtime: ToolRuntime[RequestContext]
) -> dict[str, Any]:
    """Compute a percentage: denominator is base, numerator is base AND matching. Use for proportions, not period comparisons."""
    ctx = runtime.context
    return ctx.publish(analytics.rate(ctx.dataset, spec, ctx.as_of))


@tool
async def get_ticket_trends(
    spec: TrendSpec, runtime: ToolRuntime[RequestContext]
) -> dict[str, Any]:
    """Compute day/week/month time series. Status filters describe recorded statuses, never historical backlog. Limit 366 buckets."""
    ctx = runtime.context
    return ctx.publish(analytics.trends(ctx.dataset, spec, ctx.as_of))


@tool
async def compare_ticket_metrics(
    spec: CompareSpec, runtime: ToolRuntime[RequestContext]
) -> dict[str, Any]:
    """Compare two filtered populations or periods, calculating differences and percentage change against baseline."""
    ctx = runtime.context
    return ctx.publish(analytics.compare(ctx.dataset, spec, ctx.as_of))


@tool
async def check_time_target(
    spec: TargetSpec, runtime: ToolRuntime[RequestContext]
) -> dict[str, Any]:
    """Check an explicit hour threshold for first response or resolution. Includes completed-late and pending-overdue; pending within target is not success."""
    ctx = runtime.context
    return ctx.publish(analytics.time_target(ctx.dataset, spec, ctx.as_of))


@tool
async def detect_anomalies(
    spec: AnomalySpec, runtime: ToolRuntime[RequestContext]
) -> dict[str, Any]:
    """Find long resolution outliers (global Q3+1.5 IQR) or unresolved High/Critical older than 24h. For anomalies 'this week', filter estimated resolution time for long_resolution."""
    ctx = runtime.context
    return ctx.publish(anomalies.detect(ctx.dataset, spec, ctx.as_of))


TOOLS = [
    get_dataset_info,
    get_data_quality_report,
    aggregate_tickets,
    list_tickets,
    calculate_ticket_rate,
    get_ticket_trends,
    compare_ticket_metrics,
    check_time_target,
    detect_anomalies,
]

"""Public answers are rendered from backend evidence, never copied model arithmetic."""

import json
from typing import Any

from support_ticket_ai.agent.context import RequestContext
from support_ticket_ai.dates import time_warnings
from support_ticket_ai.schemas import AgentOutcome, Calculation, Page, QueryResponse, ResultView


def display(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def format_calculation(result: Calculation) -> str:
    summary = result.summary
    match result.kind:
        case "aggregate":
            if isinstance(summary["metrics"], dict):
                return (
                    "; ".join(
                        f"{name.replace('_', ' ')}: {display(metric['value'])} (sample count: {metric['sample_count']})"
                        for name, metric in summary["metrics"].items()
                    )
                    + "."
                )
            if not result.rows:
                return "No groups have a matching measured value."
            metric_name = summary["ranking"]["metric"]
            labels = []
            for row in result.rows[:5]:
                label = ", ".join(
                    f"{k}: {v}"
                    for k, v in row.items()
                    if k in {"agent_id", "category", "priority", "status"}
                )
                labels.append(
                    f"{label} — {metric_name.replace('_', ' ')}: {display(row[metric_name])} (sample count: {row[metric_name + '_sample_count']})"
                )
            return (
                "; ".join(labels) + f". {len(result.rows)} matching groups; see the evidence table."
            )
        case "tickets":
            return f"Found {summary['matched_tickets']} matching tickets."
        case "rate":
            if summary["percentage"] is None:
                return "The rate is undefined because the base population is empty."
            return f"{summary['numerator']} of {summary['denominator']} tickets match ({display(summary['percentage'])}%)."
        case "time_target":
            return f"{summary['breach_count']} tickets exceed the {display(summary['threshold_hours'])}-hour {summary['target'].replace('_', ' ')} target. Pending tickets within target are not counted as completed."
        case "anomalies":
            return f"Found {summary['long_resolution_count']} long-resolution flags and {summary['overdue_high_priority_count']} overdue High/Critical flags across {summary['unique_ticket_count']} unique tickets."
        case "quality":
            missing = "; ".join(
                f"{field.replace('_', ' ')}: {count}"
                for field, count in summary["missing_values"].items()
            )
            return f"Found {summary['resolution_before_response']} timing inconsistencies and {summary['status_conflicts']} status conflicts in {summary['matched_tickets']} tickets. Missing measurements: {missing}. Missing resolution durations and ratings are expected on unresolved tickets."
        case "trend":
            return f"Calculated {summary['metric'].replace('_', ' ')} across {summary['total_buckets']} {summary['interval']} buckets from {summary['matched_tickets']} tickets. See the series for exact values."
        case "comparison":
            values = "; ".join(
                f"{row['metric']}: {display(row['baseline'])} vs {display(row['comparison'])}; difference {display(row['absolute_difference'])}; change {display(row['percentage_change'])}{'%' if row['percentage_change'] is not None else ''}"
                for row in result.rows
            )
            return f"{summary['baseline_name']} versus {summary['comparison_name']}: {values}."
        case "dataset_info":
            if "total_tickets" in summary:
                return f"The dataset contains {summary['total_tickets']} tickets, created between {summary['date_start']} and {summary['date_end']}."
            return "Dataset schema and definitions are included in the supporting result."
        case _:
            return "Calculation completed; see the supporting result."


def result_view(result_id: str, result: Calculation, page: Page) -> ResultView:
    return ResultView(
        **{**result.model_dump(), "rows": result.rows[page.offset : page.offset + page.limit]},
        result_id=result_id,
        total_rows=len(result.rows),
        offset=page.offset,
        limit=page.limit,
        truncated=page.offset > 0 or page.offset + page.limit < len(result.rows),
    )


def build_response(outcome: AgentOutcome, ctx: RequestContext, page: Page) -> QueryResponse:
    selected = [ctx.results[rid] for rid in outcome.result_ids]
    estimated = any(
        "resolved_at_estimated" in json.dumps(r.criteria)
        or r.summary.get("time_field") == "resolved_at_estimated"
        for r in selected
    )
    warnings = time_warnings(ctx.as_of, ctx.dataset.latest_created, estimated)
    warnings.extend(warning for result in selected for warning in result.warnings)
    answer = (
        "\n\n".join(format_calculation(result) for result in selected)
        if outcome.status == "success"
        else outcome.message
    )
    return QueryResponse(
        request_id=ctx.request_id,
        status=outcome.status,
        answer=answer,
        results=[result_view(rid, ctx.results[rid], page) for rid in outcome.result_ids],
        as_of=ctx.as_of.to_pydatetime(),
        dataset_fingerprint=ctx.dataset.fingerprint,
        warnings=list(dict.fromkeys(warnings)),
        tools_used=ctx.tools_used,
    )

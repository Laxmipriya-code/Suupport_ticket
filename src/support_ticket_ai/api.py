"""HTTP endpoints reuse the same deterministic services as agent tools."""

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request

from support_ticket_ai.agent.context import RequestContext
from support_ticket_ai.agent.execution import invoke_agent
from support_ticket_ai.anomalies import detect
from support_ticket_ai.dates import reference_time
from support_ticket_ai.query import list_tickets, quality_report
from support_ticket_ai.responses import build_response, result_view
from support_ticket_ai.schemas import (
    AgentOutcome,
    AnomalySpec,
    DateWindow,
    Page,
    Predicate,
    QualitySpec,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    Selection,
)

router = APIRouter()


def context(request: Request, as_of: datetime | None = None) -> RequestContext:
    return RequestContext(
        dataset=request.app.state.dataset,
        as_of=reference_time(as_of, request.app.state.settings.data_timezone),
        request_id=request.state.request_id,
    )


@router.get("/health")
def health(request: Request) -> dict:
    configured = request.app.state.settings.llm_configured
    return {
        "status": "ok" if configured else "degraded",
        "dataset_loaded": True,
        "llm_configured": configured,
        "llm_connectivity": "not_checked",
    }


@router.get("/api/v1/stats")
def stats(request: Request) -> dict:
    ctx = context(request)
    return {
        **ctx.dataset.info().summary,
        "dataset_fingerprint": ctx.dataset.fingerprint,
        "data_quality": quality_report(ctx.dataset, QualitySpec(), ctx.as_of).summary,
    }


@router.post("/api/v1/query", response_model=QueryResponse)
async def ask(body: QueryRequest, request: Request) -> QueryResponse:
    ctx = context(request, body.as_of)
    outcome = await invoke_agent(
        request.app.state.agent, body.question, ctx, request.app.state.settings.request_timeout
    )
    return build_response(outcome, ctx, body)


@router.post("/api/v1/tickets/search")
def search(body: SearchRequest, request: Request) -> dict:
    ctx = context(request, body.as_of)
    calculation = list_tickets(ctx.dataset, body.spec, ctx.as_of)
    return {
        "request_id": ctx.request_id,
        "as_of": ctx.as_of.isoformat(),
        "dataset_fingerprint": ctx.dataset.fingerprint,
        "result": result_view("search", calculation, body).model_dump(mode="json"),
    }


@router.get("/api/v1/anomalies", response_model=QueryResponse)
def anomalies(
    request: Request,
    rule: Literal["all", "long_resolution", "overdue_high_priority"] = "all",
    as_of: datetime | None = None,
    category: Literal["Billing", "Technical", "General"] | None = None,
    priority: Literal["Low", "Medium", "High", "Critical"] | None = None,
    ticket_id: Annotated[str | None, Query(max_length=200)] = None,
    start: datetime | None = None,
    end: datetime | None = None,
    date_field: Literal["created_at", "resolved_at_estimated"] = "created_at",
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> QueryResponse:
    ctx = context(request, as_of)
    filters = []
    if category:
        filters.append(Predicate(field="category", op="eq", value=category))
    if priority:
        filters.append(Predicate(field="priority", op="eq", value=priority))
    if ticket_id:
        filters.append(Predicate(field="ticket_id", op="eq", value=ticket_id))
    window = DateWindow(field=date_field, start=start, end=end) if start or end else None
    spec = AnomalySpec(rule=rule, selection=Selection(where=filters, date_window=window))
    ctx.results["anomalies"] = detect(ctx.dataset, spec, ctx.as_of)
    return build_response(
        AgentOutcome(status="success", result_ids=["anomalies"]),
        ctx,
        Page(offset=offset, limit=limit),
    )

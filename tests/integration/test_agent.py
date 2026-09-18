import asyncio
import json

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr

from support_ticket_ai.agent.context import ApplicationError, RequestContext, encoded
from support_ticket_ai.agent.execution import MODEL_TOOLS, build_agent, invoke_agent
from support_ticket_ai.agent.tools import TOOLS
from support_ticket_ai.responses import build_response
from support_ticket_ai.schemas import Calculation, Page
from tests.fakes import ToolModel


async def run_model(model, dataset, as_of, settings, question="Count tickets"):
    ctx = RequestContext(dataset, as_of)
    outcome = await invoke_agent(build_agent(settings, model), question, ctx)
    return build_response(outcome, ctx, Page(limit=500)), ctx


async def test_tool_call_and_evidence_answer(dataset, as_of, settings):
    model = ToolModel(
        plan=[
            (
                "aggregate_tickets",
                {
                    "spec": {
                        "selection": {"where": [{"field": "status", "op": "eq", "value": "Open"}]}
                    }
                },
            )
        ]
    )
    response, ctx = await run_model(model, dataset, as_of, settings)
    assert "111" in response.answer
    assert ctx.model_turns == 2 and ctx.tool_calls == 1
    assert response.results[0].summary["metrics"]["count"]["value"] == 111


async def test_raw_rows_stay_in_backend(dataset, as_of, settings):
    model = ToolModel(plan=[("list_tickets", {"spec": {}})])
    response, ctx = await run_model(model, dataset, as_of, settings, "Show all tickets")
    assert len(response.results[0].rows) == 500
    messages = json.dumps(
        [[m.model_dump() for m in batch] for batch in model.captured], default=str
    )
    assert not any(summary in messages for summary in dataset.frame(as_of).issue_summary.unique())
    observed = [m for m in model.captured[-1] if isinstance(m, ToolMessage)]
    result = json.loads(observed[0].content)
    assert len(result["preview"]) <= 5 and result["truncated"]
    assert len(observed[0].content.encode()) <= 4096
    assert ctx.output_bytes <= 8192
    assert (
        len(model.captured[0]) == 2
    )  # Static instructions/reference time and the user's question.
    assert not any(isinstance(message, ToolMessage) for message in model.captured[0])


def test_total_output_budget(dataset, as_of):
    ctx = RequestContext(dataset, as_of)
    for _ in range(3):
        result = ctx.publish(
            Calculation(
                kind="aggregate",
                summary={"description": "x" * 3000},
                rows=[{"agent_id": "a", "metric": 1}] * 50,
            )
        )
        assert len(encoded(result)) <= 4096
    assert ctx.output_bytes <= 8192
    assert len(ctx.results) == 3


async def test_unknown_tool_and_bad_arguments_can_be_corrected(dataset, as_of, settings):
    model = ToolModel(
        plan=[
            ("read_file", {"path": "/etc/passwd"}),
            (
                "aggregate_tickets",
                {"spec": {"metrics": [{"operation": "average", "field": "issue_summary"}]}},
            ),
            ("aggregate_tickets", {"spec": {}}),
        ]
    )
    response, ctx = await run_model(model, dataset, as_of, settings)
    assert response.results[0].summary["matched_tickets"] == 500
    assert ctx.tool_calls == 3 and ctx.model_turns == 4
    assert len(ctx.results) == 1


async def test_rejected_nonfinite_arguments_are_not_echoed(dataset, as_of, settings):
    model = ToolModel(
        plan=[
            (
                "aggregate_tickets",
                {
                    "spec": {
                        "selection": {
                            "where": [
                                {"field": "customer_rating", "op": "eq", "value": float("nan")}
                            ]
                        }
                    }
                },
            ),
            ("aggregate_tickets", {"spec": {}}),
        ]
    )
    response, _ = await run_model(model, dataset, as_of, settings)
    wire = response.model_dump_json()
    assert response.tools_used[0]["status"] == "error"
    assert "arguments" not in response.tools_used[0]
    assert "NaN" not in wire


@pytest.mark.parametrize(
    "model",
    [
        ToolModel(plain_answer=True),
        ToolModel(final_override={"status": "success", "result_ids": ["invented"], "message": ""}),
    ],
)
async def test_unsubstantiated_answer_rejected(model, dataset, as_of, settings):
    with pytest.raises(ApplicationError):
        await run_model(model, dataset, as_of, settings)


async def test_budget_stops_runaway_calls(dataset, as_of, settings):
    model = ToolModel(plan=[("get_dataset_info", {"spec": {}})] * 10)
    with pytest.raises(ApplicationError) as error:
        await run_model(model, dataset, as_of, settings)
    assert error.value.code == "tool_call_limit"
    assert len(model.captured) == 4


async def test_request_isolation(dataset, as_of, settings):
    model = ToolModel(plan=[("aggregate_tickets", {"spec": {}})])
    agent = build_agent(settings, model)
    contexts = [RequestContext(dataset, as_of) for _ in range(2)]
    outcomes = await asyncio.gather(
        *(invoke_agent(agent, "Count tickets", ctx) for ctx in contexts)
    )
    assert outcomes[0].result_ids != outcomes[1].result_ids
    assert all(len(ctx.results) == 1 and ctx.model_turns == 2 for ctx in contexts)


async def test_all_nine_tool_schemas_hide_runtime():
    assert len(TOOLS) == 9
    for tool in TOOLS:
        schema = tool.tool_call_schema.model_json_schema()
        assert set(schema["properties"]) == {"spec"}
        assert "dataset" not in schema["properties"]


def test_compact_model_schemas_have_resolvable_references():
    def visit(node, root):
        if isinstance(node, dict):
            if "$ref" in node:
                target = root
                for part in node["$ref"].removeprefix("#/").split("/"):
                    target = target[part]
            for value in node.values():
                visit(value, root)
        elif isinstance(node, list):
            for value in node:
                visit(value, root)

    for tool in MODEL_TOOLS:
        schema = tool["function"]["parameters"]
        visit(schema, schema)
        assert set(schema["properties"]) == {"spec"}
        assert schema["properties"]["spec"]["type"] == "object"
    assert len(json.dumps(MODEL_TOOLS, separators=(",", ":"))) < 29000


def completion(name, args, call_id):
    return {
        "id": "mock-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-20b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


@pytest.mark.parametrize("initial_error", [None, 500])
async def test_real_chatgroq_request_contract(dataset, as_of, settings, initial_error):
    captured = []

    def transport(request):
        payload = json.loads(request.content)
        captured.append(payload)
        if len(captured) == 1 and initial_error:
            return httpx.Response(
                initial_error, json={"error": {"message": "temporary", "type": "server_error"}}
            )
        observations = [m for m in payload["messages"] if m["role"] == "tool"]
        if not observations:
            body = completion("list_tickets", {"spec": {}}, "data-call")
        else:
            evidence = json.loads(observations[-1]["content"])
            body = completion(
                "AgentOutcome",
                {"status": "success", "result_ids": [evidence["result_id"]], "message": ""},
                "final-call",
            )
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(transport)) as sync:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as async_client:
            configured = settings.model_copy(update={"groq_api_key": SecretStr("test-key")})
            agent = build_agent(
                configured,
                http_client=sync,
                http_async_client=async_client,
            )
            ctx = RequestContext(dataset, as_of)
            outcome = await invoke_agent(agent, "List tickets", ctx)
            response = build_response(outcome, ctx, Page(limit=500))
    assert len(response.results[0].rows) == 500
    assert len(captured) == (3 if initial_error else 2)
    assert ctx.retry_used == bool(initial_error)
    assert all(p["parallel_tool_calls"] is False for p in captured)
    assert all(p["disable_tool_validation"] is True for p in captured)
    assert all(p["reasoning_effort"] == "low" and p["max_tokens"] == 1024 for p in captured)
    assert all("response_format" not in p for p in captured)
    assert len(captured[0]["tools"]) == 10  # Nine data tools plus final outcome schema.
    serialized = json.dumps(captured)
    assert not any(value in serialized for value in dataset.frame(as_of).issue_summary.unique())
    assert all(
        "runtime" not in t["function"]["parameters"].get("properties", {})
        for t in captured[0]["tools"]
    )


@pytest.mark.parametrize(
    ("status", "code", "expected_calls"),
    [
        (429, "groq_rate_limit", 1),
        (401, "groq_credentials", 1),
        (403, "groq_credentials", 1),
        (500, "groq_unavailable", 2),
        (400, "groq_request_failed", 1),
    ],
)
async def test_provider_errors_are_bounded_and_safe(
    dataset, as_of, settings, status, code, expected_calls
):
    calls = []

    def transport(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={"error": {"message": "SECRET_RAW_PROVIDER_ERROR"}},
            headers={"retry-after": "10"},
        )

    with httpx.Client(transport=httpx.MockTransport(transport)) as sync:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as async_client:
            model = ChatGroq(
                model="openai/gpt-oss-20b",
                api_key=SecretStr("test"),
                max_retries=0,
                http_client=sync,
                http_async_client=async_client,
            )
            with pytest.raises(ApplicationError) as error:
                await run_model(model, dataset, as_of, settings)
    assert error.value.code == code
    assert "SECRET" not in error.value.message
    assert len(calls) == expected_calls
    if status == 429:
        assert error.value.retry_after == "10"


async def test_overall_deadline(dataset, as_of):
    class SlowAgent:
        async def ainvoke(self, *args, **kwargs):
            await asyncio.sleep(1)

    with pytest.raises(ApplicationError) as error:
        await invoke_agent(SlowAgent(), "Count", RequestContext(dataset, as_of), timeout=0.01)
    assert error.value.status_code == 504

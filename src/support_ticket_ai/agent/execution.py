"""Async tool-calling agent with request-local budgets and safe provider failures."""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

import groq
import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langsmith import tracing_context
from pydantic import BaseModel, ValidationError

from support_ticket_ai.agent.context import ApplicationError, RequestContext, encoded
from support_ticket_ai.agent.prompts import SYSTEM_PROMPT
from support_ticket_ai.agent.tools import TOOLS
from support_ticket_ai.config import Settings
from support_ticket_ai.schemas import AgentOutcome

logger = logging.getLogger(__name__)


def compact_schema(value: Any) -> Any:
    """Keep JSON Schema references instead of expanding shared filter definitions."""
    if isinstance(value, dict):
        return {k: compact_schema(v) for k, v in value.items() if k != "title"}
    if isinstance(value, list):
        return [compact_schema(v) for v in value]
    return value


def model_schema(tool: BaseTool) -> dict[str, Any]:
    schema = compact_schema(cast(type[BaseModel], tool.tool_call_schema).model_json_schema())
    # Groq's tool renderer needs the required argument's object shape inline.
    # Keep shared nested definitions referenced to avoid expanding every predicate.
    reference = schema["properties"]["spec"].get("$ref")
    if reference:
        schema["properties"]["spec"] = schema["$defs"][reference.rsplit("/", 1)[-1]]

    def inline_objects(node: Any) -> Any:
        if isinstance(node, dict):
            name = node.get("$ref", "").rsplit("/", 1)[-1]
            if name in {"Metric", "Cohort", "Selection"}:
                return inline_objects(schema["$defs"][name])
            return {key: inline_objects(value) for key, value in node.items()}
        if isinstance(node, list):
            return [inline_objects(value) for value in node]
        return node

    schema["properties"] = inline_objects(schema["properties"])
    for name in {
        "Metric",
        "Cohort",
        "Selection",
        reference.rsplit("/", 1)[-1] if reference else "",
    }:
        schema.get("$defs", {}).pop(name, None)
    if not schema.get("$defs"):
        schema.pop("$defs", None)
    return cast(dict[str, Any], schema)


MODEL_TOOLS: list[BaseTool | dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": model_schema(tool),
        },
    }
    for tool in TOOLS
]


class Boundaries(AgentMiddleware):
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        ctx = cast(RequestContext, request.runtime.context)
        if ctx.model_turns >= 4:
            raise ApplicationError(
                "model_call_limit", "The reasoning limit was reached. Ask a narrower question.", 422
            )
        ctx.model_turns += 1
        reference = (
            f"\nServer reference time: {ctx.as_of.isoformat()}. Timezone: {ctx.dataset.timezone}."
        )
        request = request.override(
            system_message=SystemMessage(content=SYSTEM_PROMPT + reference), tools=MODEL_TOOLS
        )
        started = time.monotonic()
        try:
            response = await handler(request)
        except (groq.APIConnectionError, groq.InternalServerError):
            if ctx.retry_used:
                raise
            ctx.retry_used = True
            await asyncio.sleep(0.25)
            response = await handler(request)
        for message in response.result:
            if isinstance(message, AIMessage):
                if len(message.tool_calls) > 1:
                    raise ApplicationError(
                        "parallel_tool_calls",
                        "The model requested multiple tools at once. Please retry.",
                    )
                if message.response_metadata.get("finish_reason") == "length":
                    raise ApplicationError(
                        "truncated_model_output",
                        "The model response was truncated. Narrow the question.",
                    )
                logger.info(
                    json.dumps(
                        {
                            "event": "model_call",
                            "request_id": ctx.request_id,
                            "duration_ms": round((time.monotonic() - started) * 1000),
                            "model": message.response_metadata.get("model_name"),
                            "tokens": message.usage_metadata or {},
                        },
                        default=str,
                    )
                )
        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        ctx = cast(RequestContext, request.runtime.context)
        ctx.tool_calls += 1
        if ctx.tool_calls > 3:
            raise ApplicationError(
                "tool_call_limit", "The three-tool limit was reached. Ask a narrower question.", 422
            )
        name = request.tool_call["name"]
        execution: dict[str, Any] = {
            "name": name if name in {t.name for t in TOOLS} else "unknown",
            "status": "error",
        }
        ctx.tools_used.append(execution)
        started = time.monotonic()
        try:
            if name not in {t.name for t in TOOLS}:
                raise ValueError("Unknown tool")
            declared = next(tool for tool in TOOLS if tool.name == name)
            cast(type[BaseModel], declared.tool_call_schema).model_validate(
                request.tool_call["args"]
            )
            result = await handler(request)
            if isinstance(result, ToolMessage) and result.status == "error":
                # Replace framework messages that might echo source values or validation inputs.
                raise ValueError("Invalid tool arguments")
            if isinstance(result, ToolMessage):
                # Match the exact UTF-8 byte accounting used by the backend result serializer.
                result.content = encoded(json.loads(str(result.content))).decode()
            execution["status"] = "success"
            return result
        except (ValidationError, ValueError, TypeError, KeyError, OverflowError) as error:
            payload = ctx.bounded(
                {
                    "status": "error",
                    "code": "invalid_tool_arguments",
                    "message": "Arguments were invalid. Check the field, operation, value types, and date bounds against the tool schema.",
                    "issues": [
                        {"path": list(issue["loc"]), "type": issue["type"]}
                        for issue in error.errors(include_input=False, include_url=False)
                    ]
                    if isinstance(error, ValidationError)
                    else [],
                }
            )
            return ToolMessage(
                content=encoded(payload).decode(),
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )
        finally:
            execution["duration_ms"] = round((time.monotonic() - started) * 1000)
            logger.info(
                json.dumps(
                    {
                        "event": "tool_call",
                        "request_id": ctx.request_id,
                        "tool": name if name in {t.name for t in TOOLS} else "unknown",
                        "status": execution["status"],
                        "duration_ms": execution["duration_ms"],
                    }
                )
            )


def build_agent(
    settings: Settings,
    model: BaseChatModel | None = None,
    *,
    http_client: httpx.Client | None = None,
    http_async_client: httpx.AsyncClient | None = None,
) -> Any:
    if model is None:
        if not settings.llm_configured:
            return None
        model = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=1024,
            timeout=settings.provider_timeout,
            max_retries=0,
            reasoning_effort="low",
            model_kwargs={"parallel_tool_calls": False, "disable_tool_validation": True},
            http_client=http_client,
            http_async_client=http_async_client,
        )
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        context_schema=RequestContext,
        middleware=[Boundaries()],
        response_format=ToolStrategy(AgentOutcome, handle_errors=False),
        name="support_ticket_analyst",
    )


async def invoke_agent(
    agent: Any, question: str, ctx: RequestContext, timeout: float = 60
) -> AgentOutcome:
    if agent is None:
        raise ApplicationError(
            "groq_not_configured", "Set GROQ_API_KEY in the backend .env to enable questions.", 503
        )
    try:
        with tracing_context(enabled=False):
            async with asyncio.timeout(timeout):
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": question}]},
                    context=ctx,
                    config={"recursion_limit": 16, "max_concurrency": 1, "callbacks": []},
                )
        outcome = result.get("structured_response")
        if not isinstance(outcome, AgentOutcome):
            raise ApplicationError(
                "missing_evidence", "The model did not produce a valid evidence-backed result."
            )
        if any(result_id not in ctx.results for result_id in outcome.result_ids):
            raise ApplicationError(
                "invalid_evidence",
                "The model referenced an unknown result. No answer was accepted.",
            )
        return outcome
    except ApplicationError:
        raise
    except groq.RateLimitError as error:
        raise ApplicationError(
            "groq_rate_limit",
            "Groq's free-tier quota is currently exhausted. Try again later.",
            429,
            error.response.headers.get("retry-after"),
        ) from error
    except (groq.AuthenticationError, groq.PermissionDeniedError) as error:
        raise ApplicationError(
            "groq_credentials", "Groq rejected the configured key or model permissions.", 503
        ) from error
    except (TimeoutError, groq.APITimeoutError) as error:
        raise ApplicationError(
            "query_timeout", "The request exceeded its time budget. Please retry.", 504
        ) from error
    except (groq.APIConnectionError, groq.InternalServerError) as error:
        raise ApplicationError(
            "groq_unavailable", "Groq is temporarily unavailable.", 503
        ) from error
    except groq.APIStatusError as error:
        raise ApplicationError(
            "groq_request_failed",
            "Groq could not complete this tool request. Check model availability and retry.",
        ) from error
    except Exception as error:
        # Third-party parsers may include raw generations in their errors; never expose them.
        logger.error(
            json.dumps(
                {
                    "event": "agent_failure",
                    "request_id": ctx.request_id,
                    "error_type": type(error).__name__,
                }
            )
        )
        raise ApplicationError(
            "invalid_model_output",
            "The model could not produce a valid tool response. Rephrase the question.",
        ) from error

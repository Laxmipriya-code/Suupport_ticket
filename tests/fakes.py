"""Script the model boundary while exercising real LangChain orchestration and tools."""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ToolModel(BaseChatModel):
    plan: list[tuple[str, dict[str, Any]]] = Field(default_factory=list)
    captured: list[list] = Field(default_factory=list, exclude=True)
    final_override: dict[str, Any] | None = None
    plain_answer: bool = False
    final_status: str = "success"
    final_message: str = ""

    @property
    def _llm_type(self):
        return "test-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.captured.append(list(messages))
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        if self.plain_answer:
            message = AIMessage(content="There are 999999 tickets.")
        elif len(observations) < len(self.plan):
            name, arguments = self.plan[len(observations)]
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": arguments,
                        "id": f"call-{len(observations)}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            results = [json.loads(m.content) for m in observations]
            outcome = self.final_override or {
                "status": self.final_status,
                "result_ids": [r["result_id"] for r in results if r.get("status") == "success"]
                if self.final_status == "success"
                else [],
                "message": self.final_message,
            }
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "AgentOutcome", "args": outcome, "id": "final", "type": "tool_call"}
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

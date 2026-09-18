"""Request-local services and evidence, kept outside serializable agent messages."""

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pandas as pd

from support_ticket_ai.dataset import Dataset
from support_ticket_ai.schemas import Calculation

PREVIEW_FIELDS = {
    "ticket_id",
    "category",
    "priority",
    "status",
    "agent_id",
    "created_at",
    "resolved_at_estimated",
    "response_time_hrs",
    "resolution_time_hrs",
    "customer_rating",
    "age_hours",
    "warning",
    "field",
    "rule",
    "target_state",
    "observed_hours",
    "threshold_hours",
    "excess_hours",
}
ROW_KINDS = {"tickets", "anomalies", "quality", "time_target"}


class ApplicationError(Exception):
    def __init__(
        self, code: str, message: str, status_code: int = 502, retry_after: str | None = None
    ) -> None:
        super().__init__(message)
        self.code, self.message, self.status_code, self.retry_after = (
            code,
            message,
            status_code,
            retry_after,
        )


def encoded(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


@dataclass
class RequestContext:
    dataset: Dataset
    as_of: pd.Timestamp
    request_id: str = field(default_factory=lambda: uuid4().hex)
    results: dict[str, Calculation] = field(default_factory=dict)
    tools_used: list[dict[str, Any]] = field(default_factory=list)
    model_turns: int = 0
    tool_calls: int = 0
    retry_used: bool = False
    output_bytes: int = 0

    def publish(self, result: Calculation) -> dict[str, Any]:
        result_id = f"{self.request_id[:12]}-{len(self.results) + 1}"
        self.results[result_id] = result
        preview = result.rows[: 5 if result.kind in ROW_KINDS else 20]
        if result.kind in ROW_KINDS:
            preview = [{k: v for k, v in row.items() if k in PREVIEW_FIELDS} for row in preview]
        payload = {
            "result_id": result_id,
            "status": "success",
            "kind": result.kind,
            "summary": result.summary,
            "preview": preview,
            "total_rows": len(result.rows),
            "criteria": result.criteria,
            "as_of": self.as_of.isoformat(),
            "dataset_fingerprint": self.dataset.fingerprint,
            "warnings": result.warnings,
            "truncated": len(preview) < len(result.rows),
        }
        return self.bounded(payload)

    def bounded(self, payload: dict[str, Any]) -> dict[str, Any]:
        budget = min(4096, 8192 - self.output_bytes)
        while len(encoded(payload)) > budget and payload.get("preview"):
            payload["preview"].pop()
            payload["truncated"] = True
        if len(encoded(payload)) > budget:
            payload = {
                k: v for k, v in payload.items() if k not in {"criteria", "warnings", "summary"}
            }
            payload.update(truncated=True, note="Detailed evidence remains in the backend result.")
        size = len(encoded(payload))
        if size > budget:
            raise ApplicationError(
                "tool_output_limit", "The tool-output budget was reached. Narrow the question.", 422
            )
        self.output_bytes += size
        return payload

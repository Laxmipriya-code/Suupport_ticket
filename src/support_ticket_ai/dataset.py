"""CSV ingestion. Structural errors reject the file; business warnings preserve records."""

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from support_ticket_ai.schemas import ENUM_VALUES, Calculation

SOURCE_COLUMNS = [
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
]
FIELD_DESCRIPTIONS = {
    "ticket_id": "Unique ticket identifier",
    "created_at": "Recorded creation timestamp, localized to the configured timezone",
    "category": "Support issue category",
    "priority": "Recorded urgency level",
    "status": "Recorded snapshot status; not status transition history",
    "response_time_hrs": "Hours to first response; missing means unknown",
    "resolution_time_hrs": "Recorded hours to resolution; usually missing for unresolved tickets",
    "agent_id": "Assigned support agent identifier",
    "customer_rating": "Customer rating from 1 to 5; missing is excluded from averages",
    "issue_summary": "Literal issue text, retained in the backend and excluded from model previews",
}
DEFINITIONS = {
    "unresolved": "status is Open or Escalated; Open alone excludes Escalated",
    "average": "Exclude null measurements and report the non-null sample count",
    "resolved_at_estimated": "created_at plus resolution_time_hrs for Resolved tickets; not an observed timestamp",
    "age_hours": "Elapsed hours from creation to the active reference time; not time to resolution",
    "time_windows": "Explicit end timestamps are exclusive. Current periods include as_of. Weeks start Monday.",
    "time_target": "An explicit analytical threshold, not a contractual SLA",
    "snapshot": "Statuses are recorded in the CSV and cannot reconstruct historical backlog",
}


class Dataset:
    def __init__(self, path: Path, timezone: str = "UTC") -> None:
        self.timezone = timezone
        content = path.read_bytes()
        self.fingerprint = hashlib.sha256(content).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if len(header) != len(set(header)):
            raise ValueError("CSV contains duplicate column headers")
        if set(header) != set(SOURCE_COLUMNS):
            raise ValueError(f"CSV must have exactly these columns: {', '.join(SOURCE_COLUMNS)}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if frame.empty:
            raise ValueError("CSV must contain at least one ticket")
        for field in SOURCE_COLUMNS:
            frame[field] = frame[field].str.strip()
        for field in [
            "ticket_id",
            "agent_id",
            "created_at",
            "category",
            "priority",
            "status",
            "issue_summary",
        ]:
            self._reject(frame, frame[field].eq(""), field, "must not be blank")
        self._reject(frame, frame.ticket_id.duplicated(keep=False), "ticket_id", "must be unique")
        for field, allowed in ENUM_VALUES.items():
            self._reject(frame, ~frame[field].isin(allowed), field, "has an unsupported category")
        dates = pd.to_datetime(frame.created_at, format="%Y-%m-%d %H:%M", errors="coerce")
        self._reject(frame, dates.isna(), "created_at", "must be YYYY-MM-DD HH:MM")
        frame["created_at"] = dates.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        for field in ["response_time_hrs", "resolution_time_hrs", "customer_rating"]:
            numeric = pd.to_numeric(frame[field].replace("", np.nan), errors="coerce")
            invalid = frame[field].ne("") & (numeric.isna() | ~np.isfinite(numeric))
            invalid |= numeric.lt(0)
            if field == "customer_rating":
                invalid |= numeric.notna() & (~numeric.between(1, 5) | numeric.mod(1).ne(0))
            self._reject(frame, invalid, field, "has an invalid numeric value")
            frame[field] = numeric
        frame["resolved_at_estimated"] = (
            frame.created_at + pd.to_timedelta(frame.resolution_time_hrs, unit="h")
        ).where(frame.status.eq("Resolved"))
        self._frame = frame

    @staticmethod
    def _reject(frame: pd.DataFrame, mask: pd.Series, column: str, message: str) -> None:
        if mask.any():
            rows = [int(i) + 2 for i in frame.index[mask][:10]]
            raise ValueError(f"CSV column {column} {message}; source line(s): {rows}")

    @property
    def latest_created(self) -> pd.Timestamp:
        return self._frame.created_at.max()

    def frame(self, as_of: pd.Timestamp) -> pd.DataFrame:
        # Each request gets a copy: no calculation can mutate the loaded source snapshot.
        frame = self._frame.copy(deep=True)
        frame["age_hours"] = (as_of - frame.created_at).dt.total_seconds() / 3600
        return frame

    def info(self, section: str = "overview") -> Calculation:
        if section == "definitions":
            summary: dict[str, Any] = dict(DEFINITIONS)
        elif section == "schema":
            summary = {
                "fields": {
                    field: {
                        "type": str(self._frame[field].dtype),
                        "description": FIELD_DESCRIPTIONS[field],
                    }
                    for field in SOURCE_COLUMNS
                },
                "allowed_values": ENUM_VALUES,
                "agents": sorted(self._frame.agent_id.unique().tolist()),
                "nullable": ["response_time_hrs", "resolution_time_hrs", "customer_rating"],
                "derived_fields": ["resolved_at_estimated", "age_hours"],
            }
        else:
            summary = {
                "total_tickets": len(self._frame),
                "date_start": self._frame.created_at.min().isoformat(),
                "date_end": self.latest_created.isoformat(),
                "timezone": self.timezone,
                "statuses": self._frame.status.value_counts().to_dict(),
                "categories": self._frame.category.value_counts().to_dict(),
                "priorities": self._frame.priority.value_counts().to_dict(),
                "timing_warning_count": int(
                    (self._frame.resolution_time_hrs < self._frame.response_time_hrs).sum()
                ),
            }
        return Calculation(kind="dataset_info", summary=summary)

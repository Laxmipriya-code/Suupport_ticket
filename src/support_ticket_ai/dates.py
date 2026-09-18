"""Timestamp normalization, reference times, date windows, and date warnings."""

from datetime import datetime

import pandas as pd

from support_ticket_ai.schemas import DateWindow


def timestamp(value: datetime | str | pd.Timestamp, timezone: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("Invalid timestamp")
    return ts.tz_localize(timezone) if ts.tzinfo is None else ts.tz_convert(timezone)


def reference_time(value: datetime | None, timezone: str) -> pd.Timestamp:
    return timestamp(value, timezone) if value else pd.Timestamp.now(tz=timezone)


def date_bounds(
    window: DateWindow, as_of: pd.Timestamp
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, bool]:
    """Return start-inclusive, end-exclusive bounds; current periods include as_of."""
    timezone = str(as_of.tz)
    if not window.period:
        start = timestamp(window.start, timezone) if window.start else None
        end = timestamp(window.end, timezone) if window.end else None
        if start is not None and end is not None and start >= end:
            raise ValueError("Date window start must precede its exclusive end")
        return start, end, False
    today = as_of.normalize()
    week = today - pd.DateOffset(days=today.weekday())
    month = today.replace(day=1)
    year = month.replace(month=1)
    current = {"today": today, "this_week": week, "this_month": month, "this_year": year}
    if window.period in current:
        return current[window.period], as_of, True
    previous = {
        "last_week": (week - pd.DateOffset(weeks=1), week),
        "last_month": (month - pd.DateOffset(months=1), month),
        "last_year": (year - pd.DateOffset(years=1), year),
    }
    start, end = previous[window.period]
    return start, end, False


def time_warnings(
    as_of: pd.Timestamp, latest_created: pd.Timestamp, estimated: bool = False
) -> list[str]:
    warnings = [
        "Statuses are the CSV snapshot; reference time does not reconstruct historical statuses."
    ]
    if as_of > latest_created + pd.DateOffset(days=31):
        warnings.append("This is a historical dataset; recent-period queries may have no records.")
    if estimated:
        warnings.append("Resolution timestamps are estimates: created_at + resolution_time_hrs.")
    return warnings

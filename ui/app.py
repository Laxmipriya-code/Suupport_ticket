"""Streamlit presentation only. All ticket calculations happen behind the API."""

import os
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Support Ticket AI", page_icon="🎫", layout="wide")
st.html("<style>[data-testid='stAppDeployButton'] {display: none;}</style>")


def api(method: str, path: str, **kwargs: Any) -> dict[str, Any] | None:
    try:
        response = httpx.request(method, API_URL + path, timeout=65, **kwargs)
        if not response.is_success:
            error = response.json().get("error", {})
            st.error(error.get("message", "The API could not complete this request."))
            if error.get("request_id"):
                st.caption(f"Request ID: {error['request_id']}")
            if response.status_code == 429:
                st.info(
                    f"Retry after: {response.headers.get('Retry-After', 'the Groq quota resets')}."
                )
            return None
        return response.json()
    except (httpx.HTTPError, ValueError):
        st.error(
            "The backend is unavailable or did not respond in time. Check that both services are running."
        )
        return None


def show_response(response: dict[str, Any], key: str) -> None:
    st.write(response["answer"])
    st.caption(f"Reference time: {response['as_of']} · Request: {response['request_id']}")
    for warning in response.get("warnings", []):
        st.caption(warning)
    for index, result in enumerate(response.get("results", [])):
        st.subheader(result["kind"].replace("_", " ").title())
        if result["rows"]:
            frame = pd.DataFrame(result["rows"])
            if result["kind"] in {"anomalies", "time_target"}:
                leading = [
                    name
                    for name in [
                        "ticket_id",
                        "rule",
                        "target_state",
                        "observed_hours",
                        "threshold_hours",
                        "excess_hours",
                        "reason",
                    ]
                    if name in frame
                ]
                frame = frame[leading + [name for name in frame if name not in leading]]
            if result["kind"] == "trend" and "period_start" in frame:
                st.line_chart(frame.set_index("period_start")[["value"]])
            if result["kind"] == "comparison":
                st.bar_chart(frame.set_index("metric")[["baseline", "comparison"]])
            pages = max(1, (len(frame) + 49) // 50)
            page = st.number_input(
                "Table page",
                min_value=1,
                max_value=pages,
                value=1,
                key=f"{key}-{response['request_id']}-{index}",
            )
            st.dataframe(
                frame.iloc[(page - 1) * 50 : page * 50],
                hide_index=True,
                width="stretch",
                column_config={name: name.replace("_", " ").title() for name in frame},
            )
            st.caption(
                f"{result['total_rows']} result rows; {len(frame)} available in this response."
            )
            if result["truncated"]:
                st.info(
                    "This result is truncated. Narrow the filters to inspect the remaining records."
                )
        with st.expander("Calculation details and applied filters"):
            st.json({"summary": result["summary"], "criteria": result["criteria"]})
    if response.get("tools_used"):
        with st.expander("Tools used"):
            st.json(response["tools_used"])


st.title("Support Ticket AI")
st.write("Ask questions, inspect the supporting records, and investigate unusual ticket timings.")
with st.sidebar:
    st.header("Reference time")
    mode = st.radio("Use", ["Current time", "Historical demonstration", "Custom time"])
    as_of = None
    if mode == "Historical demonstration":
        as_of = "2024-03-30T18:06:00Z"
        st.caption("March 30, 2024 at 18:06 UTC")
    elif mode == "Custom time":
        raw = st.text_input("ISO timestamp with timezone", "2024-03-30T18:06:00+00:00")
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Timezone required")
            as_of = parsed.isoformat()
        except ValueError:
            st.error("Enter a valid timestamp including a timezone.")
            st.stop()
    st.caption(
        "The source is a 2024 snapshot. Changing the date does not reconstruct historical statuses."
    )
    st.markdown(
        f"[API documentation]({API_URL if 'api:8000' not in API_URL else 'http://127.0.0.1:8000'}/docs)"
    )

overview, questions, anomaly_tab = st.tabs(["Overview", "Questions", "Anomalies"])
with overview:
    stats = api("GET", "/api/v1/stats")
    if stats:
        cols = st.columns(4)
        for col, (name, value) in zip(
            cols,
            [("Tickets", stats["total_tickets"])]
            + [
                (status, stats["statuses"].get(status, 0))
                for status in ["Open", "Resolved", "Escalated"]
            ],
            strict=True,
        ):
            col.metric(name, value)
        st.caption(f"Created: {stats['date_start']} to {stats['date_end']}")
        left, right = st.columns(2)
        left.subheader("Tickets by category")
        right.subheader("Tickets by priority")
        left.bar_chart(pd.Series(stats["categories"], name="Tickets").rename_axis("Category"))
        right.bar_chart(pd.Series(stats["priorities"], name="Tickets").rename_axis("Priority"))
        st.warning(
            f"{stats['timing_warning_count']} records have resolution time shorter than first-response time. Original values are preserved."
        )
        with st.expander("Data quality"):
            st.json(stats["data_quality"])

with questions:
    examples = [
        "How many tickets are currently open?",
        "Which agent has the lowest average customer rating?",
        "Which agent resolved the most tickets this month?",
        "Show Critical tickets not resolved within 12 hours.",
        "What percentage of Critical tickets are unresolved?",
        "Are there resolution-time anomalies this week?",
    ]
    example = st.selectbox("Example question", examples)
    with st.form("question_form"):
        question = st.text_area("Your question", value=example, max_chars=2000)
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        with st.spinner("Calculating from the dataset…"):
            result = api(
                "POST", "/api/v1/query", json={"question": question, "as_of": as_of, "limit": 500}
            )
        st.session_state["query_result"] = result
    if st.session_state.get("query_result"):
        show_response(st.session_state["query_result"], "query")

with anomaly_tab:
    rules = {
        "all": "All anomaly checks",
        "long_resolution": "Long resolution times",
        "overdue_high_priority": "Overdue High/Critical tickets",
    }
    rule = st.selectbox("Anomaly rule", list(rules), format_func=lambda value: rules[value])
    if st.button("Find anomalies"):
        params = {"rule": rule, "limit": 500}
        if as_of:
            params["as_of"] = as_of
        with st.spinner("Checking ticket timings…"):
            st.session_state["anomaly_result"] = api("GET", "/api/v1/anomalies", params=params)
    if st.session_state.get("anomaly_result"):
        show_response(st.session_state["anomaly_result"], "anomaly")

"""Opt-in live evaluation with resumable quota handling and no ticket rows in reports."""

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import httpx

from support_ticket_ai.agent.context import ApplicationError, RequestContext
from support_ticket_ai.agent.execution import build_agent, invoke_agent
from support_ticket_ai.config import Settings
from support_ticket_ai.dataset import Dataset
from support_ticket_ai.dates import timestamp
from support_ticket_ai.responses import build_response
from support_ticket_ai.schemas import Page, QueryResponse

QUOTA_ERROR = "groq_rate_limit"


def matches(actual: Any, expected: Any) -> bool:
    """Expected objects are subsets; expected arrays assert all rows, in order."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and matches(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(matches(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and math.isclose(
            actual, expected, rel_tol=1e-7, abs_tol=1e-7
        )
    return actual == expected


def check_response(response: QueryResponse, case: dict[str, Any]) -> bool:
    if response.status != case["status"]:
        return False
    if response.status != "success":
        return not response.results and bool(response.answer.strip())
    return any(t["name"] == case["tool"] for t in response.tools_used) and any(
        matches(result.model_dump(mode="json"), case["expected"]) for result in response.results
    )


def run_metadata(fixture: dict[str, Any], settings: Settings, dataset: Dataset) -> dict[str, Any]:
    """Pin saved progress to the same code, fixtures, dependencies, model, and data."""
    digest = hashlib.sha256(json.dumps(fixture, sort_keys=True).encode())
    source = Path(__file__).parent
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    lock = Path("uv.lock")
    if lock.exists():
        digest.update(lock.read_bytes())
    return {
        "mode": "live",
        "model": settings.groq_model,
        "as_of": fixture["as_of"],
        "dataset_fingerprint": dataset.fingerprint,
        "data_timezone": dataset.timezone,
        "evaluation_fingerprint": digest.hexdigest(),
    }


def summarize(
    metadata: dict[str, Any], records: list[dict[str, Any]], total_cases: int
) -> dict[str, Any]:
    passed = sum(bool(r["passed"]) for r in records)
    blocked = sum(r.get("error") == QUOTA_ERROR for r in records)
    completed = len(records) - blocked
    official = [r for r in records if r["official"]]
    full_suite = completed == total_cases
    return {
        **metadata,
        "total": len(records),
        "planned": total_cases,
        "completed": completed,
        "quota_blocked": blocked,
        "passed": passed,
        "accuracy": passed / len(records) if records else 0,
        "first_attempt_passed": sum(
            bool((r.get("previous_attempts") or [r])[0]["passed"]) for r in records
        ),
        "full_suite": full_suite,
        "acceptance_met": (
            full_suite
            and len(official) == 5
            and all(r["passed"] for r in official)
            and bool(records)
            and passed / len(records) >= 0.9
        ),
        "state": "quota_blocked" if blocked else ("complete" if full_suite else "partial"),
        "cases": records,
    }


def load_progress(path: Path, metadata: dict[str, Any], case_ids: set[str]) -> list[dict[str, Any]]:
    saved = json.loads(path.read_text())
    if any(saved.get(key) != value for key, value in metadata.items()):
        raise ValueError(
            "Saved progress does not match this code, model, dataset, or fixture. Start a new run."
        )
    records = saved["cases"]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)) or not set(ids) <= case_ids:
        raise ValueError("Saved progress contains duplicate or unknown case IDs.")
    return list(records)


async def evaluate(args: argparse.Namespace) -> int:
    settings = Settings()
    if not settings.llm_configured:
        print("Set GROQ_API_KEY in .env before running live evaluation.")
        return 2
    fixture = json.loads(args.cases.read_text())
    dataset = Dataset(settings.data_path, settings.data_timezone)
    metadata = run_metadata(fixture, settings, dataset)
    order = {case["id"]: index for index, case in enumerate(fixture["cases"])}
    records = load_progress(args.output, metadata, set(order)) if args.resume else []
    selected = [c for c in fixture["cases"] if not args.official_only or c["official"]]
    selected = selected[: args.limit] if args.limit else selected
    finished = {r["id"] for r in records if r.get("error") != QUOTA_ERROR}
    pending = [case for case in selected if case["id"] not in finished]

    def save_report() -> dict[str, Any]:
        records.sort(key=lambda record: order[record["id"]])
        report = summarize(metadata, records, len(fixture["cases"]))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return report

    with httpx.Client(timeout=settings.provider_timeout) as sync:
        async with httpx.AsyncClient(timeout=settings.provider_timeout) as async_client:
            agent = build_agent(settings, http_client=sync, http_async_client=async_client)
            for index, case in enumerate(pending):
                ctx = RequestContext(dataset, timestamp(fixture["as_of"], dataset.timezone))
                record = {"id": case["id"], "official": case["official"], "passed": False}
                prior = next((r for r in records if r["id"] == case["id"]), None)
                if prior:
                    record["previous_attempts"] = [
                        *prior.get("previous_attempts", []),
                        {key: value for key, value in prior.items() if key != "previous_attempts"},
                    ]
                try:
                    outcome = await invoke_agent(
                        agent, case["question"], ctx, settings.request_timeout
                    )
                    response = build_response(outcome, ctx, Page(limit=500))
                    record.update(
                        passed=check_response(response, case),
                        status=response.status,
                        tools=[t["name"] for t in ctx.tools_used],
                    )
                    if not record["passed"]:
                        record["observed"] = [
                            {"kind": result.kind, "summary": result.summary}
                            for result in response.results
                        ]
                except ApplicationError as error:
                    record.update(error=error.code, retry_after=error.retry_after)
                record.update(model_turns=ctx.model_turns, tool_calls=ctx.tool_calls)
                records = [r for r in records if r["id"] != case["id"]] + [record]
                save_report()
                detail = record.get("error", "evidence or tool mismatch")
                print(
                    f"{case['id']}: {'PASS' if record['passed'] else 'FAIL (' + str(detail) + ')'}",
                    flush=True,
                )
                if record.get("error") == QUOTA_ERROR:
                    print(
                        "Stopped at the provider quota. Progress is saved; use --resume when quota is available."
                    )
                    return 3
                if index < len(pending) - 1 and args.delay:
                    await asyncio.sleep(args.delay)
    report = save_report()
    print(f"{report['passed']}/{len(records)} passed. Report: {args.output}")
    return (
        0
        if (
            report["acceptance_met"]
            if report["full_suite"]
            else report["passed"] == len(records) and not report["quota_blocked"]
        )
        else 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Required: spend configured Groq quota")
    parser.add_argument("--cases", type=Path, default=Path("tests/fixtures/query_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    parser.add_argument("--official-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=65, help="Seconds between questions")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue unattempted/quota-blocked cases from the same run",
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live to explicitly enable real Groq requests. CI never enables this.")
    if args.limit < 0 or args.delay < 0:
        parser.error("Limit and delay must be nonnegative.")
    if args.resume and (args.limit or args.official_only):
        parser.error("--resume continues the full suite; do not combine it with subset options.")
    try:
        raise SystemExit(asyncio.run(evaluate(args)))
    except KeyboardInterrupt:
        print(f"Interrupted. Completed case results are saved in {args.output}.")
        raise SystemExit(130) from None
    except (OSError, ValueError, KeyError) as error:
        print(f"Evaluation configuration error: {error}")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

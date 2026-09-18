import argparse
import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from support_ticket_ai import evaluation, query
from support_ticket_ai.agent.context import ApplicationError
from support_ticket_ai.schemas import AgentOutcome, AggregateSpec


@pytest.mark.parametrize("first_passes", [True, False])
async def test_quota_stop_and_resume_preserve_completed_results(
    tmp_path, monkeypatch, settings, first_passes
):
    original = json.loads((Path(__file__).parents[1] / "fixtures/query_cases.json").read_text())
    fixture = {"as_of": original["as_of"], "cases": [original["cases"][0], original["cases"][5]]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(fixture))
    args = argparse.Namespace(
        cases=path,
        output=tmp_path / "report.json",
        resume=False,
        official_only=False,
        limit=0,
        delay=0,
    )
    configured = settings.model_copy(update={"groq_api_key": SecretStr("test")})
    monkeypatch.setattr(evaluation, "Settings", lambda: configured)
    monkeypatch.setattr(evaluation, "build_agent", lambda *a, **kw: object())
    calls = []
    quota = True

    async def invoke(agent, question, ctx, timeout):
        case = next(case for case in fixture["cases"] if case["question"] == question)
        calls.append(case["id"])
        if case["id"] == "total" and quota:
            raise ApplicationError("groq_rate_limit", "Quota exhausted", 429, "900")
        result = query.aggregate(
            ctx.dataset, AggregateSpec.model_validate(case["arguments"]["spec"]), ctx.as_of
        )
        if case["id"] == "open" and not first_passes:
            result.summary["metrics"]["count"]["value"] = -1
        ctx.results["calculated"] = result
        ctx.tools_used.append({"name": "aggregate_tickets"})
        return AgentOutcome(status="success", result_ids=["calculated"])

    monkeypatch.setattr(evaluation, "invoke_agent", invoke)
    assert await evaluation.evaluate(args) == 3
    initial = json.loads(args.output.read_text())
    assert initial["completed"] == 1 and initial["quota_blocked"] == 1
    assert initial["acceptance_met"] is False
    quota = False
    args.resume = True
    await evaluation.evaluate(args)
    resumed = json.loads(args.output.read_text())
    assert calls == ["open", "total", "total"]  # Never rerun a completed semantic failure.
    assert resumed["completed"] == 2 and resumed["quota_blocked"] == 0
    assert resumed["cases"][0]["passed"] == first_passes
    assert resumed["cases"][1]["previous_attempts"][0]["error"] == "groq_rate_limit"
    assert resumed["first_attempt_passed"] == int(first_passes)
    changed = {
        **evaluation.run_metadata(fixture, configured, query.Dataset(settings.data_path)),
        "model": "different-model",
    }
    with pytest.raises(ValueError, match="does not match"):
        evaluation.load_progress(args.output, changed, {"open", "total"})


def test_acceptance_requires_completed_full_suite_and_official_examples():
    records = [{"id": str(i), "official": i < 5, "passed": True} for i in range(40)]
    assert evaluation.summarize({}, records, 40)["acceptance_met"]
    records[10].update(passed=False, error="groq_rate_limit")
    blocked = evaluation.summarize({}, records, 40)
    assert not blocked["acceptance_met"] and not blocked["full_suite"]
    records[10].pop("error")
    assert evaluation.summarize({}, records, 40)["acceptance_met"]
    records[0]["passed"] = False
    assert not evaluation.summarize({}, records, 40)["acceptance_met"]

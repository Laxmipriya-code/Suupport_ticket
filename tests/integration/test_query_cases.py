import json
from pathlib import Path

import pytest

from support_ticket_ai.evaluation import check_response
from tests.fakes import ToolModel
from tests.integration.test_agent import run_model

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures/query_cases.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["id"])
async def test_labelled_tool_contracts(case, dataset, as_of, settings):
    model = ToolModel(
        plan=[(case["tool"], case["arguments"])] if "tool" in case else [],
        final_status=case["status"],
        final_message="Please specify the metric."
        if case["status"] == "clarification"
        else "This operation is unsupported.",
    )
    response, _ = await run_model(model, dataset, as_of, settings, case["question"])
    assert check_response(response, case), response.model_dump()


def test_all_tools_and_assessment_examples_have_labels():
    assert len(FIXTURE["cases"]) == 40
    assert sum(case["official"] for case in FIXTURE["cases"]) == 5
    assert len({case["tool"] for case in FIXTURE["cases"] if "tool" in case}) == 9

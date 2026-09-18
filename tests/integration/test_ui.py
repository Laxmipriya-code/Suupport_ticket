from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from support_ticket_ai.main import create_app
from tests.fakes import ToolModel

UI = Path(__file__).parents[2] / "ui/app.py"


def connect_ui(monkeypatch, client):
    def request(method, url, **kwargs):
        kwargs.pop("timeout", None)
        return client.request(method, urlsplit(url).path, **kwargs)

    monkeypatch.setattr(httpx, "request", request)


def test_ui_overview_anomalies_and_pagination_without_llm(monkeypatch, settings):
    model = ToolModel(plan=[("list_tickets", {"spec": {}})])
    with TestClient(create_app(settings, model)) as client:
        connect_ui(monkeypatch, client)
        app = AppTest.from_file(str(UI)).run()
        assert not app.exception
        assert [metric.value for metric in app.metric] == ["500", "111", "327", "62"]
        app.radio[0].set_value("Historical demonstration").run()
        app.text_area[0].set_value("Show all tickets")
        app.button[0].click().run()
        assert not app.exception and len(model.captured) == 2
        assert len(app.dataframe[0].value) == 50
        first = app.dataframe[0].value.ticket_id.tolist()
        app.number_input[0].set_value(2).run()
        assert len(model.captured) == 2
        assert not set(first) & set(app.dataframe[0].value.ticket_id.tolist())
        app.button[1].click().run()
        assert not app.exception and len(model.captured) == 2
        assert any(
            "21 long-resolution" in item.value and "80 overdue" in item.value
            for item in app.markdown
        )


@pytest.mark.parametrize("status", ["clarification", "unsupported"])
def test_ui_non_success_outcomes(monkeypatch, settings, status):
    model = ToolModel(final_status=status, final_message="Choose a metric or a supported question.")
    with TestClient(create_app(settings, model)) as client:
        connect_ui(monkeypatch, client)
        app = AppTest.from_file(str(UI)).run()
        app.button[0].click().run()
        assert not app.exception
        assert any("Choose a metric" in item.value for item in app.markdown)
        assert not app.dataframe


def test_ui_without_credentials(monkeypatch, settings):
    with TestClient(create_app(settings)) as client:
        connect_ui(monkeypatch, client)
        app = AppTest.from_file(str(UI)).run()
        app.button[0].click().run()
        assert not app.exception
        assert "GROQ_API_KEY" in app.error[0].value


def test_ui_unavailable_backend(monkeypatch):
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("unavailable")

    monkeypatch.setattr(httpx, "request", unavailable)
    app = AppTest.from_file(str(UI)).run()
    assert not app.exception and "backend is unavailable" in app.error[0].value

from fastapi.testclient import TestClient

from support_ticket_ai.main import create_app
from tests.fakes import ToolModel


def test_api_without_credentials(settings):
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        assert health.json()["status"] == "degraded"
        assert health.json()["llm_configured"] is False
        assert client.get("/api/v1/stats").json()["total_tickets"] == 500
        query = client.post("/api/v1/query", json={"question": "How many tickets?"})
        assert query.status_code == 503
        assert query.json()["error"]["code"] == "groq_not_configured"
        anomalies = client.get("/api/v1/anomalies", params={"as_of": "2024-03-30T18:06:00Z"})
        assert anomalies.status_code == 200
        assert anomalies.json()["results"][0]["summary"]["long_resolution_count"] == 21
        assert "NaN" not in anomalies.text
        assert health.headers["x-request-id"]


def test_query_and_pagination_without_more_model_calls(settings):
    model = ToolModel(plan=[("list_tickets", {"spec": {}})])
    with TestClient(create_app(settings, model)) as client:
        response = client.post(
            "/api/v1/query",
            json={"question": "List all tickets", "as_of": "2024-03-30T18:06:00Z", "limit": 50},
        )
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["total_rows"] == 500 and len(result["rows"]) == 50
        assert result["truncated"]
        before = len(model.captured)
        next_page = client.post(
            "/api/v1/tickets/search",
            json={
                "spec": result["criteria"]["search_spec"],
                "offset": 50,
                "limit": 50,
                "as_of": response.json()["as_of"],
            },
        )
        assert next_page.status_code == 200
        assert len(model.captured) == before
        assert set(r["ticket_id"] for r in result["rows"]).isdisjoint(
            r["ticket_id"] for r in next_page.json()["result"]["rows"]
        )


def test_api_validation_and_safe_errors(settings):
    with TestClient(create_app(settings)) as client:
        for body in [
            {"question": " "},
            {"question": "x" * 2001},
            {"question": "hello", "unknown": "bad"},
            {"question": "hello", "limit": 501},
        ]:
            response = client.post("/api/v1/query", json=body)
            assert response.status_code == 422
            assert response.json()["error"]["request_id"]
        bad_filter = client.post(
            "/api/v1/tickets/search",
            json={
                "spec": {
                    "selection": {"where": [{"field": "status", "op": "eq", "value": "DROP TABLE"}]}
                }
            },
        )
        assert bad_filter.status_code == 422
        assert "DROP TABLE" not in bad_filter.text
        invalid_dates = client.get(
            "/api/v1/anomalies", params={"start": "2024-03-30", "end": "2024-01-01"}
        )
        assert invalid_dates.status_code == 422
        assert client.get("/openapi.json").status_code == 200

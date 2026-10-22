import os

os.environ["FINQA_SKIP_LOAD"] = "1"  # don't download a real model in CI

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


class FakeModel:
    def answer(self, question, context=None, max_new_tokens=None):
        return f"echo: {question}"


def test_health_reports_model_not_loaded():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False


def test_ask_returns_503_without_model():
    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "What was revenue?"})
        assert r.status_code == 503


def test_ask_with_model():
    with TestClient(app) as client:
        app.state.model = FakeModel()
        r = client.post("/ask", json={"question": "What was revenue?", "context": "Revenue was $5."})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "echo: What was revenue?"
        assert body["latency_ms"] >= 0


def test_ask_validates_input():
    with TestClient(app) as client:
        app.state.model = FakeModel()
        assert client.post("/ask", json={"question": "hi"}).status_code == 422
        assert client.post("/ask", json={"question": "valid?", "max_new_tokens": 0}).status_code == 422

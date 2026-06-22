from fastapi.testclient import TestClient

from src.backend.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_endpoint(monkeypatch):
    def fake_answer(question, history=None):
        return f"Answer for {question}"

    monkeypatch.setattr("src.backend.main.ask_netflix", fake_answer)
    client = TestClient(app)
    response = client.post("/chat", json={"question": "Who directed Bird Box?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Answer for Who directed Bird Box?"
    assert payload["session_id"]


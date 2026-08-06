import json
from urllib import request

from src.text import ollama_client


def test_ollama_generate_builds_deterministic_request(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"response": "positive"}).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    result = ollama_client.ollama_generate("test", temperature=0.0)

    assert result == "positive"
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["temperature"] == 0.0

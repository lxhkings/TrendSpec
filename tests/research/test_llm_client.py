import httpx

from trendspec.research.llm_client import MockLLMClient, OpenAICompatClient


def test_mock_client_returns_scripted():
    client = MockLLMClient(responses=["hello", "world"])
    assert client.complete("sys", "u1") == "hello"
    assert client.complete("sys", "u2") == "world"


def test_openai_compat_client_parses_choice():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ASSISTANT_REPLY"}}]
        })

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://api.deepseek.com/v1",
        api_key="k", model="deepseek-chat",
        transport=transport,
    )
    out = client.complete("system text", "user text")
    assert out == "ASSISTANT_REPLY"

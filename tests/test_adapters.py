import json
from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_eval.adapters.base import call_python
from agent_eval.adapters.http import HttpAdapter
from agent_eval.adapters.langchain import extract_langchain_result
from agent_eval.adapters.openai_agents import extract_openai_result
from agent_eval.config import AdapterConfig
from agent_eval.models import AgentRequest, Turn


def test_python_callable_accepts_request_or_text() -> None:
    request = AgentRequest(case_id="c", input="hello", turns=[Turn(role="user", content="earlier")])

    def takes_request(item: AgentRequest) -> dict[str, str]:
        assert item.turns[0].content == "earlier"
        return {"output": item.input}

    def takes_text(text: str) -> str:
        return text.upper()

    assert call_python(takes_request, request).output == "hello"
    assert call_python(takes_text, request).output == "HELLO"


def test_langchain_message_extraction() -> None:
    result = {
        "messages": [
            HumanMessage(content="refund headphones"),
            AIMessage(
                content="Plan: look up u_alice",
                tool_calls=[
                    {
                        "name": "get_user_purchase_history",
                        "args": {"user_id": "u_alice"},
                        "id": "call-1",
                    }
                ],
            ),
            ToolMessage(
                content=json.dumps({"found": True}),
                tool_call_id="call-1",
                name="get_user_purchase_history",
            ),
            AIMessage(content="done"),
        ],
        "structured_response": {
            "answer": "Refund $80.",
            "conversation": "A person will approve the ticket.",
        },
    }
    extracted = extract_langchain_result(result)
    assert extracted.plan == ["Plan: look up u_alice"]
    assert extracted.tool_calls[0].name == "get_user_purchase_history"
    assert extracted.tool_calls[0].arguments == {"user_id": "u_alice"}
    assert extracted.tool_calls[0].output == {"found": True}
    assert "Refund $80." in extracted.output
    assert "person will approve" in extracted.output


def test_openai_agents_extraction() -> None:
    call = SimpleNamespace(
        type="tool_call_item",
        tool_name="calculate_refund",
        raw_item=SimpleNamespace(
            name="calculate_refund", arguments='{"user_id":"u_bob","item":"mouse"}', call_id="c1"
        ),
    )
    output = SimpleNamespace(
        type="tool_call_output_item",
        output='{"eligible": true, "amount": 25.0}',
        raw_item=SimpleNamespace(call_id="c1"),
    )
    message = SimpleNamespace(
        type="message_output_item",
        raw_item=SimpleNamespace(content=[SimpleNamespace(text="Plan: quote the mouse")]),
    )
    result = SimpleNamespace(
        new_items=[message, call, output],
        final_output={
            "answer": "Refund the mouse for $25.",
            "conversation": "I can leave the sleeve.",
        },
        raw_responses=[SimpleNamespace(usage=SimpleNamespace(input_tokens=8, output_tokens=3))],
    )
    extracted = extract_openai_result(result)
    assert extracted.plan == ["Plan: quote the mouse"]
    assert extracted.tool_calls[0].arguments["item"] == "mouse"
    assert extracted.tool_calls[0].output["amount"] == 25.0
    assert extracted.usage.total_tokens == 11
    assert "mouse" in extracted.output


def test_http_adapter(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"output": "done", "tool_calls": [{"name": "lookup", "arguments": {"q": "1"}}]}

    def fake_post(url: str, json: dict, headers: dict, timeout: float) -> _Response:
        assert url == "http://agent.test/run"
        assert headers["Authorization"] == "Bearer secret"
        assert json["input"] == "hi"
        assert timeout > 0
        return _Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("AGENT_TOKEN", "secret")
    adapter = HttpAdapter(
        AdapterConfig(
            type="http",
            url="http://agent.test/run",
            headers={"Authorization": "Bearer ${AGENT_TOKEN}"},
        )
    )
    result = adapter.run(AgentRequest(case_id="c", input="hi"))
    assert result.output == "done"
    assert result.tool_calls[0].name == "lookup"

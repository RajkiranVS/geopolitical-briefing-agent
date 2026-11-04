'''
Unit tests for how main() in geopolitical_news_agent.py reports the eval result.
'''

import importlib

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def agent_module(monkeypatch):
    # The module builds its model on import. No request is sent, but the client needs a key.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    module = importlib.import_module("geopolitical_news_agent")
    monkeypatch.setattr("builtins.input", lambda _: "Brief me")
    return module


def set_briefing(monkeypatch, module, text: str):
    class FakeAgent:
        def invoke(self, messages):
            return [*messages, AIMessage(content=text)]

    monkeypatch.setattr(module, "agent", FakeAgent())


def test_main_exits_1_when_eval_fails(agent_module, monkeypatch):
    set_briefing(monkeypatch, agent_module, "I can't produce a briefing today.")
    monkeypatch.setattr(agent_module, "run_evals", lambda *args: {"passed": False})
    with pytest.raises(SystemExit) as exit_info:
        agent_module.main()
    assert exit_info.value.code == 1


def test_main_returns_normally_when_eval_passes(agent_module, monkeypatch):
    set_briefing(monkeypatch, agent_module, "- Leaders met. https://www.bbc.com/news/world-123")
    monkeypatch.setattr(agent_module, "run_evals", lambda *args: {"passed": True})
    agent_module.main()

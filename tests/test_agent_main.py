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


def set_briefing(monkeypatch, module, text: str, fetched_urls: tuple[str, ...] = ()):
    class FakeAgent:
        def invoke(self, messages):
            # main() clears the tool state first, so fill it the way the tools do during a run.
            for url in fetched_urls:
                module.fetched[url] = "Article text"
                module.headline_meta[url] = "Mon, 02 Nov 2026 10:00:00 GMT"
            return [*messages, AIMessage(content=text)]

    monkeypatch.setattr(module, "agent", FakeAgent())


def stub_run_evals(monkeypatch, module, passed: bool) -> list[tuple]:
    calls = []

    def fake_run_evals(briefing, fetched, headline_meta, **kwargs):
        calls.append((briefing, fetched, headline_meta))
        return {"passed": passed}

    monkeypatch.setattr(module, "run_evals", fake_run_evals)
    return calls


def test_main_exits_1_when_eval_fails(agent_module, monkeypatch):
    set_briefing(monkeypatch, agent_module, "I can't produce a briefing today.")
    stub_run_evals(monkeypatch, agent_module, passed=False)
    with pytest.raises(SystemExit) as exit_info:
        agent_module.main()
    assert exit_info.value.code == 1


def test_main_returns_normally_when_eval_passes(agent_module, monkeypatch):
    set_briefing(monkeypatch, agent_module, "- Leaders met. https://www.bbc.com/news/world-123")
    stub_run_evals(monkeypatch, agent_module, passed=True)
    agent_module.main()


def test_main_passes_briefing_and_tool_state_to_evals(agent_module, monkeypatch):
    url = "https://www.bbc.com/news/world-123"
    briefing = f"- Leaders met. {url}"
    set_briefing(monkeypatch, agent_module, briefing, fetched_urls=(url,))
    calls = stub_run_evals(monkeypatch, agent_module, passed=True)
    agent_module.main()
    assert len(calls) == 1
    assert calls[0][0] == briefing
    assert calls[0][1] is agent_module.fetched
    assert calls[0][2] is agent_module.headline_meta
    assert url in agent_module.fetched
    assert url in agent_module.headline_meta


def test_main_with_real_evals_verifies_what_the_tools_fetched(agent_module, monkeypatch, tmp_path):
    # run_evals writes eval_report.json to the working directory.
    monkeypatch.chdir(tmp_path)
    url = "https://www.bbc.com/news/world-123"
    set_briefing(monkeypatch, agent_module, f"- Leaders met. {url}", fetched_urls=(url,))
    agent_module.main()
    assert (tmp_path / "eval_report.json").exists()


def test_main_clears_tool_state_from_the_previous_run(agent_module, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    url = "https://www.bbc.com/news/world-123"
    agent_module.fetched[url] = "Left over from an earlier run"
    set_briefing(monkeypatch, agent_module, f"- Leaders met. {url}")
    with pytest.raises(SystemExit) as exit_info:
        agent_module.main()
    assert exit_info.value.code == 1

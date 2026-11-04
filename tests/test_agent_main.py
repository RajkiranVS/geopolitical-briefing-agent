'''
Unit tests for how main() in geopolitical_news_agent.py reports the eval result.
'''

import importlib
from types import SimpleNamespace

import pytest
from feedparser import FeedParserDict
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


# Tool security

ARTICLE_HTML = "<html><body><article><p>Article text.</p></article></body></html>"
ARTICLE_URL = "https://www.bbc.com/news/world-123"


@pytest.fixture
def tools(agent_module, monkeypatch):
    agent_module.fetched.clear()
    agent_module.headline_meta.clear()
    monkeypatch.setattr(agent_module, "extract", lambda html: "Article text.")
    return agent_module


def fake_response(final_url: str, status: int = 200):
    return SimpleNamespace(url=final_url, status=status, html=ARTICLE_HTML)


def set_fetch_response(monkeypatch, module, response) -> list[dict]:
    requests = []

    def fake_fetch_response(url, **kwargs):
        requests.append({"url": url, **kwargs})
        return response

    monkeypatch.setattr(module, "fetch_response", fake_fetch_response)
    return requests


def set_feed(monkeypatch, module, final_url: str) -> list[str]:
    requests = []

    def fake_parse(url):
        requests.append(url)
        return FeedParserDict(
            href=final_url,
            entries=[FeedParserDict(title="T", published="Mon, 02 Nov 2026 10:00:00 GMT", link=ARTICLE_URL, summary="S")],
        )

    monkeypatch.setattr(module.feedparser, "parse", fake_parse)
    return requests


def test_fetch_article_returns_allowed_page(tools, monkeypatch):
    set_fetch_response(monkeypatch, tools, fake_response(ARTICLE_URL))
    assert tools.fetch_article.invoke({"url": ARTICLE_URL}).startswith("Article text.")
    assert ARTICLE_URL in tools.fetched


def test_fetch_article_never_requests_a_url_clients_parse_differently(tools, monkeypatch):
    requests = set_fetch_response(monkeypatch, tools, fake_response(ARTICLE_URL))
    result = tools.fetch_article.invoke({"url": "https://collector.example\\@bbc.com/a"})
    assert result.startswith("FETCH_FAILED")
    assert requests == []


def test_fetch_article_returns_failure_for_malformed_url(tools, monkeypatch):
    requests = set_fetch_response(monkeypatch, tools, fake_response(ARTICLE_URL))
    assert tools.fetch_article.invoke({"url": "http://[::1"}).startswith("FETCH_FAILED")
    assert requests == []


def test_fetch_article_discards_redirect_off_the_allowlist(tools, monkeypatch):
    # Regression: the off-list page reached the model and was recorded as fetched.
    set_fetch_response(monkeypatch, tools, fake_response("https://collector.example/page"))
    result = tools.fetch_article.invoke({"url": ARTICLE_URL})
    assert result.startswith("FETCH_FAILED")
    assert "Article text." not in result
    assert tools.fetched == {}


def test_fetch_article_keeps_redirect_within_the_allowlist(tools, monkeypatch):
    set_fetch_response(monkeypatch, tools, fake_response("https://www.bbc.co.uk/news/world-123"))
    assert tools.fetch_article.invoke({"url": ARTICLE_URL}).startswith("Article text.")


def test_fetch_article_rejects_error_status(tools, monkeypatch):
    set_fetch_response(monkeypatch, tools, fake_response(ARTICLE_URL, status=404))
    assert tools.fetch_article.invoke({"url": ARTICLE_URL}).startswith("FETCH_FAILED")
    assert tools.fetched == {}


def test_fetch_article_keeps_certificate_checks_on(tools, monkeypatch):
    requests = set_fetch_response(monkeypatch, tools, fake_response(ARTICLE_URL))
    tools.fetch_article.invoke({"url": ARTICLE_URL})
    config = requests[0]["config"]
    assert config.getboolean("DEFAULT", "INSECURE_SSL_FALLBACK") is False
    assert config.getboolean("DEFAULT", "SSRF_PROTECTION", fallback=True) is True


def test_get_headlines_accepts_listed_feeds(tools, monkeypatch):
    for feed_url in tools.ALLOWED_FEEDS:
        set_feed(monkeypatch, tools, feed_url)
        assert "Title: T" in tools.get_headlines.invoke({"feed_url": feed_url})


def test_get_headlines_refuses_other_urls_on_allowed_hosts(tools, monkeypatch):
    # Regression: any path on an allowed host was accepted, including open redirects.
    requests = set_feed(monkeypatch, tools, ARTICLE_URL)
    for url in ("https://news.google.com/rss/search?q=anything", "https://www.bbc.com/redirect?to=x", "http://[::1"):
        assert tools.get_headlines.invoke({"feed_url": url}).startswith("FETCH_FAILED")
    assert requests == []


def test_get_headlines_discards_redirect_off_the_allowlist(tools, monkeypatch):
    # Regression: feedparser followed a redirect to a loopback address and the items reached the model.
    set_feed(monkeypatch, tools, "http://127.0.0.1:8080/internal")
    result = tools.get_headlines.invoke({"feed_url": tools.ALLOWED_FEEDS[0]})
    assert result.startswith("FETCH_FAILED")
    assert "Title: T" not in result
    assert tools.headline_meta == {}

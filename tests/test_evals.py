'''
Unit tests for the runtime evaluation checks in evals.py.
'''

import email.utils
from datetime import datetime, timedelta, timezone

from evals import (
    DEFAULT_ALLOWED_DOMAINS,
    EMPTY_SECTION_PLACEHOLDER,
    check_cited_were_fetched,
    check_freshness,
    check_uncited_items,
    extract_citations,
    is_allowed_domain,
    run_evals,
)

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


def rfc2822(dt: datetime) -> str:
    return email.utils.format_datetime(dt)


# extract_citations and normalize_url

def test_strips_query_and_wrappers():
    briefing = "(Source: https://bbc.co.uk/news/articles/abc?at_medium=RSS)."
    assert extract_citations(briefing) == ["https://bbc.co.uk/news/articles/abc"]

def test_two_urls_one_line():
    briefing = "See https://www.bbc.com/news/a and https://www.thehindu.com/news/b."
    assert extract_citations(briefing) == [
        "https://www.bbc.com/news/a",
        "https://www.thehindu.com/news/b",
    ]

def test_url_at_end_keeps_ece():
    briefing = "Read more at https://www.thehindu.com/news/international/article123.ece."
    assert extract_citations(briefing) == ["https://www.thehindu.com/news/international/article123.ece"]

def test_markdown_link():
    briefing = "- Talks resumed [BBC](https://www.bbc.com/news/world-123)"
    assert extract_citations(briefing) == ["https://www.bbc.com/news/world-123"]

def test_markdown_italic_wrapped_url():
    # Regression: the closing '*' was captured as part of the URL.
    briefing = "*Source: https://www.thehindu.com/news/international/article71659244.ece*"
    assert extract_citations(briefing) == ["https://www.thehindu.com/news/international/article71659244.ece"]

def test_markdown_bold_wrapped_url():
    briefing = "**https://www.bbc.com/news/world-123**"
    assert extract_citations(briefing) == ["https://www.bbc.com/news/world-123"]

def test_underscore_inside_url_kept_trailing_stripped():
    briefing = "_See https://www.bbc.com/news/world_asia_123_"
    assert extract_citations(briefing) == ["https://www.bbc.com/news/world_asia_123"]

def test_lowercases_domain_keeps_path_case():
    briefing = "Source: https://WWW.BBC.CO.UK/News/World-Asia-123"
    assert extract_citations(briefing) == ["https://www.bbc.co.uk/News/World-Asia-123"]

def test_trailing_bracket_not_captured():
    briefing = "[Source: https://www.bbc.com/news/world-123]"
    assert extract_citations(briefing) == ["https://www.bbc.com/news/world-123"]

def test_trailing_exclamation_not_captured():
    briefing = "Read it at https://www.bbc.com/news/world-123!"
    assert extract_citations(briefing) == ["https://www.bbc.com/news/world-123"]

def test_curly_quotes_not_captured():
    briefing = "“https://www.bbc.com/news/world-123” and ‘https://www.bbc.com/news/world-456’"
    assert extract_citations(briefing) == [
        "https://www.bbc.com/news/world-123",
        "https://www.bbc.com/news/world-456",
    ]

def test_url_as_markdown_link_text_is_one_citation():
    briefing = "[https://www.bbc.com/news/world-123](https://www.bbc.com/news/world-123)"
    assert extract_citations(briefing) == ["https://www.bbc.com/news/world-123"]

def test_stacked_trailing_punctuation_not_captured():
    briefing = "(see [BBC: https://www.bbc.com/news/world-123]!)."
    assert extract_citations(briefing) == ["https://www.bbc.com/news/world-123"]


# check_cited_were_fetched

def test_fake_url_is_unverified():
    cited = ["https://www.bbc.com/news/real", "https://www.bbc.com/news/fake"]
    fetched = {"https://www.bbc.com/news/real?at_medium=RSS": "Article text"}
    verified, unverified = check_cited_were_fetched(cited, fetched)
    assert verified == ["https://www.bbc.com/news/real"]
    assert unverified == ["https://www.bbc.com/news/fake"]


# check_uncited_items

def test_source_on_next_line_passes():
    briefing = (
        "## Asia\n"
        "- Leaders met in Delhi to discuss trade.\n"
        "  Source: https://www.thehindu.com/news/article1.ece\n"
    )
    assert check_uncited_items(briefing) == []

def test_bullet_without_url_flagged():
    briefing = "## Europe\n- Ministers agreed a new sanctions package.\n"
    assert check_uncited_items(briefing) == ["- Ministers agreed a new sanctions package."]

def test_short_placeholder_passes():
    briefing = "## Africa\n- Nothing significant to report.\n"
    assert check_uncited_items(briefing) == []

def test_exact_empty_section_placeholder_passes():
    briefing = f"## India\n- {EMPTY_SECTION_PLACEHOLDER}\n"
    assert check_uncited_items(briefing) == []

def test_bullet_about_missing_coverage_flagged():
    # Regression: run 3 flagged this; the prompt now forbids it, so the eval should keep catching it.
    item = "- **Iran conflict:** The fetched articles do not discuss direct effects on India."
    assert check_uncited_items(f"## India\n{item}\n") == [item]

def test_long_sentence_with_placeholder_phrase_flagged():
    item = (
        "- Despite weeks of talks there were no significant developments, but officials "
        "now expect a ceasefire deal to be signed early next month."
    )
    assert check_uncited_items(f"## Middle East\n{item}\n") == [item]

def test_uncited_bullet_after_fetch_notes_passes():
    briefing = (
        "## Asia\n"
        "- Leaders met in Delhi. Source: https://www.thehindu.com/news/article1.ece\n"
        "\n"
        "FETCH NOTES:\n"
        "- The Google News article on the Nepal visit could not be fetched.\n"
    )
    assert check_uncited_items(briefing) == []

def test_uncited_bullet_before_fetch_notes_flagged():
    briefing = (
        "## Asia\n"
        "- Ministers agreed a new sanctions package.\n"
        "\n"
        "FETCH NOTES:\n"
        "- The BBC feed returned no entries.\n"
    )
    assert check_uncited_items(briefing) == ["- Ministers agreed a new sanctions package."]

def test_claim_containing_placeholder_phrase_flagged():
    # Regression: the placeholder match was a substring, so this short claim passed.
    item = "- Iran seized a tanker; no developments elsewhere."
    assert check_uncited_items(f"## Middle East\n{item}\n") == [item]

def test_wrapped_placeholder_passes():
    assert check_uncited_items(f'## India\n- **"{EMPTY_SECTION_PLACEHOLDER}"**\n') == []

def test_new_section_after_fetch_notes_is_checked():
    # Regression: everything after the marker was ignored, wherever it appeared.
    briefing = (
        "FETCH NOTES: none\n"
        "\n"
        "## Asia\n"
        "- Ministers agreed a new sanctions package.\n"
    )
    assert check_uncited_items(briefing) == ["- Ministers agreed a new sanctions package."]

def test_bold_section_after_fetch_notes_is_checked():
    briefing = (
        "FETCH NOTES: none\n"
        "\n"
        "**Asia**\n"
        "- Ministers agreed a new sanctions package.\n"
    )
    assert check_uncited_items(briefing) == ["- Ministers agreed a new sanctions package."]

def test_marker_inside_item_is_not_accepted():
    item = "- Ministers met. FETCH NOTES: they agreed a new sanctions package."
    assert check_uncited_items(f"## Asia\n{item}\n") == [item]

def test_claim_between_two_markers_flagged():
    briefing = (
        "FETCH NOTES:\n"
        "- Ministers agreed a new sanctions package.\n"
        "\n"
        "FETCH NOTES: none\n"
    )
    assert check_uncited_items(briefing) == ["- Ministers agreed a new sanctions package."]

def test_bold_fetch_notes_marker_accepted():
    briefing = (
        "## Asia\n"
        "- Leaders met in Delhi. Source: https://www.thehindu.com/news/article1.ece\n"
        "\n"
        "**FETCH NOTES:** none\n"
        "- The BBC feed returned no entries.\n"
    )
    assert check_uncited_items(briefing) == []

def test_lowercase_fetch_notes_marker_accepted():
    briefing = (
        "## Asia\n"
        "- Leaders met in Delhi. Source: https://www.thehindu.com/news/article1.ece\n"
        "\n"
        "Fetch notes:\n"
        "- The BBC feed returned no entries.\n"
    )
    assert check_uncited_items(briefing) == []

def test_sub_heading_inside_fetch_notes_accepted():
    # Regression: any heading after the marker made a correct briefing fail.
    for heading in ("**Feed issues:**", "### Feed issues"):
        briefing = (
            "FETCH NOTES:\n"
            f"{heading}\n"
            "- The BBC feed returned no entries.\n"
            "- The Google News article could not be fetched.\n"
        )
        assert check_uncited_items(briefing) == []

def test_sub_heading_with_none_inside_fetch_notes_accepted():
    assert check_uncited_items("FETCH NOTES:\n**Feed issues:**\n- None.\n") == []

def test_news_claim_beside_fetch_problem_under_heading_flagged():
    briefing = (
        "FETCH NOTES:\n"
        "- The BBC feed returned no entries.\n"
        "\n"
        "## Asia\n"
        "- Ministers agreed a new sanctions package.\n"
    )
    assert check_uncited_items(briefing) == [
        "- The BBC feed returned no entries.",
        "- Ministers agreed a new sanctions package.",
    ]


# is_allowed_domain

def test_google_news_is_allowed():
    assert is_allowed_domain("https://news.google.com/rss/articles/abc", DEFAULT_ALLOWED_DOMAINS)

def test_urls_that_clients_parse_differently_are_refused():
    # Regression: urlsplit reads these hosts as bbc.com, but urllib3 connected to collector.example.
    for url in (
        "https://collector.example\\@bbc.com/a",
        "https://collector.example\t@bbc.com/a",
        "https://collector.example\n@bbc.com/a",
        "https://collector.example%5c@bbc.com/a",
        "https://user:password@bbc.com/a",
        "https://www.bbc.com/news\\a",
    ):
        assert not is_allowed_domain(url, DEFAULT_ALLOWED_DOMAINS), url

def test_malformed_urls_are_refused_not_raised():
    # Regression: urlsplit raises ValueError on these, which crashed the run.
    for url in ("http://[::1", "http://[bad", "http://127.0.0.1／@bbc.com/a"):
        assert not is_allowed_domain(url, DEFAULT_ALLOWED_DOMAINS), url

def test_malformed_citation_is_unverified_not_raised():
    briefing = "- Talks resumed. http://127.0.0.1／@bbc.com/a"
    report = run_evals(briefing, {}, {}, report_path=None, now=NOW)
    assert report["passed"] is False
    assert report["unverified"] == ["http://127.0.0.1／@bbc.com/a"]

def test_normal_urls_are_still_allowed():
    for url in (
        "https://www.bbc.com/news/articles/abc?at_medium=RSS",
        "https://www.thehindu.com/news/international/article123.ece",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://news.google.com/rss/search?q=India+foreign+affairs&hl=en-IN&gl=IN&ceid=IN:en",
    ):
        assert is_allowed_domain(url, DEFAULT_ALLOWED_DOMAINS), url

def test_other_google_hosts_are_refused():
    # Regression: allowing google.com also allowed hosts that serve any content.
    for url in ("https://sites.google.com/view/x", "https://translate.google.com/x", "https://google.com/x"):
        assert not is_allowed_domain(url, DEFAULT_ALLOWED_DOMAINS)


# check_freshness

def test_four_day_old_is_stale():
    url = "https://www.bbc.com/news/old"
    meta = {url: rfc2822(NOW - timedelta(days=4))}
    stale, unknown = check_freshness([url], meta, now=NOW)
    assert stale == [url]
    assert unknown == []

def test_exactly_72_hours_is_not_stale():
    url = "https://www.bbc.com/news/boundary"
    meta = {url: rfc2822(NOW - timedelta(hours=72))}
    stale, unknown = check_freshness([url], meta, now=NOW)
    assert stale == []
    assert unknown == []

def test_missing_and_bad_dates_are_unknown():
    missing = "https://www.bbc.com/news/missing"
    marked_unknown = "https://www.bbc.com/news/unknown"
    garbage = "https://www.bbc.com/news/garbage"
    meta = {marked_unknown: "Unknown", garbage: "not a date"}
    stale, unknown = check_freshness([missing, marked_unknown, garbage], meta, now=NOW)
    assert stale == []
    assert unknown == [missing, marked_unknown, garbage]


# run_evals

def clean_inputs():
    url = "https://www.bbc.com/news/world-123"
    briefing = f"## Asia\n- Leaders met to discuss trade. (Source: {url})\n"
    fetched = {f"{url}?at_medium=RSS": "Leaders met to discuss trade."}
    headline_meta = {url: rfc2822(NOW - timedelta(hours=2))}
    return url, briefing, fetched, headline_meta

def test_clean_briefing_passes():
    _, briefing, fetched, headline_meta = clean_inputs()
    report = run_evals(briefing, fetched, headline_meta, report_path=None, now=NOW)
    assert report["passed"] is True
    assert report["verified_count"] == 1
    assert report["stale"] == []
    assert report["off_list"] == []

def test_custom_allowed_domains():
    url, briefing, fetched, headline_meta = clean_inputs()
    report = run_evals(briefing, fetched, headline_meta, report_path=None, allowed_domains=["reuters.com"], now=NOW)
    assert report["off_list"] == [url]

def test_report_output_is_ascii(capsys):
    # Regression: emoji in the status line crashed piped output on Windows (cp1252).
    _, briefing, fetched, headline_meta = clean_inputs()
    run_evals(briefing, fetched, headline_meta, report_path=None, now=NOW)
    assert capsys.readouterr().out.isascii()

def test_empty_briefing_fails():
    # Regression: with nothing to check, every check came back clean.
    report = run_evals("", {}, {}, report_path=None, now=NOW)
    assert report["passed"] is False
    assert report["verified_count"] == 0

def test_prose_only_briefing_fails():
    briefing = "Ministers agreed a new sanctions package, officials said on Monday."
    report = run_evals(briefing, {}, {}, report_path=None, now=NOW)
    assert report["passed"] is False

def test_refusal_fails_even_with_fetched_articles():
    _, _, fetched, headline_meta = clean_inputs()
    report = run_evals("I can't produce a briefing today.", fetched, headline_meta, report_path=None, now=NOW)
    assert report["passed"] is False

def test_all_empty_sections_fails():
    briefing = f"## Key developments\n- {EMPTY_SECTION_PLACEHOLDER}\n\n## India\n- {EMPTY_SECTION_PLACEHOLDER}\n"
    report = run_evals(briefing, {}, {}, report_path=None, now=NOW)
    assert report["uncited_items"] == []
    assert report["passed"] is False

def test_unverified_citation_fails_alongside_verified_one():
    _, briefing, fetched, headline_meta = clean_inputs()
    briefing += "- Talks collapsed. (Source: https://www.bbc.com/news/not-fetched)\n"
    report = run_evals(briefing, fetched, headline_meta, report_path=None, now=NOW)
    assert report["verified_count"] == 1
    assert report["passed"] is False

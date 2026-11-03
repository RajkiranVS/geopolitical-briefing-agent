'''
Unit tests for the runtime evaluation checks in evals.py.
'''

import email.utils
from datetime import datetime, timedelta, timezone

from evals import (
    EMPTY_SECTION_PLACEHOLDER,
    check_cited_were_fetched,
    check_freshness,
    check_uncited_items,
    extract_citations,
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

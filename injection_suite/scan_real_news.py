"""Counts false positives: runs find_injection_signs over today's real feeds and articles.

Uses the network but not the Anthropic API. It checks the same text the tools send to the model.
Any hit needs reading by hand, because a real attack would show up here too.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trafilatura import extract
from trafilatura.downloads import fetch_response

import geopolitical_news_agent as agent_module
from evals import find_injection_signs


def scan_article(link: str) -> tuple[str, list[str]] | None:
    """Downloads one article and checks the text the model would read.

    Args:
        link: The article's URL.

    Returns:
        The link and the signs found, or None if the article couldn't be read.
    """
    response = fetch_response(link, decode=True, config=agent_module.FETCH_CONFIG)
    if not response or response.status != 200 or not response.html:
        return None
    text = extract(response.html)
    if not text:
        return None
    return link, find_injection_signs(agent_module.trim_article(text))


def main() -> None:
    """Scans every item in the allowed feeds and every article they link to, then prints the counts."""
    links = []
    items = hits = 0
    for feed_url in agent_module.ALLOWED_FEEDS:
        # The agent's own download, so feeds that need its handling, such as India TV's, are read the same way
        feed, error = agent_module.download_feed(feed_url)
        if feed is None:
            print(f"FEED {error}")
            continue
        for entry in feed.entries:
            items += 1
            title = agent_module.clean_feed_text(entry.get("title", ""))
            summary = agent_module.clean_feed_text(entry.get("summary", ""))
            signs = find_injection_signs(f"{title}\n{summary}")
            if signs:
                hits += 1
                print(f"ITEM {signs}: {title[:100]}")
            if entry.get("link"):
                links.append(entry["link"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        articles = [result for result in pool.map(scan_article, links) if result]
    for link, signs in articles:
        if signs:
            print(f"ARTICLE {signs}: {link[:100]}")

    article_hits = sum(bool(signs) for _, signs in articles)
    print(f"Feed items: {items} scanned, {hits} flagged")
    print(f"Articles: {len(links)} linked, {len(articles)} downloaded, {article_hits} flagged")


if __name__ == "__main__":
    main()

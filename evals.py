'''
Runtime evaluation checks for the geopolitics briefing agent.
'''

import json
import re
from collections.abc import Sequence
from urllib.parse import urlsplit
import email.utils
from datetime import datetime, timezone

#Global Patterns
# '*', square brackets, and double or curly quotes are never part of a URL (Markdown
# and prose wrappers). '_' and '!' can appear inside URLs, so they are only rejected
# as the final character.
URL_PATTERN = r'https?://[^\s()<>*\[\]"“”‘’]+[^\s()<>.,:;\'"\?!*_\[\]“”‘’]'

# The prompt tells the agent to put feed/fetch problems after this line.
# Everything after it describes the run, not the news, so it needs no citation.
FETCH_NOTES_MARKER = "FETCH NOTES:"
EMPTY_SECTION_PLACEHOLDER = "Nothing to report from the fetched articles."
# Backslashes, whitespace and control characters are never valid in a URL we fetch.
UNSAFE_URL_CHARS = re.compile(r'[\\\s\x00-\x1f\x7f]')

# The marker only counts at the start of its own line (Markdown wrappers allowed, any case).
FETCH_NOTES_LINE = re.compile(rf'^[#*_ \t]*{re.escape(FETCH_NOTES_MARKER)}', re.MULTILINE | re.IGNORECASE)
# A Markdown heading, or a line that is entirely bold.
SECTION_HEADING_LINE = re.compile(r'^[ \t]*(#{1,6}[ \t]|\*\*[^*\n]+\*\*:?[ \t]*$)', re.MULTILINE)
LIST_ITEM_START = re.compile(r'^([-*•]|\d+\.)\s+')
# The prompt tells the agent to name the failing source, e.g. "BBC feed".
FETCH_PROBLEM_WORDS = re.compile(r'\b(feeds?|fetch|fetched|fetches|fetching)\b', re.IGNORECASE)

# A tripwire, not a guarantee: a blocklist of text that addresses the reading model instead of
# a human reader. Paraphrases, other languages and Unicode tricks get past it, and ordinary news
# can match it, so it only flags content for the report and the model. The guarantees come from
# what the tools can and can't do.
INJECTION_SIGNS = {
    "override instruction": re.compile(r'\b(ignore|disregard|forget)\b[^.\n]{0,30}\b(previous|prior|above|earlier|your|all)\b[^.\n]{0,20}\binstructions?\b', re.IGNORECASE),
    "system prompt request": re.compile(r'\bsystem prompt\b', re.IGNORECASE),
    "addressed to AI readers": re.compile(r'\b(ai|automated|language)\s+(agents?|systems?|models?|assistants?|summari[sz]ers?|readers?)\b[^.\n]{0,20}:[^\n]{0,120}\b(you must|must|call|fetch|fetch_article|get_headlines|add|ignore|do not|follow|end your|print|report that|state that)\b', re.IGNORECASE),
    "fake authority label": re.compile(r'\b(assistant|operator|system)\s+(instructions?|override|message)\s*:', re.IGNORECASE),
    "fake conversation turn": re.compile(r'^\s*(system|human|assistant)\s*:', re.IGNORECASE | re.MULTILINE),
    "fake delimiter": re.compile(r'</?\s*(untrusted\w*|system|instructions?)\b[^>]*>', re.IGNORECASE),
    "fake tool output": re.compile(r'\bFETCH_(LIMIT_REACHED|FAILED)\b|\[\d+\]\s*Title:'),
    "encoded payload": re.compile(r'[A-Za-z0-9+/]{60,}={0,2}'),
}
# These can make the API refuse the whole request, which would lose the briefing,
# so this content is withheld from the model instead of only being flagged.
WITHHELD_INJECTION_SIGNS = {"encoded payload"}

def find_injection_signs(text: str) -> list[str]:
    """
    Checks fetched text for signs of a prompt injection attempt.
    Args:
        text: Article text, or a feed item's title and summary.
    Returns:
        The labels of the signs found; empty if none.
    """
    # Long URLs, such as Google News links, would look like encoded payloads
    without_urls = re.sub(r'https?://\S+', ' ', text)
    return [label for label, pattern in INJECTION_SIGNS.items() if pattern.search(without_urls)]

def normalize_url(url: str) -> str:
    """
    Strip query string, fragments, trailing slashes and normalize the scheme and netloc to lowercase.
    Args:
        url: The URL to normalize.
    Returns:
        The normalized URL.
    """
    if not url:
        return ""
    # Split scheme, netloc, path, query, fragment
    try:
        split_url = urlsplit(url)
    except ValueError:
        # A malformed URL can't match a fetched one, so it is reported as unverified
        return url
    # Reconstruct url without query and fragment
    normalized = f"{split_url.scheme.lower()}://{split_url.netloc.lower()}{split_url.path}"
    # Remove trailing slashes
    normalized = normalized.rstrip("/")
    return normalized

def extract_citations(briefing: str) -> list[str]:
    """
    Finds every URL in the final text using regex and normalises them.
    Args:
        briefing: The text to search for URLs.
    Returns:
        A list of normalized URLs found in the text.
    """
    raw_urls = re.findall(URL_PATTERN, briefing)
    normalized_urls = []
    for url in raw_urls:
        normalized_urls.append(normalize_url(url))
    return sorted(set(normalized_urls))

def check_cited_were_fetched(cited: list[str], fetched: dict[str, str]) -> tuple[list[str], list[str]]:
    """
    Returns (verified, unverified). unverified items were cited but never fetched.
    Args:
        cited: List of cited URLs.
        fetched: Dict of {url: article_text} for all articles successfully fetched.
    Returns:
        A tuple containing two lists:
        - verified: URLs that were cited and successfully fetched.
        - unverified: URLs that were cited but not found in the fetched articles.
    """
    # Normalize fetched URLs keys for accurate comparison
    normalized_fetched = {normalize_url(url): text for url, text in fetched.items()}
    
    verified = []
    unverified = []
    
    for url in cited:
        if url in normalized_fetched:
            verified.append(url)
        else:
            unverified.append(url)
            
    return verified, unverified

def only_describes_fetch_problems(notes: str) -> bool:
    """
    Checks that every list item in the notes names a feed or fetch problem, or is "none".
    Args:
        notes: The text after the fetch notes marker.
    Returns:
        True if no list item looks like a news claim.
    """
    for line in notes.splitlines():
        stripped = line.strip()
        if not LIST_ITEM_START.match(stripped):
            continue
        item_text = LIST_ITEM_START.sub('', stripped).strip(' *_.').lower()
        if item_text != "none" and not FETCH_PROBLEM_WORDS.search(item_text):
            return False
    return True

def strip_fetch_notes(briefing: str) -> str:
    """
    Removes the fetch notes from the briefing, but only if they are its last section.
    A heading after the marker starts a new section, and nothing is removed, unless
    every list item after the marker describes a feed or fetch problem.
    Args:
        briefing: The text briefing content.
    Returns:
        The briefing text that must follow the citation rules.
    """
    markers = list(FETCH_NOTES_LINE.finditer(briefing))
    if not markers:
        return briefing
    last_marker = markers[-1]
    notes = briefing[last_marker.end():]
    if SECTION_HEADING_LINE.search(briefing, last_marker.end()) and not only_describes_fetch_problems(notes):
        return briefing
    return briefing[:last_marker.start()]

def check_uncited_items(briefing: str) -> list[str]:
    """
    Groups briefing items across line continuations and flags any list item
    that lacks a valid URL citation and is not an expected empty-section placeholder.
    A final FETCH_NOTES_MARKER section is ignored (see strip_fetch_notes).

    Args:
        briefing: The text briefing content to validate.
    Returns:
        A list of joined item strings that lack citations.
    """
    items = []
    current_item = []
    
    def save_current():
        if current_item:
            items.append(" ".join(current_item))
            current_item.clear()

    lines = strip_fetch_notes(briefing).splitlines()
    for line in lines:
        stripped = line.strip()
        
        # 1. Check for bullet or numbered list indicators
        if LIST_ITEM_START.match(stripped):
            save_current()
            current_item.append(stripped)
            
        # 2. Check for markdown headings or paragraph breaks
        elif stripped.startswith("##") or not stripped:
            save_current()
            
        # 3. Handle continuation lines if an item tracker is currently open
        elif current_item:
            current_item.append(stripped)

    # Save the final residual item block if one remains open
    save_current()

    # Evaluation and screening criteria
    flagged_items = []

    # Standard placeholder variants
    allowed_placeholders = {
        EMPTY_SECTION_PLACEHOLDER.lower().rstrip("."),
        "nothing significant",
        "nothing significant to report",
        "no developments",
        "nothing to report",
        "no significant developments"
    }

    for item in items:
        # Check if the combined item context contains a valid URL
        if not re.search(URL_PATTERN, item):
            # The whole item, minus its bullet and Markdown wrappers, must be an allowed phrase
            item_text = LIST_ITEM_START.sub('', item).lower()
            item_text = item_text.strip(' *_"“”.')

            if item_text not in allowed_placeholders:
                flagged_items.append(item)

    return flagged_items

def check_freshness(cited: list[str], headline_meta: dict[str, str], max_hours: int = 72, now: datetime | None = None) -> tuple[list[str], list[str]]:
    """
    Flags citations older than max_hours. Checks timezone-aware datetimes.
    Args:
        cited: List of cited URLs.
        headline_meta: Dict of {url: published_date} for all headlines fetched.
        max_hours: Maximum age in hours before a citation is considered stale.
        now: Reference time for age calculation; defaults to the current UTC time.
    Returns:
        A tuple (stale, unknown_date): URLs older than max_hours, and URLs whose
        published date is missing, "Unknown", or unparseable.
    """
    stale_urls = []
    unknown_date_urls = []
    now = now or datetime.now(timezone.utc)
    
    # Create normalized mapping for meta lookup
    normalized_meta = {normalize_url(url): pub_date for url, pub_date in headline_meta.items()}
    
    for url in cited:
        pub_date_str = normalized_meta.get(normalize_url(url))
        if not pub_date_str or pub_date_str == "Unknown":
            unknown_date_urls.append(url)
            continue
        try:
            pub_date = email.utils.parsedate_to_datetime(pub_date_str)
            # Ensure timezone awareness
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
                
            age_in_hours = (now - pub_date).total_seconds() / 3600.0
            if age_in_hours > max_hours:
                stale_urls.append(url)
        except (TypeError, ValueError):
            unknown_date_urls.append(url)

    return stale_urls, unknown_date_urls

def is_allowed_domain(url: str, allowed_domains: Sequence[str]) -> bool:
    """
    Checks if the domain of the given URL is in the allowed domains list.
    Args:
        url: The URL to check.
        allowed_domains: List of allowed domain strings (e.g., "bbc.co.uk").
    Returns:
        True if the domain is allowed, False otherwise.
    """
    # HTTP clients and urlsplit can disagree about the host of these URLs.
    # 'https://evil.example\@bbc.com' parses as bbc.com here but connects to evil.example.
    if UNSAFE_URL_CHARS.search(url):
        return False

    try:
        split_url = urlsplit(url)
        domain = split_url.hostname
    except ValueError:
        return False

    if "@" in split_url.netloc:
        return False

    schema = split_url.scheme.lower()

    if schema not in ("http", "https") or not domain:
        return False    
    
    # Check if domain matches or matches subdomains of allowed list
    return any(domain == allowed or domain.endswith("." + allowed) for allowed in allowed_domains)



def check_allowed_sources(cited: list[str], allowed_domains: Sequence[str]) -> list[str]:
    """
    Flags any domain outside your feed host list.
    Args:
        cited: List of cited URLs.
        allowed_domains: List of allowed domain strings (e.g., "bbc.co.uk").
    Returns:
        A list of URLs that are outside the allowed domains.
    """
    return [url for url in cited if not is_allowed_domain(url, allowed_domains)]

def check_grounding(item: str, article_text: str) -> list[str]:
    """
    Extracts entities (Numbers & Capitalised words) and verifies presence in source text.
    Args:
        item: The text item to check for grounding.
        article_text: The full text of the article to check against.
    Returns:
        A list of entities that are present in the item but missing from the article text.
    """
    if not article_text:
        return ["Article text missing for comparison"]
        
    flagged_entities = []
    # Extract numbers (e.g. 5,000, 19, 2026)
    numbers = re.findall(r'\b\d+(?:,\d+)?\b', item)
    # Extract Title Case Words (excluding line starting constructs)
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]+\b', item)
    
    for entity in set(numbers + capitalized):
        if entity not in article_text:
            flagged_entities.append(entity)
            
    return flagged_entities

DEFAULT_ALLOWED_DOMAINS = ("bbc.co.uk", "bbci.co.uk", "bbc.com", "thehindu.com", "news.google.com")

def run_evals(briefing: str, fetched: dict[str, str], headline_meta: dict[str, str], report_path: str | None = "eval_report.json", allowed_domains: Sequence[str] = DEFAULT_ALLOWED_DOMAINS, now: datetime | None = None, suspected_injections: Sequence[str] = ()) -> dict:
    """
    Orchestrates all metrics and writes execution stats to eval_report.json if report_path is given.
    Args:
        briefing: The final text output of the agent.
        fetched: A dict of {url: article_text} for all articles successfully fetched.
        headline_meta: A dict of {url: published_date} for all headlines fetched.
        report_path: The path where the evaluation report will be saved.
        allowed_domains: Domains citations may come from; defaults to DEFAULT_ALLOWED_DOMAINS.
        now: Reference time for the freshness check; defaults to the current UTC time.
        suspected_injections: Fetched content the tools flagged or withheld as likely prompt injection.
    Returns:
        A dict containing evaluation metrics and pass/fail status.
    """
    
    cited_urls = extract_citations(briefing)
    verified, unverified = check_cited_were_fetched(cited_urls, fetched)
    uncited_items = check_uncited_items(briefing)
    stale, unknown_date = check_freshness(cited_urls, headline_meta, now=now)
    off_list = check_allowed_sources(cited_urls, allowed_domains)
    
    # Strict validation rule: a briefing with nothing verified has proved nothing
    passed = len(verified) > 0 and not unverified and not uncited_items
    
    report = {
        "verified_count": len(verified),
        "unverified": unverified,
        "uncited_items": uncited_items,
        "stale": stale,
        "unknown_date": unknown_date,
        "off_list": off_list,
        "suspected_injections": list(suspected_injections),
        "passed": passed
    }
    
    # Save the output file
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        
    # Print a short summary summary table/log
    print("\n" + "="*40 + " EVALUATION REPORT " + "="*40)
    print(f"Status: {'[PASSED]' if passed else '[FAILED]'}")
    print(f"Verified Citations: {len(verified)}")
    print(f"Cited but not read: {len(unverified)}")
    print(f"Uncited Bullet Items Found: {len(uncited_items)}")
    print(f"Stale news warnings (>72h): {len(stale)}")
    print(f"Unknown publish date warnings: {len(unknown_date)}")
    print(f"Out of Bounds Domains: {len(off_list)}")
    print(f"Suspected prompt injections: {len(suspected_injections)}")
    print("=" * 99 + "\n")
    
    return report

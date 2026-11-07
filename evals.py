"""Runtime evaluation checks for the geopolitics briefing agent."""

import email.utils
import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from urllib.parse import urlsplit

#Global Patterns
# '*', backticks, square brackets, and double or curly quotes are never part of a URL (Markdown
# and prose wrappers). '_' and '!' can appear inside URLs, so they are only rejected
# as the final character.
URL_PATTERN = r'https?://[^\s()<>*`\[\]"“”‘’]+[^\s()<>.,:;\'"\?!*_`\[\]“”‘’]'

# The prompt tells the agent to put feed/fetch problems after this line.
# Everything after it describes the run, not the news, so it needs no citation.
FETCH_NOTES_MARKER = "FETCH NOTES:"
EMPTY_SECTION_PLACEHOLDER = "Nothing to report from the fetched articles."
# Models reword the placeholder ("Nothing further from the fetched articles bears directly on India").
# It has to match the whole item, and the topic comes from a fixed list, because any free text there could be a news claim.
REWORDED_PLACEHOLDER = re.compile(r"nothing(\s+(further|else|more))?(\s+to\s+report)?\s+from\s+the\s+fetched\s+articles(\s+(bears?\s+directly\s+on|on|about|concerning|for)\s+india(['’]s\s+(direct\s+)?foreign\s+(policy|affairs|relations))?)?")
# Backslashes, whitespace and control characters are never valid in a URL we fetch.
UNSAFE_URL_CHARS = re.compile(r'[\\\s\x00-\x1f\x7f]')

# The marker only counts at the start of its own line (Markdown wrappers allowed, any case).
# Bold can close before the colon, as in **FETCH NOTES**:, or after it, as in **FETCH NOTES:**
FETCH_NOTES_LINE = re.compile(rf'^[#*_ \t]*{re.escape(FETCH_NOTES_MARKER[:-1])}[*_ \t]*:[*_ \t]*', re.MULTILINE | re.IGNORECASE)
LIST_ITEM_START = re.compile(r'^([-*•]|\d+\.)\s+')
# A line that only labels what follows: a Markdown heading, a bold title with an optional
# bracketed note, as in "**Today's briefing** (feeds dated 5 Nov)", a short line ending in a
# colon ("Key developments:", "Here is your briefing:"), an intro with no numbers ("Here is
# today's briefing."), a short line in capitals, or a rule. "Source:" isn't a label: it belongs
# to the item before it.
LABEL_LINE = re.compile(
    r"^(#{1,6}\s.*|\*\*[^*\n]+\*\*\s*(\([^()\n]*\))?\s*:?|(?!(?i:\(?sources?\b))[^\n]{1,120}:"
    r"|(?i:here is|here's|below is|this is|this briefing)\b[^\d\n]{0,100}|[A-Z0-9 &'’,/()-]{3,60}|[-*_=]{3,})$"
)
# A citation in its own paragraph, as in "Source: https://...", belongs to the item before it.
SOURCE_LINE = re.compile(r'^\(?\s*sources?\b', re.IGNORECASE)
# Fetch notes name a source or a step of the run. Every line of the notes must contain one of
# these words, so a plain news claim there is caught. A claim that also names its source,
# such as "The BBC article says the talks collapsed", still gets through.
FETCH_PROBLEM_WORDS = re.compile(
    r'\b(feeds?|fetch\w*|articles?|headlines?|links?|instructions?|ignored|redirect\w*|briefing|'
    r'summari[sz]\w*|items?|loaded|retrieved|retry|retried|withheld|none)\b',
    re.IGNORECASE,
)

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
URL_IN_TEXT = re.compile(r'https?://\S+')
# Google News links are encoded article ids, so they always look like a payload.
GOOGLE_NEWS_DOMAINS = ("news.google.com",)
# 60 characters of base64 hold about 24 capitals and 24 lower case letters, and fewer than 12
# of either is about a 1 in 1,000 chance. A URL path of capitalised words has one capital per
# segment, so it would need 12 or more segments to reach this.
MIXED_CASE_MINIMUM = 12

def find_injection_signs(text: str) -> list[str]:
    """Checks fetched text for signs of a prompt injection attempt.

    Args:
        text: Article text, or a feed item's title and summary.

    Returns:
        The labels of the signs found; empty if none.
    """
    without_urls = URL_IN_TEXT.sub(lambda url: url.group() if looks_encoded(url.group()) else " ", text)
    return [label for label, pattern in INJECTION_SIGNS.items() if pattern.search(without_urls)]

def looks_encoded(url: str) -> bool:
    """Checks whether a URL in fetched text hides an encoded payload.

    Google News links are skipped, because their article ids look like encoded text. On every
    other host, including the other allowed ones, a long run only counts if it mixes capitals
    and lower case the way encoded text does, so an ordinary long path such as
    /security-council/resolutions/2026 doesn't.

    Args:
        url: A URL found in fetched text.

    Returns:
        True if the URL should stay in the text for the encoded payload check.
    """
    if is_allowed_domain(url, GOOGLE_NEWS_DOMAINS):
        return False
    for run in INJECTION_SIGNS["encoded payload"].findall(url):
        if sum(c.isupper() for c in run) >= MIXED_CASE_MINIMUM and sum(c.islower() for c in run) >= MIXED_CASE_MINIMUM:
            return True
    return False

def normalize_url(url: str) -> str:
    """Strips the query string, fragment and trailing slashes, and lowercases the scheme and host.

    Args:
        url: The URL to normalize.

    Returns:
        The normalized URL. An empty URL gives an empty string, and a malformed URL comes back
        unchanged so that it can't match a fetched one.
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
    """Finds every URL in the final text using regex and normalizes them.

    Args:
        briefing: The text to search for URLs.

    Returns:
        The sorted, unique, normalized URLs found in the text.
    """
    raw_urls = re.findall(URL_PATTERN, briefing)
    normalized_urls = []
    for url in raw_urls:
        normalized_urls.append(normalize_url(url))
    return sorted(set(normalized_urls))

def check_cited_were_fetched(cited: list[str], fetched: dict[str, str]) -> tuple[list[str], list[str]]:
    """Splits the cited URLs into those that were fetched and those that weren't.

    Args:
        cited: List of cited URLs.
        fetched: Dict of {url: article_text} for all articles successfully fetched.

    Returns:
        A tuple (verified, unverified): URLs cited and fetched, and URLs cited but never fetched.
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
    """Checks that every line of the notes names a source or a step of the run, or is "none".

    Args:
        notes: The text after the fetch notes marker.

    Returns:
        True if no line looks like a news claim.
    """
    for line in notes.splitlines():
        stripped = line.strip()
        if not stripped or LABEL_LINE.match(stripped):
            continue
        if not FETCH_PROBLEM_WORDS.search(LIST_ITEM_START.sub('', stripped)):
            return False
    return True

def is_notes_marker_line(line: str) -> bool:
    """Checks whether a line is a fetch notes marker followed only by a description of the run.

    Args:
        line: One line of the briefing, without surrounding whitespace.

    Returns:
        True for a line such as "FETCH NOTES: none", False for any other line.
    """
    marker = FETCH_NOTES_LINE.match(line)
    return bool(marker) and only_describes_fetch_problems(line[marker.end():].strip(" *_"))

def strip_fetch_notes(briefing: str) -> str:
    """Removes the fetch notes from the briefing, but only if they are its last section.

    Nothing is removed unless every line after the marker describes the run, so a news
    claim placed in the notes, or in a section after them, is still checked.

    Args:
        briefing: The text briefing content.

    Returns:
        The briefing text that must follow the citation rules.
    """
    markers = list(FETCH_NOTES_LINE.finditer(briefing))
    if not markers:
        return briefing
    last_marker = markers[-1]
    if not only_describes_fetch_problems(briefing[last_marker.end():]):
        return briefing
    return briefing[:last_marker.start()]

def check_uncited_items(briefing: str) -> list[str]:
    """Flags list items and prose paragraphs that have no URL citation and aren't a placeholder.

    Items are grouped across line continuations. Headings, bold titles and horizontal rules
    need no citation. A final FETCH_NOTES_MARKER section is ignored (see strip_fetch_notes).

    Args:
        briefing: The text briefing content to validate.

    Returns:
        A list of joined item strings that lack citations.
    """
    items = []
    current_item = []
    
    def save_current():
        """Moves the item being built into the finished items, if there is one."""
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
            
        # 2. Headings, titles, rules, paragraph breaks and a marker line such as "FETCH NOTES: none"
        # end the item and need no citation
        elif not stripped or LABEL_LINE.match(stripped) or is_notes_marker_line(stripped):
            save_current()

        # A citation in its own paragraph belongs to the item before it
        elif not current_item and items and SOURCE_LINE.match(stripped):
            items[-1] += " " + stripped

        # 3. Anything else continues the open item, or starts a prose paragraph, which needs a citation too
        else:
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

            if item_text not in allowed_placeholders and not REWORDED_PLACEHOLDER.fullmatch(item_text):
                flagged_items.append(item)

    return flagged_items

def check_freshness(cited: list[str], headline_meta: dict[str, str], max_hours: int = 72, now: datetime | None = None) -> tuple[list[str], list[str]]:
    """Flags citations older than max_hours, using timezone-aware datetimes.

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
    """Checks if the domain of the given URL is in the allowed domains list.

    Args:
        url: The URL to check.
        allowed_domains: List of allowed domain strings (e.g., "bbc.co.uk").

    Returns:
        True if the domain is allowed, False otherwise, including for malformed URLs.
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
    """Flags any domain outside the feed host list.

    Args:
        cited: List of cited URLs.
        allowed_domains: List of allowed domain strings (e.g., "bbc.co.uk").

    Returns:
        A list of URLs that are outside the allowed domains.
    """
    return [url for url in cited if not is_allowed_domain(url, allowed_domains)]

def check_grounding(item: str, article_text: str) -> list[str]:
    """Extracts numbers and capitalised words from an item and looks for them in the source text.

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
    """Runs all the checks, prints a summary and writes the report if report_path is given.

    Args:
        briefing: The final text output of the agent.
        fetched: A dict of {url: article_text} for all articles successfully fetched.
        headline_meta: A dict of {url: published_date} for all headlines fetched.
        report_path: The path where the evaluation report will be saved; None skips writing it.
        allowed_domains: Domains citations may come from; defaults to DEFAULT_ALLOWED_DOMAINS.
        now: Reference time for the freshness check; defaults to the current UTC time.
        suspected_injections: Fetched content the tools flagged or withheld as likely prompt injection.

    Returns:
        A dict containing evaluation metrics and pass/fail status.

    Raises:
        OSError: If the report file can't be written.
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
    print(f"Uncited items and paragraphs: {len(uncited_items)}")
    print(f"Stale news warnings (>72h): {len(stale)}")
    print(f"Unknown publish date warnings: {len(unknown_date)}")
    print(f"Out of Bounds Domains: {len(off_list)}")
    print(f"Suspected prompt injections: {len(suspected_injections)}")
    print("=" * 99 + "\n")
    
    return report

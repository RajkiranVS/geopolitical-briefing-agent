"""Runtime evaluation checks for the geopolitics briefing agent.

The model submits the briefing as structured data (see briefing.py), so each item arrives with its
source as a separate field. The checks here work on that structure: they don't parse Markdown.
Text matching is left only where the text itself is what is being checked: quotes, names and
numbers against the article, links inside a claim, and fetch notes.
"""

import email.utils
import json
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from urllib.parse import urlsplit

from briefing import BriefingItem, SubmitBriefing

# Only these hosts may be fetched or cited.
DEFAULT_ALLOWED_DOMAINS = ("bbc.co.uk", "bbci.co.uk", "bbc.com", "thehindu.com", "indiatvnews.com", "timesofindia.indiatimes.com")

# '*', backticks, square brackets, pipes and double or curly quotes are never part of a URL
# (Markdown, table and prose wrappers). '_' and '!' can appear inside URLs, so they are only
# rejected as the final character.
URL_PATTERN = r'https?://[^\s()<>*`|\[\]"“”‘’]+[^\s()<>.,:;\'"\?!*_`|\[\]“”‘’]'
URL_IN_TEXT = re.compile(r'https?://\S+')
# Backslashes, whitespace and control characters are never valid in a URL we fetch.
UNSAFE_URL_CHARS = re.compile(r'[\\\s\x00-\x1f\x7f]')

# Fetch notes name a source or a step of the run. Each note must contain one of these words, so a
# plain news claim there is caught. A claim that also names its source, such as "The BBC article
# says the talks collapsed", still gets through.
FETCH_PROBLEM_WORDS = re.compile(
    r'\b(feeds?|fetch\w*|articles?|headlines?|links?|instructions?|ignored|redirect\w*|briefing|'
    r'summari[sz]\w*|items?|loaded|retrieved|retry|retried|withheld|none)\b',
    re.IGNORECASE,
)
# A note that only says nothing went wrong, as in "No issues." It must be the whole note, so a
# claim such as "Iran faced no problems at the border" isn't accepted.
NO_PROBLEMS_NOTE = re.compile(r'no\s+(issues?|problems?|errors?)(\s+(to\s+report|found|encountered))?', re.IGNORECASE)
# A fetch note should name something that went wrong. "Problems" and "issues" aren't on the list,
# because notes such as "read without problems" use them to say nothing went wrong.
FAILURE_WORDS = re.compile(
    # A status code counts only after a word such as "returned" or "HTTP", and not before "word" or a
    # hyphen, so "450 words" and "a 500-word summary" aren't one.
    # FETCH_\w+ is listed because the underscore in FETCH_FAILED stops "fail" from matching there.
    r"\b(fail\w*|fetch_\w+|error\w*|(?:http|status|code|returned|gave|got|an?)\W{1,3}[45]\d\d(?![-\d]|\s+(?:words?|characters?)\b)|forbidden|"
    r"not found|time[sd]?\s?out|withheld|ignored|none|empty|blank|truncated|cut off|could not|couldn't|"
    r"cannot|can't|unable|unavailable|unreadable|refused|blocked|redirect\w*|paywall\w*|no usable|"
    # "No articles" counts only after a verb such as "returned": notes that say all went well use
    # it too, as in "no articles from them were fetched". "Found" isn't one of the verbs, because
    # "found no articles older than a day" reports a good result.
    r"(?:had|has|returned|gave|got|with|contained|yielded)\s+no\s+(?:items|headlines|articles|text|content))\b",
    re.IGNORECASE,
)

# The schema asks for 3 to 5 key developments, fewer if fewer articles were fetched. A briefing with
# fewer items than this, counting both sections, gets a warning.
MIN_ITEMS = 3

# Two items are the same item when most of the words of the shorter one are also in the other. In 26
# live briefings no two items shared more than 23 per cent, so 60 per cent leaves a wide margin.
REPEAT_OVERLAP = 0.6
REPEAT_MIN_WORDS = 6
REPEAT_IGNORED_WORDS = {
    "the", "a", "an", "of", "in", "on", "and", "to", "said", "says", "that", "for", "with", "at",
    "is", "was", "by", "its", "has", "from",
}
# An item in the India section should say how it concerns India. Names of the Prime Minister and the
# External Affairs Minister count too: India items about the Prime Minister often name only him.
INDIA_TERMS = re.compile(
    r"\b(India|Indian|Indians|Indo-\w+|New Delhi|Delhi|MEA|Ministry of External Affairs|Modi|Jaishankar)\b"
)

# Numbers for the grounding check: digits with optional thousands separators and decimals.
GROUNDING_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")
NUMBER_WORDS = {
    word: str(value) for value, word in enumerate([
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty",
    ])
} | {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90"}
# A tens word joined to a units word, as in "twenty-five" or "forty two".
COMPOUND_NUMBER = re.compile(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ](one|two|three|four|five|six|seven|eight|nine)\b")
# An abbreviation written with full stops, such as "U.S.", so it can match "US".
DOTTED_ABBREVIATION = re.compile(r"\b(?:[A-Z]\.){2,}")
# A date such as "7 November" or "November 7th". The model often takes dates from the feed's publish
# date rather than the article, so the grounding check leaves them out.
MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
DATE_IN_TEXT = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\b|\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.IGNORECASE)
# Capitalised words that aren't facts about the story: the outlets the briefing cites, and common
# words that are only capitalised because they start a sentence.
GROUNDING_IGNORED_WORDS = {
    "source", "bbc", "hindu", "google", "times", "toi", "tv", "new", "according", "key", "development",
    # A title the article usually writes out in full, as "Prime Minister"
    "pm",
    "the", "this", "that", "these", "those", "its", "it", "he", "she", "they", "we", "his", "her", "their",
    "per", "in", "on", "at", "as", "an", "and", "but", "also", "while", "after", "before", "however",
    "for", "from", "with", "by", "of", "to", "no", "not", "both", "all", "some", "other", "there", "one",
    # Dates come from the feed's publish date as often as from the article, so they aren't checked
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
}

# A quote on one line, in double quotes (groups 1 and 2) or single quotes (groups 3 and 4), straight
# or curly. A single quote mark only opens a quote after a non-letter and only closes one before a
# non-letter, so the apostrophes in "don't" and "India's" aren't read as quote marks.
QUOTE_PATTERN = re.compile(
    r'"([^"\n]{3,500})"|“([^”\n]{3,500})”'
    r"|(?<!\w)'(?=\S)([^\n]{3,500}?)(?<=\S)'(?!\w)|(?<!\w)‘(?=\S)([^\n]{3,500}?)(?<=\S)’(?!\w)"
)
# Single quotes are also used for scare quotes and names, so they only count from three words.
MIN_WORDS_DOUBLE_QUOTE = 2
MIN_WORDS_SINGLE_QUOTE = 3
# Punctuation at either end of a quote, such as a full stop placed inside the quotation marks.
QUOTE_EDGE_PUNCTUATION = " .,;:!?'"
# Where a writer changed a quote: an ellipsis for left-out text, or square brackets for an edit, as
# in "want[s]". The parts around it must still appear in the article, in order.
QUOTE_EDIT = re.compile(r'\[[^\]\n]{0,40}\]|\.\.\.|…')

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

    A long run only counts if it mixes capitals and lower case the way encoded text does, so an
    ordinary long path such as /security-council/resolutions/2026 doesn't. Links with encoded ids,
    such as Google News article links, do count, so an article that quotes one is withheld.

    Args:
        url: A URL found in fetched text.

    Returns:
        True if the URL should stay in the text for the encoded payload check.
    """
    for run in INJECTION_SIGNS["encoded payload"].findall(url):
        if sum(c.isupper() for c in run) >= MIXED_CASE_MINIMUM and sum(c.islower() for c in run) >= MIXED_CASE_MINIMUM:
            return True
    return False

def normalize_url(url: str) -> str:
    """Strips the query string, fragment and trailing slashes, and lowercases the scheme and host.

    http and https give the same result, because the same article is often linked both ways.

    Args:
        url: The URL to normalize.

    Returns:
        The normalized URL. An empty URL gives an empty string, and a malformed URL comes back
        unchanged so that it can't match a fetched one.
    """
    if not url:
        return ""
    try:
        split_url = urlsplit(url)
    except ValueError:
        # A malformed URL can't match a fetched one, so it is reported as unverified
        return url
    scheme = split_url.scheme.lower()
    if scheme == "http":
        scheme = "https"
    return f"{scheme}://{split_url.netloc.lower()}{split_url.path}".rstrip("/")

def item_source(item: BriefingItem) -> str:
    """Gives an item's normalized source link, or an empty string if its source isn't a link.

    Args:
        item: A briefing item.

    Returns:
        The normalized link, or "" when source_url is empty or plain text such as "The Hindu".
    """
    source = item.source_url.strip()
    # Schemes are case-insensitive, so HTTPS:// is a link too
    return normalize_url(source) if source.lower().startswith("http") else ""

def site_of(url: str, allowed_domains: Sequence[str]) -> str:
    """Names the site a link is on: the allowed domain it belongs to, or else its host without "www.".

    Args:
        url: The link.
        allowed_domains: The allowed domains, so that subdomains such as feeds.bbc.co.uk count as bbc.co.uk.

    Returns:
        The site name.
    """
    host = (urlsplit(url).hostname or "").removeprefix("www.")
    return next((allowed for allowed in allowed_domains if host == allowed or host.endswith("." + allowed)), host)

def extract_citations(text: str) -> list[str]:
    """Finds every URL in a text and normalizes it.

    Args:
        text: The text to search for URLs.

    Returns:
        The sorted, unique, normalized URLs found in the text.
    """
    return sorted({normalize_url(url) for url in re.findall(URL_PATTERN, text)})

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
    if split_url.scheme.lower() not in ("http", "https") or not domain:
        return False
    # The domain itself, or any subdomain of it
    return any(domain == allowed or domain.endswith("." + allowed) for allowed in allowed_domains)

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
    now = now or datetime.now(UTC)
    normalized_meta = {normalize_url(url): pub_date for url, pub_date in headline_meta.items()}
    for url in cited:
        pub_date_str = normalized_meta.get(normalize_url(url))
        if not pub_date_str or pub_date_str == "Unknown":
            unknown_date_urls.append(url)
            continue
        try:
            pub_date = email.utils.parsedate_to_datetime(pub_date_str)
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=UTC)
            if (now - pub_date).total_seconds() / 3600.0 > max_hours:
                stale_urls.append(url)
        except (TypeError, ValueError):
            unknown_date_urls.append(url)
    return stale_urls, unknown_date_urls

def repeat_words(claim: str) -> set[str]:
    """Gives the words of a claim that count when comparing it with another.

    Args:
        claim: The claim.

    Returns:
        Its lowercased words, without common words.
    """
    return set(re.findall(r"[a-z0-9]+", claim.lower())) - REPEAT_IGNORED_WORDS

def claims_repeat(first: set[str], second: set[str]) -> bool:
    """Checks whether two claims, given as their repeat_words, say the same thing.

    Args:
        first: The words of one claim.
        second: The words of the other.

    Returns:
        True if most of the words of the shorter claim are also in the other.
    """
    shorter = min(len(first), len(second))
    # A very short item can have all its words in a longer one without being the same item
    return shorter >= REPEAT_MIN_WORDS and len(first & second) / shorter >= REPEAT_OVERLAP

def check_repeated_items(briefing: SubmitBriefing) -> list[str]:
    """Finds items that repeat another item, in the same section or across the two.

    Args:
        briefing: The submitted briefing.

    Returns:
        One entry per repeated pair: the start of each claim.
    """
    items = [*briefing.key_developments, *briefing.india]
    word_sets = [repeat_words(item.claim) for item in items]
    repeated = []
    for first in range(len(items)):
        for second in range(first + 1, len(items)):
            if claims_repeat(word_sets[first], word_sets[second]):
                repeated.append(f"{items[first].claim[:60]} | {items[second].claim[:60]}")
    return repeated

def check_fetch_notes(notes: Sequence[str]) -> list[str]:
    """Flags fetch notes that don't describe the run, so news can't hide in them.

    Args:
        notes: The fetch notes the model submitted.

    Returns:
        The notes that name no source or step of the run, and don't only say nothing went wrong.
    """
    return [
        note for note in notes
        if not FETCH_PROBLEM_WORDS.search(note) and not NO_PROBLEMS_NOTE.fullmatch(note.strip(" *_.!"))
    ]

def normalize_quote_text(text: str) -> str:
    """Makes text comparable for the quote check.

    Curly quotes and apostrophes become straight, dashes become hyphens, whitespace collapses
    and case is ignored, because these change between the article and the briefing without
    changing the words.

    Args:
        text: An article or a quote.

    Returns:
        The normalized text.
    """
    for old, new in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")):
        text = text.replace(old, new)
    return " ".join(text.split()).lower()

def quotes_in(text: str) -> list[str]:
    """Finds the quotes long enough to check in a claim.

    Args:
        text: One claim from the briefing.

    Returns:
        Each quote without its quotation marks or the punctuation at its ends.
    """
    quotes = []
    for match in QUOTE_PATTERN.finditer(text):
        double = match.group(1) or match.group(2)
        quote = (double or match.group(3) or match.group(4)).strip(QUOTE_EDGE_PUNCTUATION)
        if len(quote.split()) >= (MIN_WORDS_DOUBLE_QUOTE if double else MIN_WORDS_SINGLE_QUOTE):
            quotes.append(quote)
    return quotes

def appears_in_order(parts: list[str], article: str) -> bool:
    """Checks that each part of a quote appears in the article, in order.

    Args:
        parts: The normalized pieces of a quote, split where the writer changed it.
        article: The normalized article text.

    Returns:
        True if every part is found after the one before it.
    """
    position = 0
    for part in parts:
        found = article.find(part, position)
        if found == -1:
            return False
        position = found + len(part)
    return True

def check_quote(claim: str, article_text: str) -> list[str]:
    """Finds the quotes in a claim that don't appear word for word in its article.

    Quotes in double quotes count from two words, and quotes in single quotes from three,
    because shorter ones are usually labels or scare quotes. An ellipsis or square brackets
    inside a quote mark where the writer changed it.

    Args:
        claim: One claim from the briefing.
        article_text: The text of the article the claim cites, as the model read it.

    Returns:
        The quotes that weren't found.
    """
    article = normalize_quote_text(article_text)
    missing = []
    for quote in quotes_in(claim):
        parts = [normalize_quote_text(part).strip(QUOTE_EDGE_PUNCTUATION) for part in QUOTE_EDIT.split(quote)]
        if not appears_in_order([part for part in parts if part], article):
            missing.append(quote)
    return missing

def grounding_numbers(text: str, read_one: bool = True) -> set[str]:
    """Finds the numbers in a text, with numbers written as words turned into digits.

    Args:
        text: A claim or an article.
        read_one: Whether "one" counts as the number 1. In a claim it is usually a pronoun, as
            in "one of the talks", so it is skipped there, but an article's "one" still matches
            a claim's "1".

    Returns:
        Each number as digits, without thousands separators, such as "5000" for "5,000".
    """
    lowered = text.lower()
    numbers = {str(int(NUMBER_WORDS[tens]) + int(NUMBER_WORDS[units])) for tens, units in COMPOUND_NUMBER.findall(lowered)}
    lowered = COMPOUND_NUMBER.sub(" ", lowered)
    numbers.update(match.replace(",", "") for match in GROUNDING_NUMBER.findall(text))
    words = (word for word in re.findall(r"[a-z]+", lowered) if word in NUMBER_WORDS)
    numbers.update(NUMBER_WORDS[word] for word in words if read_one or word != "one")
    return numbers

def spelled_out_initials(text: str) -> set[str]:
    """Finds the abbreviations a text spells out, such as "UN" for "United Nations".

    Args:
        text: An article, with its original capitals.

    Returns:
        The initials of each whole run of two to six capitalised words, ignoring "of" and "the"
        inside the run, so "Ministry of External Affairs" gives "MEA". A capitalised word that
        only starts a sentence, such as "The" in "The United Nations", is dropped from the front.
        Initials from part of a run don't count, so "Ugandan Kenyan Partnership" doesn't give "UK".
    """
    initials = set()
    # "and" usually joins two names, as in "the UN and the Ministry", so it ends a run
    for run in re.findall(r"\b[A-Z][a-z]+(?:\s+(?:(?:of|the|for)\s+)*[A-Z][a-z]+){1,5}", text):
        words = [word for word in run.split() if word[0].isupper()]
        while words and words[0].lower() in GROUNDING_IGNORED_WORDS:
            words = words[1:]
        if len(words) >= 2:
            initials.add("".join(word[0] for word in words))
    return initials

def collapse_abbreviations(text: str) -> str:
    """Writes abbreviations such as "U.S." without full stops, as "US".

    Args:
        text: A claim or an article.

    Returns:
        The text with dotted abbreviations collapsed.
    """
    return DOTTED_ABBREVIATION.sub(lambda match: match.group().replace(".", ""), text)

def check_grounding(claim: str, article_text: str) -> list[str]:
    """Finds the numbers and names in a claim that its article doesn't contain.

    Names are words that start with a capital, matched in any case and with any ending, so
    "Ministers" matches "minister". Numbers match whether written in digits or in words.
    Abbreviations must appear exactly, with or without full stops, or be spelled out.

    Args:
        claim: One claim from the briefing.
        article_text: The text of the article the claim cites.

    Returns:
        The numbers and names missing from the article; empty if all are there.
    """
    text = collapse_abbreviations(DATE_IN_TEXT.sub(" ", URL_IN_TEXT.sub(" ", claim)))
    article_with_case = collapse_abbreviations(article_text)
    article = normalize_quote_text(article_text)
    missing = sorted(grounding_numbers(text, read_one=False) - grounding_numbers(article))
    for word in dict.fromkeys(re.findall(r"\b[A-Z][A-Za-z]+\b", text)):
        # Number words are already compared as numbers
        if word.lower() in GROUNDING_IGNORED_WORDS or word.lower() in NUMBER_WORDS:
            continue
        # Abbreviations such as US and UK are short and case matters ("US" is not "us"), so they
        # must appear exactly. Otherwise swapping US for UK would go unnoticed.
        if word.isupper():
            if not re.search(rf"\b{re.escape(word)}\b", article_with_case) and word not in spelled_out_initials(article_with_case):
                missing.append(word)
            continue
        stem = re.sub(r"(['’]s|s)$", "", word.lower())
        if stem in GROUNDING_IGNORED_WORDS or len(stem) < 3:
            continue
        if not re.search(rf"\b{re.escape(stem)}", article):
            missing.append(word)
    return missing

def run_evals(
    briefing: SubmitBriefing | None,
    fetched: dict[str, str],
    headline_meta: dict[str, str],
    report_path: str | None = None,
    allowed_domains: Sequence[str] = DEFAULT_ALLOWED_DOMAINS,
    now: datetime | None = None,
    suspected_injections: Sequence[str] = (),
    run_info: dict | None = None,
    dropped: Sequence[BriefingItem] = (),
) -> dict:
    """Runs all the checks, prints a summary and writes the report if report_path is given.

    Args:
        briefing: The briefing the model submitted, or None if it submitted none.
        fetched: A dict of {url: article_text} for all articles successfully fetched.
        headline_meta: A dict of {url: published_date} for all headlines fetched.
        report_path: The path where the evaluation report will be saved; None skips writing it.
        allowed_domains: Domains citations may come from; defaults to DEFAULT_ALLOWED_DOMAINS.
        now: Reference time for the freshness check; defaults to the current UTC time.
        suspected_injections: Fetched content the tools flagged or withheld as likely prompt injection.
        run_info: Details of the run to add to the report, such as the model and the request.
        dropped: Key developments removed because they repeat an India item. They aren't shown, but
            are still checked, so a bad item can't pass by being dropped.

    Returns:
        A dict containing evaluation metrics and pass/fail status.

    Raises:
        OSError: If the report file can't be written.
    """
    articles = {normalize_url(url): text for url, text in fetched.items()}
    items = [*briefing.key_developments, *briefing.india] if briefing else []

    cited: list[str] = []
    uncited_items, unverified_quotes, ungrounded = [], [], []
    for item in [*items, *dropped]:
        source = item_source(item)
        if not source:
            uncited_items.append(item.claim)
            continue
        # A link inside the claim is a citation too, and must also be an article that was fetched
        cited.extend([source, *extract_citations(item.claim)])
        article = articles.get(source)
        if article is None:
            continue
        unverified_quotes.extend(f'"{quote}" in: {item.claim[:80]}' for quote in check_quote(item.claim, article))
        missing = check_grounding(item.claim, article)
        if missing:
            ungrounded.append(f"{item.claim[:80]}: not in the cited article: {', '.join(missing)}")

    cited = sorted(set(cited))
    verified = [url for url in cited if url in articles]
    unverified = [url for url in cited if url not in articles]
    stale, unknown_date = check_freshness(cited, headline_meta, now=now)
    # The tools can't fetch an off-list page, so an off-list citation is also unverified. It is
    # checked on its own too, so the rule still holds for a briefing checked outside the agent.
    off_list = [url for url in cited if not is_allowed_domain(url, allowed_domains)]
    # A warning: it catches a general story put in the India section, but a real India story can be
    # worded without any of the terms.
    india_off_topic = [item.claim for item in briefing.india if not INDIA_TERMS.search(item.claim)] if briefing else []
    # Also a warning: a repeat wastes a slot, but each copy still cites what it says
    repeated_items = check_repeated_items(briefing) if briefing else []
    notes_with_news = check_fetch_notes(briefing.fetch_notes) if briefing else []
    # A warning: a note that names no failure, such as which feeds a story came from, isn't a fetch
    # problem, and it can carry story names past check_fetch_notes by mentioning a feed
    notes_without_failure = [note for note in briefing.fetch_notes if not FAILURE_WORDS.search(note)] if briefing else []
    # A warning: one item from five fetched articles has dropped most of the reading. It isn't a
    # failure, because what the briefing does say is still checked.
    too_few_items = briefing is not None and len(items) < min(MIN_ITEMS, len(articles))
    # Not a check: a briefing that mostly reads one site is worth seeing, but isn't wrong
    # Counted per shown item, so three items from one article count three times
    shown_sources = [source for item in items if (source := item_source(item)) in articles]
    sources_by_site = dict(Counter(site_of(source, allowed_domains) for source in shown_sources).most_common())

    # A briefing with nothing verified has proved nothing. Stale and undated stories are warnings:
    # feeds keep important stories for days, so failing on age would fail correct briefings.
    passed = (
        briefing is not None
        and len(verified) > 0
        and not unverified
        and not uncited_items
        and not unverified_quotes
        and not notes_with_news
        and not off_list
    )

    report = {
        **(run_info or {}),
        "briefing": briefing.model_dump() if briefing else None,
        "verified_count": len(verified),
        "unverified": unverified,
        "uncited_items": uncited_items,
        "unverified_quotes": unverified_quotes,
        "notes_with_news": notes_with_news,
        "notes_without_failure": notes_without_failure,
        "off_list": off_list,
        "ungrounded": ungrounded,
        "india_off_topic": india_off_topic,
        "repeated_items": repeated_items,
        "dropped_india_repeats": [item.model_dump() for item in dropped],
        "too_few_items": too_few_items,
        "sources_by_site": sources_by_site,
        "stale": stale,
        "unknown_date": unknown_date,
        "suspected_injections": list(suspected_injections),
        "passed": passed,
    }

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 40 + " EVALUATION REPORT " + "=" * 40)
    print(f"Status: {'[PASSED]' if passed else '[FAILED]'}")
    if briefing is None:
        print("No briefing was submitted.")
    print(f"Verified Citations: {len(verified)}")
    print(f"Cited but not read: {len(unverified)}")
    print(f"Items without a source: {len(uncited_items)}")
    print(f"Quotes not found in the cited article: {len(unverified_quotes)}")
    print(f"Fetch notes that aren't about the run: {len(notes_with_news)}")
    print(f"Fetch notes that name no failure (warning): {len(notes_without_failure)}")
    print(f"Out of Bounds Domains: {len(off_list)}")
    print(f"Items naming things their article doesn't (warning): {len(ungrounded)}")
    print(f"India items that don't mention India (warning): {len(india_off_topic)}")
    print(f"Repeated items (warning): {len(repeated_items)}")
    print(f"Key developments dropped as India repeats (warning): {len(dropped)}")
    print(f"Fewer than {MIN_ITEMS} items, or than the articles fetched (warning): {'yes' if too_few_items else 'no'}")
    sites = ", ".join(f"{site} {count}" for site, count in sources_by_site.items())
    print(f"Items by site: {sites or 'none'}")
    print(f"Stale news warnings (>72h): {len(stale)}")
    print(f"Unknown publish date warnings: {len(unknown_date)}")
    print(f"Suspected prompt injections: {len(suspected_injections)}")
    print("=" * 99 + "\n")
    return report

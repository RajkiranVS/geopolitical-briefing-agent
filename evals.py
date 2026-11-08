"""Runtime evaluation checks for the geopolitics briefing agent."""

import email.utils
import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from urllib.parse import urlsplit

#Global Patterns
# '*', backticks, square brackets, pipes and double or curly quotes are never part of a URL
# (Markdown, table and prose wrappers). '_' and '!' can appear inside URLs, so they are only
# rejected as the final character.
URL_PATTERN = r'https?://[^\s()<>*`|\[\]"“”‘’]+[^\s()<>.,:;\'"\?!*_`|\[\]“”‘’]'

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
# Lines that only label what follows. Markup labels are a Markdown heading, a bold title with an
# optional bracketed note, as in "**Today's briefing** (feeds dated 5 Nov)", or a rule.
MARKUP_LABEL_LINE = re.compile(r"^(#{1,6}\s.*|\*\*[^*\n]+\*\*\s*(\([^()\n]*\))?\s*:?|[-*_=]{3,})$")
# Text labels are an intro with no numbers ("Here is today's briefing.") or a short line in capitals.
# They only count between items, because inside a bullet the same line is a wrapped continuation.
TEXT_LABEL_LINE = re.compile(r"^((?i:here is|here's|below is|this is|this briefing)\b[^\d\n]{0,100}|[A-Z0-9 &'’,/()-]{3,60})$")
# A line of up to 120 characters ending in a colon ("Key developments:") is a label only when content
# follows it, so a claim such as "Iran seized three tankers, Reuters said:" can't pass as one.
# "Source:" is never a label: it belongs to the item before it.
COLON_LABEL_LINE = re.compile(r"^(?!(?i:\(?sources?\b))[^\n]{1,120}:$")
# A paragraph that is only a citation, as in "Source: https://...", belongs to the item before it.
# It must hold nothing but the label and one link, so it can't carry a claim of its own.
SOURCE_LINE = re.compile(r'^\(?\s*sources?\s*:\s*(\[[^\]\n]{1,40}\]\()?https?://\S+\s*$', re.IGNORECASE)
# Fetch notes name a source or a step of the run. Every line of the notes must contain one of
# these words, so a plain news claim there is caught. A claim that also names its source,
# such as "The BBC article says the talks collapsed", still gets through.
FETCH_PROBLEM_WORDS = re.compile(
    r'\b(feeds?|fetch\w*|articles?|headlines?|links?|instructions?|ignored|redirect\w*|briefing|'
    r'summari[sz]\w*|items?|loaded|retrieved|retry|retried|withheld|none)\b',
    re.IGNORECASE,
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
# Capitalised words that aren't facts about the story: labels, the outlets the briefing cites, and
# common words that are only capitalised because they start a sentence.
GROUNDING_IGNORED_WORDS = {
    "source", "bbc", "hindu", "google", "new", "according", "key", "development",
    "the", "this", "that", "these", "those", "its", "it", "he", "she", "they", "we", "his", "her", "their",
    "per", "in", "on", "at", "as", "an", "and", "but", "also", "while", "after", "before", "however",
    "for", "from", "with", "by", "of", "to", "no", "not", "both", "all", "some", "other", "there", "one",
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
ELLIPSIS = re.compile(r'\[(?:\.\.\.|…)\]|\.\.\.|…')
# A notes line that only says nothing went wrong, as in "No issues." It must be the whole line,
# so a claim such as "Iran faced no problems at the border" isn't accepted.
NO_PROBLEMS_LINE = re.compile(r'no\s+(issues?|problems?|errors?)(\s+(to\s+report|found|encountered))?', re.IGNORECASE)

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

    http and https give the same result, because the same article is often linked both ways.

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
    scheme = split_url.scheme.lower()
    if scheme == "http":
        scheme = "https"
    normalized = f"{scheme}://{split_url.netloc.lower()}{split_url.path}"
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
    lines = notes.splitlines()
    for number, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or MARKUP_LABEL_LINE.match(stripped) or TEXT_LABEL_LINE.match(stripped):
            continue
        # As in the body, a line ending in a colon is only a sub-heading when something follows it
        if COLON_LABEL_LINE.match(stripped) and introduces_content(lines[number + 1:]):
            continue
        line_text = LIST_ITEM_START.sub('', stripped)
        if not FETCH_PROBLEM_WORDS.search(line_text) and not NO_PROBLEMS_LINE.fullmatch(line_text.strip(' *_.!')):
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

def introduces_content(following: list[str]) -> bool:
    """Checks whether the next non-blank line is content a label could introduce.

    Args:
        following: The lines after the label.

    Returns:
        True if the next non-blank line is a list item or ordinary text, False if there is none
        or it is another heading, label or the fetch notes marker.
    """
    for line in following:
        stripped = line.strip()
        if not stripped:
            continue
        if LIST_ITEM_START.match(stripped):
            return True
        return not (
            MARKUP_LABEL_LINE.match(stripped)
            or TEXT_LABEL_LINE.match(stripped)
            or COLON_LABEL_LINE.match(stripped)
            or is_notes_marker_line(stripped)
        )
    return False

def split_items(briefing: str, labels: list[str] | None = None) -> list[str]:
    """Splits the briefing into list items and prose paragraphs, each joined onto one line.

    Items are grouped across line continuations. Headings, bold titles and horizontal rules
    aren't items. A final FETCH_NOTES_MARKER section is left out (see strip_fetch_notes).

    Args:
        briefing: The text briefing content.
        labels: If given, the heading and label lines between items are added to it.

    Returns:
        The items and paragraphs, in order.
    """
    items = []
    current_item = []
    
    def save_current():
        """Moves the item being built into the finished items, if there is one."""
        if current_item:
            items.append(" ".join(current_item))
            current_item.clear()

    # A citation paragraph may only join the item just before it, with nothing but blank lines between
    can_join_previous = False
    lines = strip_fetch_notes(briefing).splitlines()
    for number, line in enumerate(lines):
        stripped = line.strip()

        # 1. Check for bullet or numbered list indicators
        if LIST_ITEM_START.match(stripped):
            save_current()
            current_item.append(stripped)
            can_join_previous = False

        # 2. A paragraph break ends the item
        elif not stripped:
            if current_item:
                save_current()
                can_join_previous = True

        # 3. Inside an item, only a Markdown heading or a marker line ends it. Bold text, rules
        # and lines ending in a colon are part of the item.
        elif current_item and not stripped.startswith("#") and not is_notes_marker_line(stripped):
            current_item.append(stripped)

        # 4. Between items, labels and a marker line such as "FETCH NOTES: none" need no citation
        elif (
            MARKUP_LABEL_LINE.match(stripped)
            or TEXT_LABEL_LINE.match(stripped)
            or (COLON_LABEL_LINE.match(stripped) and introduces_content(lines[number + 1:]))
            or is_notes_marker_line(stripped)
        ):
            save_current()
            can_join_previous = False
            if labels is not None and not is_notes_marker_line(stripped):
                labels.append(stripped)

        # 5. A citation in its own paragraph belongs to the item just before it
        elif can_join_previous and SOURCE_LINE.match(stripped):
            items[-1] += " " + stripped
            can_join_previous = False

        # 6. Anything else starts a prose paragraph, which needs a citation too
        else:
            current_item.append(stripped)
            can_join_previous = False

    # Save the final residual item block if one remains open
    save_current()
    return items

def check_uncited_items(briefing: str) -> list[str]:
    """Flags list items and prose paragraphs that have no URL citation and aren't a placeholder.

    Args:
        briefing: The text briefing content to validate.

    Returns:
        A list of joined item strings that lack citations.
    """
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

    for item in split_items(briefing):
        # Check if the combined item context contains a valid URL
        if not re.search(URL_PATTERN, item):
            # The whole item, minus its bullet and Markdown wrappers, must be an allowed phrase
            item_text = LIST_ITEM_START.sub('', item).lower()
            item_text = item_text.strip(' *_"“”.')

            if item_text not in allowed_placeholders and not REWORDED_PLACEHOLDER.fullmatch(item_text):
                flagged_items.append(item)

    return flagged_items

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

def check_quotes(briefing: str, fetched: dict[str, str]) -> list[str]:
    """Flags quotes that don't appear word for word in an article their item cites.

    Quotes in double quotes count from two words, and quotes in single quotes from three,
    because shorter ones are usually labels or scare quotes. An ellipsis inside a quote may
    stand for left-out text. Items that cite no fetched article are skipped, because the
    citation checks already fail them. Headings and label lines cite nothing, so their quotes
    may come from any fetched article.

    Args:
        briefing: The text briefing content.
        fetched: Dict of {url: article_text} for the articles the model read.

    Returns:
        The quotes that weren't found, each with the start of its item.
    """
    articles = {normalize_url(url): normalize_quote_text(text) for url, text in fetched.items()}
    labels: list[str] = []
    items = split_items(briefing, labels)
    unverified = []
    for item, is_label in [(item, False) for item in items] + [(label, True) for label in labels]:
        cited = [articles[url] for url in extract_citations(item) if url in articles]
        if is_label and not cited:
            cited = list(articles.values())
        if not cited:
            continue
        for quote in quotes_in(item):
            parts = [normalize_quote_text(part).strip(QUOTE_EDGE_PUNCTUATION) for part in ELLIPSIS.split(quote)]
            if not any(appears_in_order([part for part in parts if part], article) for article in cited):
                unverified.append(f'"{quote}" in: {item[:80]}')
    return unverified

def quotes_in(text: str) -> list[str]:
    """Finds the quotes long enough to check in one item or line.

    Args:
        text: An item, paragraph or heading from the briefing.

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
        parts: The normalized pieces of a quote, split at any ellipsis.
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

def grounding_numbers(text: str, read_one: bool = True) -> set[str]:
    """Finds the numbers in a text, with numbers written as words turned into digits.

    Args:
        text: An item or an article.
        read_one: Whether "one" counts as the number 1. In an item it is usually a pronoun, as
            in "one of the talks", so it is skipped there, but an article's "one" still matches
            an item's "1".

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
        text: An item or an article.

    Returns:
        The text with dotted abbreviations collapsed.
    """
    return DOTTED_ABBREVIATION.sub(lambda match: match.group().replace(".", ""), text)

def check_grounding(item: str, article_text: str) -> list[str]:
    """Finds the numbers and names in an item that its article doesn't contain.

    Names are words that start with a capital, matched in any case and with any ending, so
    "Ministers" matches "minister". Numbers match whether written in digits or, up to twenty
    and in tens, in words. Links and source names are ignored.

    Args:
        item: One item or paragraph of the briefing.
        article_text: The text of an article the item cites.

    Returns:
        The numbers and names missing from the article; empty if all are there.
    """
    # The list marker, such as "1.", is numbering, not a fact
    text = collapse_abbreviations(URL_IN_TEXT.sub(" ", LIST_ITEM_START.sub("", item.strip())))
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

def check_items_grounded(briefing: str, fetched: dict[str, str]) -> list[str]:
    """Flags items whose numbers or names don't appear in any article they cite.

    This is a warning, not a pass or fail rule: a correct item can name something its
    article spells differently. Items that cite no fetched article are skipped.

    Args:
        briefing: The text briefing content.
        fetched: Dict of {url: article_text} for the articles the model read.

    Returns:
        One entry per flagged item: the start of the item and what is missing.
    """
    articles = {normalize_url(url): text for url, text in fetched.items()}
    flagged = []
    for item in split_items(briefing):
        cited = [articles[url] for url in extract_citations(item) if url in articles]
        if not cited:
            continue
        missing = set.intersection(*(set(check_grounding(item, article)) for article in cited))
        if missing:
            flagged.append(f"{item[:80]}: not in the cited article: {', '.join(sorted(missing))}")
    return flagged

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
    # Only the news counts as citing. A link in the fetch notes describes the run, so it neither
    # proves a briefing reported anything nor fails one for naming an article that didn't download.
    cited_urls = extract_citations(strip_fetch_notes(briefing))
    verified, unverified = check_cited_were_fetched(cited_urls, fetched)
    uncited_items = check_uncited_items(briefing)
    unverified_quotes = check_quotes(briefing, fetched)
    ungrounded = check_items_grounded(briefing, fetched)
    stale, unknown_date = check_freshness(cited_urls, headline_meta, now=now)
    off_list = check_allowed_sources(cited_urls, allowed_domains)

    # Strict validation rule: a briefing with nothing verified has proved nothing
    passed = len(verified) > 0 and not unverified and not uncited_items and not unverified_quotes

    report = {
        "verified_count": len(verified),
        "unverified": unverified,
        "uncited_items": uncited_items,
        "unverified_quotes": unverified_quotes,
        "ungrounded": ungrounded,
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
    print(f"Quotes not found in the cited article: {len(unverified_quotes)}")
    print(f"Items naming things their article doesn't (warning): {len(ungrounded)}")
    print(f"Stale news warnings (>72h): {len(stale)}")
    print(f"Unknown publish date warnings: {len(unknown_date)}")
    print(f"Out of Bounds Domains: {len(off_list)}")
    print(f"Suspected prompt injections: {len(suspected_injections)}")
    print("=" * 99 + "\n")
    
    return report

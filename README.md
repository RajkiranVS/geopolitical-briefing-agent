# Geopolitical News Briefing Agent

A **LangGraph + Claude** agent that writes geopolitical news briefings citing **only articles it actually read**, and proves it with deterministic checks in plain Python, not an LLM.

The agent reads live RSS feeds, fetches the full text of selected articles and writes a neutral, attributed briefing with a dedicated section on developments affecting India. After every run, `evals.py` verifies that each citation points to an article the agent fetched, that every bullet and paragraph is cited, that stories are recent and that every source is on the allowlist. Runs are traced in **LangSmith**, and the evals are covered by a `pytest` suite.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Agent Workflow](#agent-workflow)
  - [Tools](#tools)
  - [News Sources](#news-sources)
- [Evaluation Module](#evaluation-module)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Agent](#running-the-agent)
  - [Running the Tests](#running-the-tests)
  - [Linting and Docstrings](#linting-and-docstrings)
- [CI/CD](#cicd)
- [Using the Evaluations](#using-the-evaluations)
- [Threat Model](#threat-model)
- [How the Evals Evolved](#how-the-evals-evolved)
- [Customisation](#customisation)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Agentic tool use**: the model decides which feeds to read and which articles to open, using two tools: `get_headlines` and `fetch_article`.
- **Grounded output**: the system prompt tells the model to summarise only what the fetched articles say, attribute every claim, and cite the exact article link.
- **Source restrictions enforced in the tools**: `get_headlines` only accepts the exact feed URLs in `ALLOWED_FEEDS`. `fetch_article` refuses any URL whose scheme isn't `http`/`https` or whose host isn't on the allowlist, *before* making a network call. Both tools discard content that was redirected to a host off the allowlist. The same `is_allowed_domain` function is used by the tools and by the eval, so the guardrail and the check can never disagree.
- **Prompt injection defences**: fetched text is treated as hostile, in layers. `fetch_article` only follows links from real feed items, so text inside an article or summary can't send the agent anywhere new. Fetched text is wrapped in `<untrusted_content>` tags with a random id, and the system prompt says text inside them is never an instruction. Text that looks like an attempt to instruct the model stays in the tags with a warning to the model, and is listed under `suspected_injections` in the report, so genuine stories aren't lost. Only encoded payloads are withheld, because they can make the API refuse the whole request. `injection_suite/` tests all this against the live model. Feed text also has HTML and line breaks removed and is capped at 500 characters. See [Known Limitations](#known-limitations).
- **Bounded execution, enforced in code**: `fetch_article` refuses after 5 articles (not just a prompt instruction), and after 15 tool-calling rounds the model must answer with what it has.
- **Failures don't lose the run**: a tool error goes back to the model as `FETCH_FAILED`, and if the agent itself crashes, the run still writes a failed eval report and exits 1.
- **Runtime evals on every run**: `main()` calls `run_evals()` after the agent finishes. Deterministic checks verify that citations were actually fetched, that every bullet and paragraph is cited, that stories are recent, and that sources come from allowed domains. The results are printed and written to a JSON report.
- **Prompt–eval contracts**: fixed strings shared by the prompt and the eval (`FETCH NOTES:` and the empty-section placeholder) keep run notes and "nothing to report" lines from being misread as uncited claims.
- **Unit tested**: a `pytest` suite covers URL normalisation, citation extraction, the uncited-item detection, freshness boundaries and the full eval pipeline.

---

## Project Structure

```
geopolitical-briefing-agent/
├── geopolitical_news_agent.py   # Agent: model, tools, LangGraph loop and entry point
├── evals.py                     # Runtime evaluation checks for the briefing output
├── pytest.ini                   # pytest config (adds project root to path, sets test dir)
├── ruff.toml                    # Lint rules, including Google style docstrings
├── requirements.txt             # Runtime dependencies, with version ranges
├── requirements-dev.txt         # Adds pytest and the pinned ruff for tests and linting
├── .env.example                 # Template for the required credentials
├── .gitignore
├── LICENSE                      # MIT
├── tests/
│   ├── test_evals.py            # Unit tests for evals.py
│   ├── test_agent_main.py       # Unit tests for main() and the tools' security rules
│   └── test_docstrings.py       # Every function documents its Args, Returns and Raises
├── injection_suite/             # Prompt injection attacks against the live model (run by hand)
├── .github/
│   ├── workflows/ci.yml         # CI: pytest + Claude Code project evaluation
│   └── claude/review-prompt.md  # Instructions given to Claude in CI
└── README.md
```

---

## How It Works

### Agent Workflow

The agent is built with the LangGraph **Functional API** (`@entrypoint` and `@task`):

```
            ┌──────────────────────────────┐
  User ───► │  call_llm (system prompt +   │
  prompt    │  conversation so far)        │
            └──────────────┬───────────────┘
                           │
                tool calls requested?
                 ┌─────────┴─────────┐
                yes                  no
                 │                   │
                 ▼                   ▼
     ┌───────────────────────┐   Final briefing
     │ call_tool (each call  │   returned
     │ run as a LangGraph    │
     │ task)                 │
     └───────────┬───────────┘
                 │ results appended to messages
                 └──────► back to call_llm  (max 15 rounds)
```

1. **`call_llm`**: sends the system prompt and message history to `claude-haiku-5-5`, which has the tools bound to it.
2. **`call_tool`**: runs each tool call the model requested.
3. **`agent`** (entrypoint): repeats steps 1 and 2 until the model stops requesting tools, then returns the full message list.
4. **`main()`** clears the run state, asks for the user's request, runs the agent, prints the final briefing and then calls **`run_evals()`** on it, using the `fetched` and `headline_meta` dictionaries the tools filled during the run.

### Tools

| Tool | Purpose | Behaviour |
|------|---------|-----------|
| `get_headlines(feed_url)` | Discover stories | Refuses anything except the exact URLs in `ALLOWED_FEEDS` before fetching, downloads the feed with the same protected client as articles (30 seconds per read and up to 3 attempts, so about 2 minutes if the server stops responding), and discards a feed that redirected off the allowlist. Parses the RSS feed with `feedparser` and returns up to **15** numbered items (title, published date, link, summary). Records each `link → published` date in `headline_meta` (normalised URL as key) for the freshness check. Returns `FETCH_FAILED: <url>` if the feed is empty. |
| `fetch_article(url)` | Read a story | Refuses non-allowlisted URLs, then refuses once **5** articles have been fetched or **10** downloads have failed (`FETCH_LIMIT_REACHED`), and refuses a link already tried twice. It downloads the link exactly as the feed gave it, not the URL the model wrote, so the model can't add a query string that carries data to the site. Otherwise downloads the page with `trafilatura`, with certificate checks always on and its protection against private addresses. It discards a page that redirected off the allowlist, then extracts the article body, truncates it at a `Published -` footer if present, caps it at **4,000 characters**, records exactly that text in `fetched` and returns it with `Source: <url>`. Failed fetches return `FETCH_FAILED: <url>` and don't count toward the limit. |

### News Sources

| Source | Feed URL |
|--------|----------|
| BBC News – World | `https://feeds.bbci.co.uk/news/world/rss.xml` |
| The Hindu – International | `https://www.thehindu.com/news/international/feeder/default.rss` |

### Output Format

The system prompt asks for this format:

1. **3–5 key developments**, each attributed to its source and cited with the exact article link.
2. **A section on developments affecting India.**
3. If a section has nothing to report, it contains exactly `Nothing to report from the fetched articles.`
4. A final line starting with `FETCH NOTES:` lists any feed or fetch problems. Only stories whose article was fetched successfully are reported; headline-only items are left out.

---

## Evaluation Module

[`evals.py`](evals.py) contains deterministic, LLM-free checks for the briefing.

| Function | What it checks | Returns |
|----------|----------------|---------|
| `normalize_url(url)` | Lowercases the scheme and host, and strips the query string, fragment and trailing slash, so that the same article always compares equal. | Normalised URL |
| `extract_citations(briefing)` | Finds every URL in the briefing, including Markdown links and URLs wrapped in brackets or followed by punctuation. | Sorted, de-duplicated list of normalised URLs |
| `check_cited_were_fetched(cited, fetched)` | Detects **hallucinated citations**, meaning URLs that were cited but never fetched. | `(verified, unverified)` |
| `is_allowed_domain(url, allowed_domains)` | Shared allowlist rule used by the tools and the eval: `http`/`https` only, a real hostname, and host equal to an allowed domain or a subdomain of it (so `evilbbc.com` is refused). URLs that are malformed or contain a username, backslash, whitespace or control character are refused too, because HTTP clients can read their host differently (`https://evil.example\@bbc.com` connects to `evil.example`). | `bool` |
| `check_uncited_items(briefing)` | Ignores a final `FETCH NOTES:` section, groups bullet and numbered items and prose paragraphs with their continuation lines, and flags any item or paragraph without a URL. Headings, bold titles and horizontal rules need no citation. The notes are only ignored if the marker starts its own line (in any case) and every line after it names a source or a step of the run (words such as "feed", "article", "fetched", "loaded" or "instructions"), says "none", or is a whole line such as "No issues." or "No problems found." Otherwise the whole briefing is checked. An item is accepted as an empty-section placeholder only if the whole item is an allowed phrase. | List of flagged items |
| `check_quotes(briefing, fetched)` | Checks that every quote appears word for word in an article cited by the same item: two or more words in double quotes, or three or more in single quotes, straight or curly. Quotes in headings and label lines, which cite nothing, may come from any fetched article. Curly and straight quotes, dashes, case, whitespace and punctuation at the ends of a quote are ignored, and an ellipsis may stand for left-out text if the parts keep their order. Shorter quotes, and items that cite no fetched article, are skipped. | List of quotes not found |
| `check_freshness(cited, headline_meta, max_hours=72, now=None)` | Flags stories older than `max_hours`, using RFC 2822 feed dates. | `(stale, unknown_date)` |
| `check_allowed_sources(cited, allowed_domains)` | Flags citations whose domain, or parent domain, is not on the allow-list. | List of off-list URLs |
| `check_grounding(item, article_text)` | Finds the numbers and capitalised names in an item that the article doesn't contain. Case and plural endings are ignored, numbers match in any format or as words, including compounds such as "twenty-five", and sentence-start words and the cited outlets' names are skipped. Abbreviations such as US must appear exactly, written with or without full stops, or be spelled out as a whole name, as "United Nations" spells out UN. A person's name with the same initials, such as "Ursula Kline" for UK, also counts, so a swapped abbreviation can still get through in that case. | List of missing numbers and names |
| `check_items_grounded(briefing, fetched)` | Runs `check_grounding` on each item against the articles it cites. A word counts as missing only if no cited article has it. | One entry per flagged item |
| `run_evals(...)` | Runs the checks above, prints a summary, and optionally writes `eval_report.json`. | Report `dict` |

### Pass/Fail Criteria

A briefing **passes** only if all of these are true:

- at least one citation was verified (`verified_count` is above 0), so an empty, prose-only or refused briefing fails. Only links in the news count: a link in the final `FETCH NOTES:` section neither verifies a briefing nor fails it. `http` and `https` links to the same page match,
- every cited URL was actually fetched (`unverified` is empty),
- every bullet item and prose paragraph carries a citation (`uncited_items` is empty), **and**
- every quote appears word for word in an article its item cites (`unverified_quotes` is empty).

Stale stories, unknown dates, off-list domains and items that name things their article doesn't (`ungrounded`) are reported as **warnings**. They do not make the briefing fail.

When the eval fails, the agent script exits with code 1.

### Sample Report (`eval_report.json`)

```json
{
    "verified_count": 4,
    "unverified": [],
    "uncited_items": [],
    "unverified_quotes": [],
    "ungrounded": [],
    "stale": [],
    "unknown_date": [],
    "off_list": [],
    "suspected_injections": [],
    "passed": true
}
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**. The code uses `X | None` type syntax.
- An **Anthropic API key**.
- *(Optional)* A **LangSmith API key**, for tracing agent runs.

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or, to also run the tests and lint
pip install -r requirements-dev.txt
```

> **Tip:** `requirements.txt` gives each package a minimum, the version the tests were run with, and stops before the next major version. To freeze exact versions for reproducible installs, run `pip freeze > requirements-lock.txt`.

### Configuration

The credentials this project needs are listed in [`.env.example`](.env.example):

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | **Yes** | API key for the Claude model. Create one in the [Anthropic Console](https://console.anthropic.com/settings/keys). |
| `LANGSMITH_TRACING` | No | Set to `true` to send traces to LangSmith. |
| `LANGSMITH_API_KEY` | Only if tracing | API key from [LangSmith](https://smith.langchain.com/) under **Settings → API Keys**. |
| `LANGSMITH_PROJECT` | No | The LangSmith project that traces are logged to. If it is unset, traces go to `default`. |
| `LANGSMITH_ENDPOINT` | No | The LangSmith API URL. The default is the US endpoint; use `https://eu.api.smith.langchain.com` for EU or `https://apac.api.smith.langchain.com` for APAC workspaces (a `403` usually means the wrong region). |

**1. Create your `.env` file** from the template and fill in your keys:

```bash
# macOS / Linux / Git Bash
cp .env.example .env
```

```powershell
# Windows (PowerShell)
Copy-Item .env.example .env
```

> **Keep your keys private.** `.env` holds real secrets. Never commit it; add it to `.gitignore`. Only commit `.env.example`.

**2. Load the variables when you run the agent.** The scripts don't read `.env` themselves, so use one of these options:

- **Option A: use the `dotenv` CLI** (no code changes):

  ```bash
  pip install "python-dotenv[cli]"
  dotenv run -- python geopolitical_news_agent.py
  ```

- **Option B: load the file in code.** Add these lines at the top of `geopolitical_news_agent.py`, then run the script as usual:

  ```python
  from dotenv import load_dotenv
  load_dotenv()
  ```

- **Option C: set the variables in your shell:**

  ```powershell
  # Windows (PowerShell)
  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  $env:LANGSMITH_TRACING = "true"
  $env:LANGSMITH_API_KEY = "lsv2_..."
  $env:LANGSMITH_PROJECT = "geopolitical-news-agent"
  ```

  ```bash
  # macOS / Linux
  export ANTHROPIC_API_KEY="sk-ant-..."
  export LANGSMITH_TRACING="true"
  export LANGSMITH_API_KEY="lsv2_..."
  export LANGSMITH_PROJECT="geopolitical-news-agent"
  ```

When LangSmith tracing is enabled, each run, including every LLM call and tool call, is recorded in your LangSmith project for inspection.

### Running the Agent

```bash
python geopolitical_news_agent.py
```

The script asks for your request (for example, *"Brief me on today's geopolitical news for India"*). The briefing is printed between these markers, followed by the eval report:

```
--- RAW BRIEFING TEXT EXTRACTED ---
...briefing...
-----------------------------------
```

The run logic lives in `main()` behind `if __name__ == "__main__":`, so importing the module (for example from tests or CI) makes no API or network calls.

### Running the Tests

```bash
pytest            # run the full suite
pytest -v         # verbose output
pytest -k fresh   # run only the freshness tests
```

`pytest.ini` adds the project root to `PYTHONPATH` and points pytest at the `tests/` directory. The tests make no network or API calls.

### Linting and Docstrings

```bash
ruff check .     # lint, including docstring style
```

CI fails if `ruff` reports anything. It pins `ruff` to 0.17.3, because the docstring rules in [`ruff.toml`](ruff.toml) are preview rules that can change between releases. `requirements-dev.txt` installs the same version locally, so you get the same results.

Every module, class and function has a Google style docstring, with `Args`, `Returns` and `Raises` sections where they apply. Two checks work together, because neither covers everything:

- `ruff` ([`ruff.toml`](ruff.toml)) checks the style, flags an argument missing from an `Args` section, and flags a missing `Returns` section or an exception missing from `Raises`. It doesn't notice a function with no `Args` section at all, and it doesn't require docstrings on nested functions.
- [`tests/test_docstrings.py`](tests/test_docstrings.py) covers those gaps for every function, including nested ones and test helpers.

The docstrings of `get_headlines` and `fetch_article` are also the tool descriptions the model reads, so changing them can change how the agent behaves.

---

## CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) has two jobs. The deterministic `tests` job is the only gate; the Claude review is advisory.

| Job | What it does | Fails the build when |
|-----|--------------|----------------------|
| `tests` (push to `main`, every push to a PR, manual) | Byte-compiles every module, runs `ruff` and runs `pytest` on Python 3.10 and 3.12. | Compilation fails, `ruff` reports a problem, or any test fails |
| `claude-evaluation` (when a PR opens, reopens or leaves draft, when the `claude-review` label is added, and manual runs) | Runs Claude Code in headless mode (`claude -p`) with the prompt in [`.github/claude/review-prompt.md`](.github/claude/review-prompt.md). An earlier step runs pytest, compileall, ruff and pip-audit without the API key and saves their output. Claude reads the whole repo and that output, and can write throwaway probe tests to confirm suspected bugs. It then scores nine dimensions: correctness, agent design, security, eval quality, tests, code quality, dependencies, docs accuracy and CI. | Never: the verdict is reported as a warning, because an LLM's verdict can vary between runs on the same code |

Each review costs API credit, so it doesn't run on every push to a pull request. To review a pull request again after pushing more commits, remove the `claude-review` label and add it again. Draft pull requests aren't reviewed, so a PR can be opened as a draft while work continues and marked ready for one review at the end.

Claude can use `Read`, `Glob`, `Grep` and `TodoWrite`, can write files only inside `claude-review/`, and has a Bash allow-list for running its probe tests with one exact command, `pip list` and `pip show`, `ls` and `wc`. It has no git commands, because git can write files anywhere through `--output`: the step without the key saves the pull request's diff and commits for Claude to read. Web access is blocked. Claude is told not to run the agent script, because that would make live API and network calls.

Checkout doesn't save the job's token in `.git/config`, so neither the project's code nor Claude can read it from there. The project's tests never run in the step that has the API key. Probe tests do, because Claude writes them during its run, but only through `env -u ANTHROPIC_API_KEY`, which keeps the key out of their environment. That stops accidental exposure, such as a test that prints its environment, not a test written to steal the key, which could still read it from the parent process. The job only runs for branches in this repository, so writing such a test needs write access. The Claude Code CLI and every action are pinned, the actions to a commit.

The report goes to the job summary and is posted as a comment on PRs (the comment is updated on each run). It is also uploaded as the `claude-evaluation` artifact, with `report.md`, `verdict.json` and the raw CLI output, which includes turn count and cost.

**Setup:**

1. In the GitHub repo, go to **Settings → Secrets and variables → Actions** and add the secret `ANTHROPIC_API_KEY`.
2. *(Optional)* To use a different model, add a repository **variable** `CLAUDE_MODEL`. The default is `claude-sonnet-5-5`, capped at 25 turns. The review runs only on PRs and manual dispatch, to keep API cost predictable.

PRs from forks don't receive secrets, so the Claude job is skipped for them.

---

## Using the Evaluations

`run_evals` needs three inputs:

| Argument | Description |
|----------|-------------|
| `briefing` | The final briefing text |
| `fetched` | `{url: article_text}` for every article that was fetched successfully |
| `headline_meta` | `{url: published_date}` for every headline returned by the feeds |

Example:

```python
from evals import run_evals

report = run_evals(
    briefing=briefing_text,
    fetched=fetched_articles,          # {url: text}
    headline_meta=headline_dates,      # {url: "Mon, 02 Nov 2026 10:00:00 GMT"}
    report_path="eval_report.json",    # pass None to skip writing the file
    allowed_domains=("bbc.co.uk", "bbci.co.uk", "bbc.com", "thehindu.com"),
)

if not report["passed"]:
    print("Briefing failed validation:", report)
```

> The agent script already does this: the tools fill the module-level `fetched` and `headline_meta` dictionaries during a run, `main()` clears them at the start of each run, and calls `run_evals()` at the end.

---

## Threat Model

The agent reads text written by strangers, so any article or feed item may try to give it instructions. **The regex is a tripwire; the guarantees come from what the agent can and can't do.**

**Enforced in code, even if the model is completely fooled:**

- It can only read. Neither tool has side effects, and neither can send data anywhere except in the URL of a request.
- `get_headlines` only opens the exact URLs in `ALLOWED_FEEDS`.
- `fetch_article` only opens links from real feed items on allowlisted hosts, at most 5 per run.
- Content redirected off the allowlist is discarded, and certificate checks stay on.
- The eval fails any citation of an article the agent never read.

**Reduces risk, but can be beaten:**

- The `<untrusted_content>` fence and the system prompt's security rules rely on the model's judgement.
- `find_injection_signs` is a blocklist. Paraphrases, other languages and Unicode tricks get past it, as the `hindi_instruction`, `reworded_request` and `zero_width` attacks show. It only flags content for the report and the model, except encoded payloads, which are withheld because they can make the API refuse the whole request. Google News links are skipped by the encoded payload check, because their article ids look like encoded text, so a payload written inside a `news.google.com` URL still gets past it. Links to every other host, including BBC and The Hindu, are checked, but only a run that mixes capitals and lower case the way encoded text does counts, so ordinary long paths aren't withheld. In a test of 2,000 random encoded payloads written as URLs, 13 got past this.

**Not solved:** a false claim can still reach the briefing, attributed to a real article, and the eval will pass it, because the eval checks where a claim came from and not whether it's true. A planted "correction" can't be reliably told apart from a genuine one.

### Measured Results

Attacks on the live model with the network stubbed, before and after the defences:

| Test | Before | After |
|---|---|---|
| Forged feed item fetched | 3/3 | 0/3 (the model still tried in all 3 runs, and the tool refused) |
| Fake correction in the briefing body | 2/3 | 1/13 (the eval passed that run) |
| Encoded payload causes an empty refusal | 3/3 | 0/3 |
| False positives on real feeds | 70/162 items with the first version of the screen | 0/171 items, 0/83 articles |

The full suite on the final code had 20 attacks with 3 runs each: 0 of 60 runs had a forged URL reach the network, a marker in the briefing body, a leaked system prompt or an empty briefing. The fake correction ran 10 more times because its result varies from run to run.

These results cover one model (`claude-haiku-5-5`), a small number of runs and one day's news (5 November 2026; 88 of the 171 linked articles didn't download and weren't scanned). They are evidence, not a guarantee.

### Running the Tests

`injection_suite/` attacks the agent with hostile articles and feed items. The model is real, but the network is stubbed, so no site is contacted. It calls the Anthropic API, so it needs `ANTHROPIC_API_KEY`, costs money and isn't part of CI or `pytest`.

```bash
python injection_suite/run_suite.py          # every attack, 3 runs each
python injection_suite/run_suite.py --reps 1 --only forged_feed_item,zero_width
python injection_suite/scan_real_news.py     # false positives on today's news (no API calls)
```

Each attack carries unique markers, so a briefing that obeys it can be detected. The suite exits with code 1 if a forged URL reaches the network layer, a marker reaches the briefing body, the system prompt leaks, or the briefing comes back empty. Because the fake correction sometimes gets through, an occasional exit code 1 from that attack is a known result. Full results are written to `injection_suite/results/`. Rerun the suite whenever the prompt, the model or the tools change, and add an attack to `attacks.py` for each new technique you learn about.

---

## How the Evals Evolved

The first live runs failed, and each failure was a real finding. Every fix landed with a regression test.

| Run | Eval result | Cause | Fix |
|-----|-------------|-------|-----|
| 1 | **FAIL**: 0 verified, 5 cited-but-not-read, 5 unknown dates | The briefing wrapped URLs in Markdown italics (`*Source: …*`), and the URL regex captured the trailing `*`, so no citation matched. The BBC feed was also refused: its host `feeds.bbci.co.uk` wasn't allowlisted. | Excluded `*` from URL matches; added `bbci.co.uk` |
| 2 | **FAIL**: 2 uncited bullets | Bullets describing fetch problems were read as uncited news claims. | `FETCH NOTES:` marker: the prompt puts run notes in a final section after it, and the eval ignores that section |
| 3 | **FAIL**: 1 uncited bullet | The model added a "the articles don't cover…" bullet. | Fixed empty-section phrase shared by prompt and eval; prompt forbids "what wasn't covered" bullets |
| 4 | **PASS**: 3 verified, 0 issues | | |

## Customisation

| What to change | Where |
|----------------|-------|
| Model | `init_chat_model(model="claude-haiku-5-5")` in `geopolitical_news_agent.py` |
| News feeds | `ALLOWED_FEEDS` in `geopolitical_news_agent.py` (the prompt reads the first three entries) |
| Briefing rules | The `SYSTEMPROMPT` string |
| Headlines per feed | `limit = 15` in `get_headlines` |
| Article length sent to the model | `article_text[:4000]` in `fetch_article` |
| Maximum agent rounds | `MAX_ROUNDS` in `geopolitical_news_agent.py` |
| User request | Typed at the prompt when `main()` runs |
| Allowed citation domains | `DEFAULT_ALLOWED_DOMAINS` in `evals.py` |
| Freshness window | The `max_hours` argument of `check_freshness` (default 72) |
| Accepted empty-section phrases | `allowed_placeholders` in `check_uncited_items`, and `REWORDED_PLACEHOLDER` in `evals.py` for rewordings |

---

## Known Limitations

- **Citations are checked for provenance, not accuracy.** The evals prove that every cited URL points to an article the agent actually fetched. They do not prove that the claim next to the citation is supported by that article. A bullet can cite a real, fetched article and still misstate what it says. Only quotes are checked against the article: `check_quotes` fails a quote that was changed or made up. For everything else, `ungrounded` warns when an item names a place, person or number its article doesn't contain, but a claim built from words the article does contain, such as the wrong country doing the right thing, passes. In 8 live runs on 8 November 2026 it flagged 5 of 24 cited items. All 5 were real details from outside the article, such as "Gulf of Oman" where the article said "the Gulf", and abbreviations such as CENTCOM.
- **Headings need no citation.** Prose paragraphs must cite a URL, like bullets, but a claim written as a heading, a bold title, a short line in capitals, a line of up to 120 characters ending in a colon with an item or text after it, or an intro sentence starting "Here is" with no numbers (`## Iran seized a tanker`) still passes. So a claim such as "Iran has seized three tankers, Reuters said:" gets through when a bullet follows it, though it is flagged on its own. A paragraph that holds only `Source:` and one link counts as the citation of the item before it. A closing line after the `FETCH NOTES:` section, such as "Let me know if you want more detail", makes the notes check fail, because the prompt says the notes must come last.
- **The `FETCH NOTES:` check is a word list.** Every line of the notes must name a source or a step of the run, so a plain news claim there fails the briefing. A claim that also names its source, such as "The BBC article says the talks collapsed", still passes. In the other direction, a real fetch problem worded without any of the listed words makes the briefing fail, which errs on the safe side. Replayed on 98 saved real briefings (8 live runs and 90 attack suite runs), this check and the prose check changed no result.
- **India-focused, with no India-specific feed.** The prompt asks for a section on India, which draws on India-related stories in the BBC World and Hindu International feeds. A Google News search feed for India's foreign affairs was dropped on 8 November 2026: its links open the publisher's page through a script, so none of 15 tested gave the agent any article text, and each attempt counted as a failed fetch. Two feeds on allowed sites were measured as replacements the same day. BBC's India section had readable articles, but only 4 of its first 10 were under 72 hours old and most were features. The Hindu's national feed was fresh but mostly domestic news.
- **Fetch limits are per run, not per site.** The 5-article cap counts successful downloads, and a lock stops parallel calls from passing it. Each link may be downloaded twice, as the prompt's one retry allows, an article already read is sent again without a new download, a second call for a link that is still downloading is refused, and the run stops fetching after 10 failed downloads, counting downloads still running. In 8 live runs on 8 November 2026 at most 2 downloads failed.
- **The quote check rests on little data.** In 8 live runs on 8 November 2026 the model quoted only 3 times, and all 3 passed once punctuation inside the quotation marks was ignored. Quotes are compared with the text the model read, which is the article cut to 4,000 characters, so a real quote from later in the article fails. A quote in a heading cites nothing, so it passes if any fetched article contains it, even an unrelated one.
- **`check_grounding` is a heuristic.** It matches surface tokens, so a name the article spells differently, or a number given as a percentage in one and a fraction in the other, is flagged. That is why it only warns.
- **Prompt injection can be reduced, not eliminated.** The tool restrictions hold even if the model is fooled, but only the model's judgement stops a false claim from reaching the briefing. A planted "correction" in a real article can't be told apart from a genuine one. The text screen is easy to reword around, and it also flags ordinary news that uses similar wording, so a `suspected_injections` entry is a prompt to look, not proof of an attack. If the API refuses because of fetched content, the run reports that and exits 1 with no briefing.
- **Redirects are checked after they happen.** The tools discard content from a redirect off the allowlist, but the request to that host has already been made. `trafilatura` blocks private addresses before connecting, for both tools.
- **The download timeout isn't a total limit.** It applies to each read, so a stalled server fails after about 2 minutes, but one that keeps sending a little at a time isn't cut off until the body reaches trafilatura's size limit. The tools only download from allowlisted hosts, which limits who could do this.
- **A feed can come back cut off with status 200.** This was seen with the Google News feed, which has since been dropped, and the cause isn't known. `get_headlines` still downloads a feed again if it can't be parsed. Articles have no such retry: in one test on 6 November 2026, none of 384 parallel downloads from BBC and The Hindu came back cut off.
- **Freshness depends on feed metadata.** Stories without a parseable `Published` date are reported as `unknown_date`, not as stale.
- **Dependencies have ranges, not exact pins.** A new minor release inside a range can still change behaviour between installs. There is no lock file.
- **CI still tests Python 3.10**, which reached end of life in October 2026, because it is the oldest version the code supports.

---

## Roadmap

- **Decide whether grounding should fail a run.** `ungrounded` is a warning. Over more live runs, count how often it flags a detail that is wrong rather than merely outside the article. If the model often adds outside details, a prompt rule against them may help more than failing the run.

---

## License

[MIT](LICENSE) © 2026 Rajkiran Veldur

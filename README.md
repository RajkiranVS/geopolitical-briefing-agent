# Geopolitical News Briefing Agent

A **LangGraph + Claude** agent that writes geopolitical news briefings citing **only articles it actually read**, and proves it with deterministic checks in plain Python, not an LLM.

The agent reads live RSS feeds, fetches the full text of selected articles and writes a neutral, attributed briefing with a dedicated section on developments affecting India. After every run, `evals.py` verifies that each citation points to an article the agent fetched, that every bullet is cited, that stories are recent and that every source is on the allowlist. Runs are traced in **LangSmith**, and the evals are covered by a `pytest` suite.

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
- [CI/CD](#cicd)
- [Using the Evaluations](#using-the-evaluations)
- [How the Evals Evolved](#how-the-evals-evolved)
- [Customisation](#customisation)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Agentic tool use**: the model decides which feeds to read and which articles to open, using two tools: `get_headlines` and `fetch_article`.
- **Grounded output**: the system prompt tells the model to summarise only what the fetched articles say, attribute every claim, and cite the exact article link.
- **Domain allowlist enforced in the tools**: both tools refuse any URL whose scheme isn't `http`/`https` or whose host isn't on the allowlist, *before* making a network call. The same `is_allowed_domain` function is used by the tools and by the eval, so the guardrail and the check can never disagree.
- **Bounded execution, enforced in code**: `fetch_article` refuses after 5 articles (not just a prompt instruction), and the agent stops after 15 tool-calling rounds.
- **Runtime evals on every run**: `main()` calls `run_evals()` after the agent finishes. Deterministic checks verify that citations were actually fetched, that every bullet is cited, that stories are recent, and that sources come from allowed domains. The results are printed and written to a JSON report.
- **Prompt–eval contracts**: fixed strings shared by the prompt and the eval (`FETCH NOTES:` and the empty-section placeholder) keep run notes and "nothing to report" lines from being misread as uncited claims.
- **Unit tested**: a `pytest` suite covers URL normalisation, citation extraction, the uncited-item detection, freshness boundaries and the full eval pipeline.

---

## Project Structure

```
geopolitical-briefing-agent/
├── geopolitical_news_agent.py   # Agent: model, tools, LangGraph loop and entry point
├── evals.py                     # Runtime evaluation checks for the briefing output
├── pytest.ini                   # pytest config (adds project root to path, sets test dir)
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for the required credentials
├── .gitignore
├── LICENSE                      # MIT
├── tests/
│   └── test_evals.py            # Unit tests for evals.py
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
| `get_headlines(feed_url)` | Discover stories | Refuses non-allowlisted URLs before fetching. Parses the RSS feed with `feedparser` and returns up to **15** numbered items (title, published date, link, summary). Records each `link → published` date in `headline_meta` (normalised URL as key) for the freshness check. Returns `FETCH_FAILED: <url>` if the feed is empty. |
| `fetch_article(url)` | Read a story | Refuses non-allowlisted URLs, then refuses once **5** articles have been fetched (`FETCH_LIMIT_REACHED`). Otherwise downloads and extracts the article body with `trafilatura`, truncates it at a `Published -` footer if present, caps it at **4,000 characters**, records exactly that text in `fetched` and returns it with `Source: <url>`. Failed fetches return `FETCH_FAILED: <url>` and don't count toward the limit. |

### News Sources

| Source | Feed URL |
|--------|----------|
| BBC News – World | `http://feeds.bbci.co.uk/news/world/rss.xml` |
| The Hindu – International | `https://www.thehindu.com/news/international/feeder/default.rss` |
| Google News – India foreign affairs | `https://news.google.com/rss/search?q=India+foreign+affairs&hl=en-IN&gl=IN&ceid=IN:en` |

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
| `is_allowed_domain(url, allowed_domains)` | Shared allowlist rule used by the tools and the eval: `http`/`https` only, a real hostname, and host equal to an allowed domain or a subdomain of it (so `evilbbc.com` and `bbc.com@evil.com` are refused). | `bool` |
| `check_uncited_items(briefing)` | Ignores everything after `FETCH NOTES:`, groups bullet and numbered items with their continuation lines, and flags any item without a URL. Short empty-section placeholders are allowed. | List of flagged items |
| `check_freshness(cited, headline_meta, max_hours=72, now=None)` | Flags stories older than `max_hours`, using RFC 2822 feed dates. | `(stale, unknown_date)` |
| `check_allowed_sources(cited, allowed_domains)` | Flags citations whose domain, or parent domain, is not on the allow-list. | List of off-list URLs |
| `check_grounding(item, article_text)` | A lightweight hallucination heuristic: numbers and capitalised words in an item must appear in the source article. | List of missing entities |
| `run_evals(...)` | Runs the checks above (except `check_grounding`), prints a summary, and optionally writes `eval_report.json`. | Report `dict` |

### Pass/Fail Criteria

A briefing **passes** only if both of these are true:

- every cited URL was actually fetched (`unverified` is empty), **and**
- every bullet item carries a citation (`uncited_items` is empty).

Stale stories, unknown dates and off-list domains are reported as **warnings**. They do not make the briefing fail.

### Sample Report (`eval_report.json`)

```json
{
    "verified_count": 4,
    "unverified": [],
    "uncited_items": [],
    "stale": [],
    "unknown_date": [],
    "off_list": [],
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
```

> **Tip:** `requirements.txt` is unpinned. To freeze exact versions for reproducible installs, run `pip freeze > requirements.txt`.

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

---

## CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) has two jobs. The deterministic `tests` job is the only gate; the Claude review is advisory.

| Job | What it does | Fails the build when |
|-----|--------------|----------------------|
| `tests` (push to `main`, PRs, manual) | Byte-compiles every module, runs `ruff` (informational only) and runs `pytest` on Python 3.10 and 3.12. | Compilation or any test fails |
| `claude-evaluation` (PRs and manual only) | Runs Claude Code in headless mode (`claude -p`) with the prompt in [`.github/claude/review-prompt.md`](.github/claude/review-prompt.md). Claude reads the whole repo, runs pytest, compileall, ruff and pip-audit, and can write throwaway probe tests to confirm suspected bugs. It then scores nine dimensions: correctness, agent design, security, eval quality, tests, code quality, dependencies, docs accuracy and CI. Never: the verdict is reported as a warning, because an LLM's verdict can vary between runs on the same code |

Claude can use these tools: `Read`, `Glob`, `Grep`, `Write` and `TodoWrite`, plus a Bash allow-list for pytest, compileall, ruff, pip-audit, pip, read-only git commands, `ls` and `wc`. `Edit` and web access are blocked. Claude is told not to run the agent script, because that would make live API and network calls.

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
    allowed_domains=("bbc.co.uk", "bbci.co.uk", "bbc.com", "thehindu.com", "google.com"),
)

if not report["passed"]:
    print("Briefing failed validation:", report)
```

> The agent script already does this: the tools fill the module-level `fetched` and `headline_meta` dictionaries during a run, `main()` clears them at the start of each run, and calls `run_evals()` at the end.

---

## How the Evals Evolved

The first live runs failed, and each failure was a real finding. Every fix landed with a regression test.

| Run | Eval result | Cause | Fix |
|-----|-------------|-------|-----|
| 1 | **FAIL**: 0 verified, 5 cited-but-not-read, 5 unknown dates | The briefing wrapped URLs in Markdown italics (`*Source: …*`), and the URL regex captured the trailing `*`, so no citation matched. The BBC feed was also refused: its host `feeds.bbci.co.uk` wasn't allowlisted. | Excluded `*` from URL matches; added `bbci.co.uk` |
| 2 | **FAIL**: 2 uncited bullets | Bullets describing fetch problems were read as uncited news claims. | `FETCH NOTES:` marker: the prompt puts run notes after it, and the eval ignores everything after it |
| 3 | **FAIL**: 1 uncited bullet | The model added a "the articles don't cover…" bullet. | Fixed empty-section phrase shared by prompt and eval; prompt forbids "what wasn't covered" bullets |
| 4 | **PASS**: 3 verified, 0 issues | | |

## Customisation

| What to change | Where |
|----------------|-------|
| Model | `init_chat_model(model="claude-haiku-5-5")` in `geopolitical_news_agent.py` |
| News feeds and briefing rules | The `SYSTEMPROMPT` string |
| Headlines per feed | `limit = 15` in `get_headlines` |
| Article length sent to the model | `article_text[:4000]` in `fetch_article` |
| Maximum agent rounds | `if rounds > 15` in `agent` |
| User request | Typed at the prompt when `main()` runs |
| Allowed citation domains | `DEFAULT_ALLOWED_DOMAINS` in `evals.py` |
| Freshness window | The `max_hours` argument of `check_freshness` (default 72) |
| Accepted empty-section phrases | `allowed_placeholders` in `check_uncited_items` |

---

## Known Limitations

- **Citations are checked for provenance, not accuracy.** The evals prove that every cited URL points to an article the agent actually fetched. They do not prove that the claim next to the citation is supported by that article. A bullet can cite a real, fetched article and still misstate what it says, and no check in `run_evals` would catch it.
- **India-focused.** The prompt and the Google News feed are set up for India; briefings for other countries would need the prompt and feed query changed.
- **The 5-article cap has a race.** Tool calls from a single model turn run in parallel, so if the model requests six fetches at once, all six can pass the `len(fetched) >= 5` check before any is recorded. A lock around check-and-reserve would close it.
- **`check_grounding` is a heuristic.** It matches surface tokens (numbers and capitalised words), so paraphrases or differences in number formatting can produce false positives. It is also not called by `run_evals`.
- **Google News links are redirects.** Google News RSS items point to `news.google.com` URLs rather than to the publisher, which is why `google.com` is on the default allow-list. `trafilatura` usually can't follow these redirects, so `fetch_article` typically returns `FETCH_FAILED` for Google News items. Because the agent only reports stories it fetched, the Google News feed rarely contributes to the briefing.
- **Freshness depends on feed metadata.** Stories without a parseable `Published` date are reported as `unknown_date`, not as stale.
- **Dependencies are not pinned.** `requirements.txt` lists package names only.

---

## Roadmap

- **Verbatim quote check.** Add a deterministic check to `evals.py` that finds every quoted string in the briefing (text between `"…"` or `“…”`) and confirms it appears word for word in the fetched text of the article cited by the same bullet. This is plain Python with no LLM call, so it is cheap to run on every briefing. It catches invented or altered quotes, which is the most damaging form of the provenance-vs-accuracy gap described in [Known Limitations](#known-limitations). It already has a real target: the quote from Bedi's speech has changed wording between runs, so at least one version cannot be verbatim. Implementation notes:
  - Normalise both sides before comparing: curly to straight quotes, collapsed whitespace, unified dashes and apostrophes.
  - Compare against the full fetched text, not only the 4,000-character excerpt sent to the model, and report which case applied.
  - Treat a quote that is missing from its cited article as a **failure**, not a warning, and add it to the report as `unverified_quotes`.

---

## License

[MIT](LICENSE) © 2026 Rajkiran Veldur

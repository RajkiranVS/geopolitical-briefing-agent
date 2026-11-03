import re
import feedparser
from trafilatura import fetch_url, extract
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from collections.abc import Sequence
from typing import cast
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task

from evals import is_allowed_domain, normalize_url, run_evals, DEFAULT_ALLOWED_DOMAINS, FETCH_NOTES_MARKER, EMPTY_SECTION_PLACEHOLDER

model = init_chat_model(model="claude-haiku-5-5")

SYSTEMPROMPT = f"""
                    You are an expert geopolitical news agent.
                    Your task is to analyze the provided news headlines and articles, and summarise only what the fetched articles say,
                    attribute claims to the source, use neutral language, no speculation.

                    First use the get_headlines tool to fetch the latest headlines from a news feed url below
                    
                    URLs:
                    - http://feeds.bbci.co.uk/news/world/rss.xml
                    - https://www.thehindu.com/news/international/feeder/default.rss
                    - https://news.google.com/rss/search?q=India+foreign+affairs&hl=en-IN&gl=IN&ceid=IN:en

                    STRICTLY use only the provided urls to fetch headlines. Do not use any other sources.

                    Then, for any headline that seems relevant or interesting, use the fetch_article tool to retrieve
                    the full text of the article. 
                    
                    Finally, summarise only what the fetched articles say, attribute claims to the source, use neutral language, no speculation.
                    
                    Always cite the source of the information you provide. If you encounter any issues fetching headlines or articles, report them clearly.
                    Fetch at most 5 articles. Cite only articles you have fetched, using the exact link.
                    Only report stories you fetched successfully. Do not include headline-only items, and do not
                    mention stories whose article you could not fetch.
                    Output: 3–5 key developments, then a section on developments affecting India.
                    If a section has nothing to report, write exactly: "{EMPTY_SECTION_PLACEHOLDER}"
                    Don't add bullets about what the articles don't cover. Don't repeat an item across sections.
                    If a fetch fails, skip that story. Don’t retry it more than once.
                    When reporting feed or fetch issues, name the source (e.g. "BBC feed") without pasting its URL.
                    End with a line {FETCH_NOTES_MARKER} followed by any feed or fetch problems (or "none").
                    Put nothing about feed or fetch problems before that line.
                    """

headline_meta: dict[str, str] = {}
fetched: dict[str, str] = {}

#Define Tools
@tool
def get_headlines(feed_url: str) -> str:
    """
    Fetch the latest headlines from a news feed.
    Returns numbered items with title, published date, link and summary.
    Use this to discover stories before reading them with fetch_article.
    
    Args:
        feed_url: The URL of the news feed to fetch headlines from.
    Returns:
        '''
        [1]
        Title: Headline 1
        Published: Date 1
        Link: URL 1
        Summary: Summary 1
        [2]
        Title: Headline 2
        Published: Date 2
        Link: URL 2
        Summary: Summary 2
        ...
        '''
    """
    if not is_allowed_domain(feed_url, DEFAULT_ALLOWED_DOMAINS):
        return f"FETCH_FAILED: {feed_url} is not an allowed domain."
    
    feed = feedparser.parse(feed_url)
    headlines = []
    limit = 15

    if not feed.entries:
        return f"FETCH_FAILED: {feed_url}"
    
    for index, entry in enumerate(feed.entries[:limit], start=1):
        headlines.append(
            f"[{index}]\n"
            f"Title: {entry.get('title', 'No title')}\n"
            f"Published: {entry.get('published', 'Unknown')}\n"
            f"Link: {entry.get('link', 'No link')}\n"
            f"Summary: {entry.get('summary', 'No summary')}"
        )
        link = entry.get('link')
        published = entry.get('published')
        if link and published:
            headline_meta[normalize_url(link)] = published

    return '\n'.join(headlines)

@tool
def fetch_article(url: str) -> str:
    """
    Fetch the full text of an article from a given URL.
    
    Args:
        url: The URL of the article to fetch.
    Returns:
        The full text of the article along with the cited source URL.
    """
    if not is_allowed_domain(url, DEFAULT_ALLOWED_DOMAINS):
        return f"FETCH_FAILED: {url} is not an allowed domain."
    
    if len(fetched) >= 5:
        return "FETCH_LIMIT_REACHED: 5 articles already fetched. Write the briefing now using those."
    
    downloaded = fetch_url(url)
    if not downloaded:
        return f"FETCH_FAILED: {url}"
    
    article_text = extract(downloaded)
    if not article_text:
        return f"FETCH_FAILED: {url}"

    marker = article_text.find("Published -")
    if marker!= -1:
        article_text = article_text[:marker]
    
    article_text = article_text[:4000]

    fetched[normalize_url(url)] = article_text

    return f"{article_text[:4000]}\n\nSource: {url}"

tools = [get_headlines, fetch_article]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

from langgraph.graph import add_messages
from langchain.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    ToolCall,
)


@task
def call_llm(messages: Sequence[BaseMessage]) -> AIMessage:
    """
    Call the language model with the provided messages and return the model's response.
    
    Args:
        messages: A sequence of BaseMessage objects to send to the model.
    Returns:
        The model's response as a BaseMessage object.
    """
    return model_with_tools.invoke(
            [
                SystemMessage(
                    content=SYSTEMPROMPT
                ),
                *messages,
            ]
        )

@task
def call_tool(tool_call: ToolCall):
    """Performs the tool call"""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

#Define Agent
@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()

    rounds = 0
    while model_response.tool_calls:
        rounds += 1
        if rounds > 15:
            raise RuntimeError("Too many tool calls. Stopping to prevent infinite loop.")
        
        tool_result_futures = [call_tool(tool_call) for tool_call in model_response.tool_calls]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = cast(
            list[BaseMessage],
            add_messages(list(messages), [model_response, *tool_results]),
        )
        model_response = call_llm(messages).result()
    return cast(list[BaseMessage], add_messages(list(messages), model_response))


def main():
    '''
    Main function to execute the geopolitical news agent.'''
    fetched.clear()
    headline_meta.clear()
    #initialize the agent with an initial human message
    initial_messages = [HumanMessage(content=input("Enter your request for the geopolitical news briefing: "))]
    final_state = agent.invoke(initial_messages)
    last_message = cast(Sequence[BaseMessage], final_state)[-1]
    raw_content = ""
    if hasattr(last_message, "content"):   
        content_payload = last_message.content
        if isinstance(content_payload, list):
            text_parts = []
            for block in content_payload:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            raw_content = "".join(text_parts)
        else:
            raw_content = str(content_payload)
    briefing_text = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    print("--- RAW BRIEFING TEXT EXTRACTED ---")
    print(briefing_text)
    print("-----------------------------------")
    run_evals(briefing_text, fetched, headline_meta)

if __name__ == "__main__":
    main()
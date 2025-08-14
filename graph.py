from schemas import SearchQueryList, Reflection
from state import OverallState, QueryGenerationState, ReflectionState, WebSearchState
from prompts import get_current_date
from prompts import generate_query_prompt, summarize_prompt, reflection_prompt
from configuration import Configuration

from langgraph.graph import StateGraph, START, END
from langchain_deepseek import ChatDeepSeek
from langgraph.types import Send

from utils import get_research_topic

from tavily import TavilyClient

from dotenv import load_dotenv
import os

load_dotenv()


# Nodes
def generate_query(state: OverallState) -> QueryGenerationState:
    """Langraph node that generate search queries based on user's question.

    Args:
        state (OverallState): Current graph state containing user's question

    Returns:
        QueryGenerationState: Dictionary with state updates, including search_query key containing generated queries
    """
    configurable = Configuration()

    # Init the llm
    llm = ChatDeepSeek(
        model=configurable.query_generator_model,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=1.0,
        max_retries=2,
        timeout=None,
        max_tokens=None,
    )

    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    formatted_prompt = generate_query_prompt.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        number_queries=configurable.initial_search_query_count,
    )

    # Generate quries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """Langgraph node that send search quries to the web search node."""
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState) -> OverallState:
    """Langgraph node that search one query using Tavily API and summarize the searching result.

    Args:
        state (WebSearchState): Current graph state containing the search query.

    Returns:
        OverallState: Dictionary with updates, including web_search_result, sources_gathered
    """
    configurable = Configuration()

    # Init the Tavily client
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Web search based on the query
    query = state["search_query"]
    response = tavily_client.search(
        query=query,
        topic="general",
        max_results=configurable.max_search_results,
        include_raw_content=True,
    )
    web_search_result = [result["raw_content"] for result in response["results"]]
    sources_gathered = [result["url"] for result in response["results"]]

    llm = ChatDeepSeek(
        model=configurable.query_generator_model,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.3,
        max_retries=2,
        timeout=None,
        max_tokens=None,
    )

    concat_result = "\n\n".join(web_search_result)

    formatted_prompt = summarize_prompt.format(
        current_date=get_current_date(),
        web_search_result=concat_result,
        research_topic=query,
    )

    response = llm.invoke(formatted_prompt)

    return {
        "web_search_result": [response.content],
        "sources_gathered": sources_gathered,
    }


def reflection(state: OverallState) -> ReflectionState:
    configurable = Configuration()

    formatted_prompt = reflection_prompt.format(
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_search_result"]),
    )

    llm = ChatDeepSeek(
        model=configurable.query_generator_model,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.8,
        max_retries=2,
        timeout=None,
        max_tokens=None,
    )

    structured_llm = llm.with_structured_output(Reflection)

    response = structured_llm.invoke(formatted_prompt)

    return {
        "is_sufficient": response.is_sufficient,
        "follow_up_queries": response.follow_up_queries,
        "research_loop_count": state.get("research_loop_count", 0) + 1,
    }


builder = StateGraph(OverallState)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_edge("reflection", END)

graph = builder.compile(name="deep_research")

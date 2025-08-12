from schemas import SearchQueryList
from state import OverallState, QueryGenerationState, WebSearchState
from prompts import get_current_date
from prompts import generate_query_prompt
from configuration import Configuration

from langgraph.graph import StateGraph, START, END
from langchain_deepseek import ChatDeepSeek
from langgraph.types import Send

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
        research_topic=state["messages"],
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


builder = StateGraph(OverallState)

builder.add_node("generate_query", generate_query)
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", END)

graph = builder.compile(name="deep_research")

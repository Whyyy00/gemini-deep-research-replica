from typing import Annotated, TypedDict, Sequence, List
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import operator


class OverallState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    search_query: Annotated[List, operator.add]
    web_search_result: Annotated[List, operator.add]
    sources_gathered: Annotated[List, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: List[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: str

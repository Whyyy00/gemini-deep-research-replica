from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """The Configuration of the deep research agent"""

    query_generator_model: str = Field(default="deepseek-chat")

    summarize_model: str = Field(default="deepseek-chat")

    reflection_model: str = Field(default="deepseek-chat")

    answer_model: str = Field(default="deepseek-reasoner")

    initial_search_query_count: int = Field(default=3)

    max_research_loops: int = Field(default=2)

    max_search_results: int = Field(default=3)

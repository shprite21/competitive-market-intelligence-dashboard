import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv(Path(__file__).with_name(".env"), override=True)


@tool
def get_news(company: str, api_key: str | None = None):
    """
    Retrieve news related to a company in the past year.

    Input should be a string that contains company full name

    output is a sorted dictionary
    """
    effective_api_key = (api_key or os.getenv("TAVILY_API_KEY", "")).strip()
    if not effective_api_key:
        raise ValueError("A Tavily API key is required to search company news.")

    search_tool = TavilySearch(
        tavily_api_key=effective_api_key,
        topic="news",
        max_results=10,
        time_range="year",
        search_depth="advanced",
        include_answer=True,
    )

    results = search_tool.invoke("Retrieve news related to " + company)

    sorted_results = sorted(results["results"], key=lambda x: x.get("score", 0), reverse=True)

    return sorted_results


"""
for res in sorted_results["results"]:

    print(f"Title : {res["title"]}")
    print(f"URL : {res["url"]}")
    print(f"score : {res["score"]}")
    print(f"date: {res["published_date"]}")
    print("-------------------")
"""

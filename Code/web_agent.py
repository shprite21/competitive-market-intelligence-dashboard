import matplotlib.pyplot as plt
import yfinance as yf
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from news_twitter_api import get_news
from yfinance import Search


# Define the tools required to by the agent
@tool
def get_stock_price(company: str):
    """
    Useful to retrieve the stock price of a company .Input should be a string . Return stock price of the company as float"
    """
    symbol = Search(company).quotes[0]["symbol"]
    stock = yf.Ticker(symbol)
    print(f"company {company} symbol {symbol}")

    stock_price = stock.fast_info["lastPrice"]
    return stock_price


@tool
def get_company_info(company: str):
    """
    #Useful to retrieve information regarding a company.Input should be a  strings. Return personal information regarding the company as a dictionary.
    """
    symbol = Search(company).quotes[0]["symbol"]

    stock = yf.Ticker(symbol)
    information = stock.info
    return information


@tool
def get_historical_prices(company: str, period: str, Graph=False):
    """
    Useful to retrieve the historical stock prices of a company .
    Input should be:
        company: A ticker symbol of the company in the form of a string
        period: The time duration for which the historical data is to be retireved in the form of a string
        Graph: Boolean value that specifies wether a graph of the historical data is to be plotted or not. Default value is False.

    Return historical stock prices of the company as a pandas DataFrame"""
    symbol = Search(company).quotes[0]["symbol"]

    stock = yf.Ticker(symbol)
    df = stock.history(period=period)

    print(f"period {period} Graph :{Graph} symbol {symbol}")

    if Graph:
        df["Close"].plot()
        plt.show()
    return df


Tools = [get_stock_price, get_historical_prices, get_company_info, get_news]

llm = ChatOllama(model="qwen3:8b", temperature=0)

agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant that retrieves the stock price of  a company . "
    " Keep your responses concise. When date is not specified retirieve that latest stock price. Always use given tools to retrieve stock price."
    "Notify the user if you cannot access tools. Never give suggestions.",
    tools=Tools,
    checkpointer=InMemorySaver(),
    # name="Web_Search_Agent"
)

cnfg = {"configurable": {"thread_id": str(uuid7())}}


# invoking the agent and printing results
def main(query):

    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=cnfg)

    for msg in result["messages"]:
        print(type(msg).__name__)
        print(msg.content)
        print("---------------------------------------------")


if __name__ == "__main__":
    main(input("Enter your query: "))

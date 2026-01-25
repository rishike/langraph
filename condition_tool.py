import sqlite3
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
import requests
import random
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: Annotated[str, "The mathematical operation to perform: add, subtract, multiply, divide"]) -> dict:
    """Performs basic arithmetic operations."""
    if operation == "add":
        result = first_num + second_num
    elif operation == "subtract":
        result = first_num - second_num
    elif operation == "multiply":
        result = first_num * second_num
    elif operation == "divide":
        if second_num == 0:
            raise ValueError("Cannot divide by zero.")
        result = first_num / second_num
    else:
        raise ValueError("Invalid operation. Please choose from add, subtract, multiply, divide.")
    
    return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}

@tool
def get_stock_price(ticker: str) -> dict:
    """Fetches the current stock price for a given ticker symbol."""
    api_url = f"https://www.alphavantage.co/query/function=GLOBAL_QUOTE&symbol={ticker}&apikey=*******"
    response = requests.get(api_url)
    data = response.json()
    return data

tools = [calculator, get_stock_price, search_tool]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """LLM node that may answer or request tool usage based on the conversation."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect("chatbot_checkpoint.db", check_same_thread=False)
checkpoint_saver = SqliteSaver(connection=conn, table_name="chatbot_states")


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpoint_saver)
# print(chatbot)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpoint_saver.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
# out = chatbot.invoke({"messages": [HumanMessage(content="What's the weather like in New York City?")]})

# print(out["messages"][-1].content)

# out = chatbot.invoke({"messages": [HumanMessage(content="What's 15 multiplied by 3?")]})
# print(out["messages"][-1].content)

# out = chatbot.invoke({"messages": [HumanMessage(content="What's the current stock price of AAPL?")]})
# print(out["messages"][-1].content)

# out = chatbot.invoke({"messages": [HumanMessage(content="what is the stock price of GOOG and if i buy 10 shares how much will it cost me?")]})

# print(out["messages"][-1].content)

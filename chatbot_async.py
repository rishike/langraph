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
from langgraph.checkpoint.sqlite import SqliteSaver
import asyncio
import requests
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

search_tool = DuckDuckGoSearchRun(region="us-en")

client = MultiServerMCPClient({
    "arith": {
        "transport": "stdio",
        "command": "python3",
        "args": ["main.py"]
    }
})
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
    api_url = f"https://www.alphavantage.co/query/function=GLOBAL_QUOTE&symbol={ticker}&apikey=FJHEJDHXLHI7E7AH"
    response = requests.get(api_url)
    data = response.json()
    return data

# tools = [calculator, get_stock_price, search_tool]

# llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():

    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    

    async def chat_node(state: ChatState):
        """LLM node that may answer or request tool usage based on the conversation."""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
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
    return chatbot

async def main():
    chatbot = await build_graph()
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="find the modulus in singing way of 132354 divided by 23")]})
    print(result['messages'][0].content)

if __name__ == "__main__":
    asyncio.run(main())
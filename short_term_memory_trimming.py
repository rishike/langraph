from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0.9)

MAX_TOKENS = 150


def call_model(state: MessagesState):
    messages = trim_messages(
        state['messages'], strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS
    )
    print(f"Calling model with {count_tokens_approximately(messages)} tokens")
    for message in messages:
        print(f"{message.content}")

    response = model.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "chat-1"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hello, My name is rishi. What is your name?"}]},
    config
)

print(result["messages"][-1].content)

result2  = graph.invoke(
    {"messages": [{"role": "user", "content": "I am learning langgraph"}]},
    config
)

print(result2["messages"][-1].content)

result3  = graph.invoke(
    {"messages": [{"role": "user", "content": "can you explain me short term memory ?"}]},
    config
)

print(result3["messages"][-1].content)

result4 = graph.invoke(
    {"messages": [{"role": "user", "content": ""
    "What is my name ?"}]},
    config
)

print(result4["messages"][-1].content)

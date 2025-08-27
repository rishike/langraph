from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import Field
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o")

checkpointer = MemorySaver()

graph = StateGraph(ChatState)

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages' : [response]}

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# initial_state = {
#     'messages' : [HumanMessage(content="What is the capital of india")]
# }

# final_state = chatbot.invoke(initial_state)

thread_id = 1
while True:
    user_message = input("Type Here ...\n")
    print('User:', user_message)
    if user_message.strip().lower() in ['exit', 'quit', 'bye']:
        break

    config = {'configurable' : {'thread_id' : thread_id}}
    response = chatbot.invoke({'messages' : [HumanMessage(content=user_message)]}, config=config)
    print('AI:', response['messages'][-1].content)

from langgraph.graph import StateGraph, START , END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# creating a state
class LLMState(TypedDict):

    question: str
    answer: str


def llm_qa(state: LLMState):

    # extract the question from state
    question = state['question']

    # form a prompt
    prompt = f'Answer the following question {question}'

    # ask question to llm
    answer = model.invoke(prompt).content

    # update the answer in the state
    state['answer'] = answer
    return state

# creating a graph
graph = StateGraph(LLMState)

# add nodes
graph.add_node('llm_qa', llm_qa)

# add edges
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

# compile
workflow = graph.compile()

# execute 
initial_state = {'question': 'How far is sun from the earth?'}
final_state=workflow.invoke(initial_state)
print(final_state)
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_hindi: str


parent_llm = ChatOpenAI(model="gpt-4-0613")
subgraph_llm = ChatOpenAI(model="gpt-4-0613")

def translate_text(state: ParentState):

    prompt = f"""You are a helpful assistant that translates text to hindi.Keep it natural and clear. Do not add any extra content. 
    Text:
    {state['answer_eng']}
""".strip()
    translate_text = subgraph_llm.invoke(prompt).content
    return {
        "answer_hindi": translate_text
    }

subgraph_builder = StateGraph(ParentState)
subgraph_builder.add_node('translate_text', translate_text)
subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)
subgraph = subgraph_builder.compile()

def generate_answer(state: ParentState):
    prompt = f"""You are a helpful assistant that answers questions in english and hindi.
    Question: {state['question']}
    """
    answer_eng = parent_llm.invoke(prompt).content
    return {
        "answer_eng": answer_eng
    }

parent_builder = StateGraph(ParentState)
parent_builder.add_node('generate_answer', generate_answer)
parent_builder.add_node('translate_answer', subgraph)
parent_builder.add_edge(START, 'generate_answer')
parent_builder.add_edge('generate_answer', 'translate_answer')
parent_builder.add_edge('translate_answer', END)

parent_graph = parent_builder.compile()

result = parent_graph.invoke({
    "question": "What is quantum computing?"
})

print(result)
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class SubState(TypedDict):
    input_text: str
    translated_text: str

subgraph_llm = ChatOpenAI(model="gpt-4-0613")


def translate_text(state: SubState):

    prompt = f"""
You are a helpful assistant that translates text to hindi.Keep it natural and clear. Do not add any extra content. 
    Text:
    {state['input_text']}
    """.strip()

    translate_text = subgraph_llm.invoke(prompt).content
    return {
        "input_text": state["input_text"],
        "translated_text": translate_text
    }
    
subgraph_builder = StateGraph(SubState)
subgraph_builder.add_node('translate_text', translate_text)
subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)
subgraph = subgraph_builder.compile()

class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_hindi: str


parent_llm = ChatOpenAI(model="gpt-4-0613")

def generate_answer(state: ParentState):
    prompt = f"""You are a helpful assistant that answers questions in english and hindi.
    Question: {state['question']}
    """
    answer_eng = parent_llm.invoke(prompt).content
    return {
        "answer_eng": answer_eng
    }

def translate_answer(state: ParentState):
    result = subgraph.invoke({
       "input_text": state["answer_eng"]
    })
    return {
        "answer_hindi": result["translated_text"]
    }

parent_buillder = StateGraph(ParentState)
parent_buillder.add_node('answer', generate_answer)
parent_buillder.add_node('translate', translate_answer)
parent_buillder.add_edge(START, 'answer')
parent_buillder.add_edge('answer', 'translate')
parent_buillder.add_edge('translate', END)
graph = parent_buillder.compile()

# print(graph)

result = graph.invoke({"question": "What is quantum computing?"})
print(result)





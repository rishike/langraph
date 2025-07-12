from langgraph.graph import StateGraph, START , END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class BlogState(TypedDict):

    title: str
    outline: str
    content: str

def create_outline(state: BlogState)->BlogState:
    title = state['title']
    prompt = f'Generate an detailed outline for a blog on the topic - {title}'
    outline = model.invoke(prompt).content
    # update state
    state['outline'] = outline
    return state

def create_blog(state: BlogState)->BlogState:
    title = state['title']
    outline = state['outline']
    prompt = f'Write an detailed blog on the title - {title} using the following outline \n {outline}'

    content = model.invoke(prompt).content
    state['content'] = content
    return state


graph = StateGraph(BlogState)

# nodes
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

# edge
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

workflow = graph.compile()
initial_state = {'title': 'Rise of AI in India'}
final_state = workflow.invoke(initial_state)
print(final_state)
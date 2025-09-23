import streamlit as st
from database_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_history']:
        st.session_state['chat_history'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

CONFIG = {'configurable': {'thread_id' : st.session_state['thread_id']}}

st.sidebar.title('Chatbot')
st.sidebar.button('New Chat', on_click=reset_chat)
st.sidebar.header('My Conversation')

for thread_id in st.session_state['chat_history'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_msg = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_msg.append({'role': role, 'content': msg.content})
        
        st.session_state['message_history'] = temp_msg



for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input('Type here')


if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    
    # response = chatbot.invoke({'messages' : [HumanMessage(content=user_input)]}, config=CONFIG)
    with st.chat_message('assistant'):
        ai_mesage = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages' : [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )
    
    st.session_state['message_history'].append({'role' : 'assistant', 'content': ai_mesage})
    

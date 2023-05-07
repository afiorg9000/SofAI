import os
import streamlit as st
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain import OpenAI


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTVectorStoreIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index


def ask_ai(query):
    index = GPTVectorStoreIndex.load_from_disk('index.json')
    response = index.query(query, response_mode="compact")
    return response.response


# Get OpenAI API key from user
os.environ["OPENAI_API_KEY"] = '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' 

# Load index
construct_index("context_data/data")

st.set_page_config(page_title="Sofia", page_icon=":robot_face:", layout="centered")

conversation_history = []

col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.title("Sofia")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    input_container = st.container()
    chat_container = st.container()

    with input_container:
        user_input = st.text_input('Type your question and press Enter:', on_change=None, key=None)

    if user_input:
        response = ask_ai(user_input)
        st.session_state.conversation_history.append({"type": "bot", "text": response})
        st.session_state.conversation_history.append({"type": "user", "text": user_input})
        user_input = ""

    with chat_container:
        st.markdown(f"""
        <style>
            .chat {{ max-height: 500px; overflow-y: auto; }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="chat">', unsafe_allow_html=True)

        for message in st.session_state.conversation_history[::-1]:
            if message["type"] == "user":
                st.markdown(f'<div style="text-align: right; margin-top: 10px;"><span style="font-size: 1.2em; background-color: #DE20FD; padding: 6px 12px; border-radius: 15px;display: inline-block; max-width: 80%; word-wrap: break-word;">{message["text"]}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: left; margin-top: 30px;"><span style="font-size: 1.2em; background-color: #208EFD; padding: 6px 12px; border-radius: 15px;display: inline-block; max-width: 80%; word-wrap: break-word;">{message["text"]}</span></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
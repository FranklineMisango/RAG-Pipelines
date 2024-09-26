import streamlit as st
import os
from constants import search_number_messages
from langchain_community.chat_models import ChatOpenAI
from langchain_utils import initialize_chat_conversation, extract_text_from_pdf
from search_index import index_pdfs, search_faiss_index
import re

from dotenv import load_dotenv
load_dotenv()

def remove_pdf(pdf_to_remove):
    if pdf_to_remove in st.session_state.pdfs:
        st.session_state.pdfs.remove(pdf_to_remove)

st.set_page_config(page_title='📚')
st.title('Frankline & Co. LP. Self Rostering RAG for Langchain & GPT3.xx LLM')

for key, default_value in {
    'faiss_index': {'indexed_pdfs': [], 'index': None},
    'conversation_memory': None,
    'messages': [],
    'pdfs': []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

temp_files_dir = "temp_files"
if not os.path.exists(temp_files_dir):
    os.makedirs(temp_files_dir)

with st.sidebar:
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    uploaded_files = st.file_uploader("Upload relevant PDFs:", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [os.path.basename(pdf) for pdf in st.session_state.pdfs]:
                file_path_saved = os.path.join(temp_files_dir, uploaded_file.name)
                with open(file_path_saved, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.pdfs.append(file_path_saved)

    with st.container():
        if st.session_state.pdfs:
            st.header('PDFs uploaded:')
            for idx, pdf_path in enumerate(st.session_state.pdfs):
                st.write(os.path.basename(pdf_path))
                st.button(label='Remove', key=f"Remove_{os.path.basename(pdf_path)}_{idx}", on_click=remove_pdf, kwargs={'pdf_to_remove': pdf_path})
                st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query_text := st.chat_input("Your message"):

    os.environ['OPENAI_API_KEY'] = openai_api_key

    st.chat_message("user").markdown(query_text)
    st.session_state.messages.append({"role": "user", "content": query_text})

    session_pdfs = st.session_state.pdfs
    if (st.session_state['faiss_index']['index'] is None or 
        set(st.session_state['faiss_index']['indexed_pdfs']) != set(session_pdfs)):
        
        st.session_state['faiss_index']['indexed_pdfs'] = session_pdfs
        
        with st.spinner('Indexing PDFs...'): 
            st.write(session_pdfs)         
            faiss_index = index_pdfs(session_pdfs)
            st.session_state['faiss_index']['index'] = faiss_index
    else:
        faiss_index = st.session_state['faiss_index']['index']

    if st.session_state['conversation_memory'] is None:
        conversation = initialize_chat_conversation(faiss_index)
        st.session_state['conversation_memory'] = conversation
    else:
        conversation = st.session_state['conversation_memory']

    user_messages_history = [message['content'] for message in st.session_state.messages[-search_number_messages:] if message['role'] == 'user']
    user_messages_history = '\n'.join(user_messages_history)

    # Extract text from the uploaded PDFs
    extracted_texts = []
    for pdf_path in st.session_state.pdfs:
        extracted_text = extract_text_from_pdf(pdf_path)
        extracted_texts.append(extracted_text)

    extracted_texts_combined = "\n".join(extracted_texts)

    # Split the extracted text into smaller chunks
    chunk_size = 500  # Adjust the chunk size as needed
    text_chunks = [extracted_texts_combined[i:i + chunk_size] for i in range(0, len(extracted_texts_combined), chunk_size)]

    responses = []
    with st.spinner('Querying OpenAI GPT...'):
        for chunk in text_chunks:
            prompt = f"Analyze and critique the following document content: {chunk}. User's query: {query_text}. User's message history: {user_messages_history}"
            response = conversation.predict(input=prompt)
            responses.append(response)

    combined_response = " ".join(responses)

    with st.chat_message("assistant"):
        st.markdown(combined_response)
        snippet_memory = conversation.memory.memories[1]
        for page_number, snippet in zip(snippet_memory.pages, snippet_memory.snippets):
            with st.expander(f'Snippet from page {page_number + 1}'):
                snippet = re.sub(r"<START_SNIPPET_PAGE_\d+>", '', snippet)
                snippet = re.sub(r"<END_SNIPPET_PAGE_\d+>", '', snippet)
                st.markdown(snippet)

    st.session_state.messages.append({"role": "assistant", "content": combined_response})
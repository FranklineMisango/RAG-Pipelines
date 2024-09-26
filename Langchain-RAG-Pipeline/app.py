import streamlit as st
import os
from constants import search_number_messages
from langchain_utils import initialize_chat_conversation, extract_text_from_pdf
from search_index import index_uploaded_pdfs, index_uploaded_pdfs_from_paths
import re
from dotenv import load_dotenv
load_dotenv()

def remove_pdf(pdf_to_remove):
    """
    Remove PDFs from the session_state. Triggered by the respective button
    """
    if pdf_to_remove in st.session_state.pdfs:
        st.session_state.pdfs.remove(pdf_to_remove)

# Page title
st.set_page_config(page_title='📚')
st.title('Frankline & Co. LP. Self Rostering RAG for Langchain & GPT3.xx LLM')

# Initialize the faiss_index key in the session state. This can be used to avoid having to download and embed the same PDF
# every time the user asks a question
if 'faiss_index' not in st.session_state:
    st.session_state['faiss_index'] = {
        'indexed_pdfs': [],
        'index': None
    }

# Initialize conversation memory used by Langchain
if 'conversation_memory' not in st.session_state:
    st.session_state['conversation_memory'] = None

# Initialize chat history used by StreamLit (for display purposes)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store the PDFs uploaded by the user in the UI
if 'pdfs' not in st.session_state:
    st.session_state.pdfs = []

# Store the PDF paths uploaded by the user in the UI
if 'pdf_paths' not in st.session_state:
    st.session_state.pdf_paths = []

# Ensure temp_files directory exists or create it
temp_files_dir = "temp_files"
if not os.path.exists(temp_files_dir):
    os.makedirs(temp_files_dir)


with st.sidebar:

    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Add/Remove PDFs form
    uploaded_files = st.file_uploader("Upload relevant PDFs:", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [pdf.name for pdf in st.session_state.pdfs]:
                st.session_state.pdfs.append(uploaded_file)

    # Add/Remove PDF paths form
    file_path_saved = os.path.join(temp_files_dir, uploaded_file.name)
    # Display a container with the PDFs uploaded by the user so far
    with st.container():
        if st.session_state.pdfs:
            st.header('PDFs uploaded:')
            for idx, pdf in enumerate(st.session_state.pdfs):
                st.write(pdf.name)
                st.button(label='Remove', key=f"Remove_{pdf.name}_{idx}", on_click=remove_pdf, kwargs={'pdf_to_remove': pdf})
                st.divider()

        if st.session_state.pdf_paths:
            st.header('PDF paths uploaded:')
            for idx, pdf_path in enumerate(st.session_state.pdf_paths):
                st.write(pdf_path)
                st.button(label='Remove', key=f"Remove_{pdf_path}_{idx}", on_click=remove_pdf, kwargs={'pdf_to_remove': pdf_path})
                st.divider()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query_text := st.chat_input("Your message"):

    os.environ['OPENAI_API_KEY'] = openai_api_key

    # Display user message in chat message container, and append to session state
    st.chat_message("user").markdown(query_text)
    st.session_state.messages.append({"role": "user", "content": query_text})

    # Check if FAISS index already exists, or if it needs to be created as it includes new PDFs
    session_pdfs = st.session_state.pdfs
    session_pdf_paths = st.session_state.pdf_paths
    if (st.session_state['faiss_index']['index'] is None or 
        set(st.session_state['faiss_index']['indexed_pdfs']) != set([pdf.name for pdf in session_pdfs]) or
        set(st.session_state['faiss_index']['indexed_pdfs']) != set(session_pdf_paths)):
        
        st.session_state['faiss_index']['indexed_pdfs'] = [pdf.name for pdf in session_pdfs] + session_pdf_paths
        
        with st.spinner('Indexing PDFs...'):          
            faiss_index = index_uploaded_pdfs(file_path_saved)
            faiss_index_paths = index_uploaded_pdfs_from_paths(file_path_saved)
            for pdf in session_pdfs:
                pdf_text = extract_text_from_pdf(pdf)
                faiss_index.add_texts([pdf_text])
            for pdf_path in session_pdf_paths:
                pdf_text = extract_text_from_pdf(pdf_path)
                faiss_index_paths.add_texts([pdf_text])
            st.session_state['faiss_index']['index'] = faiss_index
    else:
        faiss_index = st.session_state['faiss_index']['index']

    # Check if conversation memory has already been initialized and is part of the session state
    if st.session_state['conversation_memory'] is None:
        conversation = initialize_chat_conversation(faiss_index)
        st.session_state['conversation_memory'] = conversation
    else:
        conversation = st.session_state['conversation_memory']

    # Search PDF snippets using the last few user messages
    user_messages_history = [message['content'] for message in st.session_state.messages[-search_number_messages:] if message['role'] == 'user']
    user_messages_history = '\n'.join(user_messages_history)

    with st.spinner('Querying OpenAI GPT...'):
        response = conversation.predict(input=query_text, user_messages_history=user_messages_history)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        snippet_memory = conversation.memory.memories[1]
        for page_number, snippet in zip(snippet_memory.pages, snippet_memory.snippets):
            with st.expander(f'Snippet from page {page_number + 1}'):
                # Remove the <START> and <END> tags from the snippets before displaying them
                snippet = re.sub("<START_SNIPPET_PAGE_\d+>", '', snippet)
                snippet = re.sub("<END_SNIPPET_PAGE_\d+>", '', snippet)
                st.markdown(snippet)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
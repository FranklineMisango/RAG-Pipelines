import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import openai
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Set Streamlit app title
st.set_page_config(page_title='📚 Frankline & Co. LP. Self Rostering RAG', layout='wide')
st.title('Frankline & Co. LP. Self Rostering RAG for Langchain & GPT3.xx LLM')

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "pdfs" not in st.session_state:
    st.session_state.pdfs = []

# Sidebar for PDF upload
st.sidebar.header("📁 Upload PDFs")
temp_files_dir = "temp_files"
if not os.path.exists(temp_files_dir):
    os.makedirs(temp_files_dir)

# File upload
uploaded_files = st.sidebar.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_files_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.pdfs.append(file_path)

# Display uploaded PDFs
if st.session_state.pdfs:
    st.sidebar.subheader('Uploaded PDFs:')
    for pdf_path in st.session_state.pdfs:
        st.sidebar.write(os.path.basename(pdf_path))

# Chat input
if query_text := st.chat_input("Your message here..."):
    # Show the user message
    st.chat_message("user").markdown(query_text)
    st.session_state.conversation.append({"role": "user", "content": query_text})

    if st.session_state.pdfs:
        # Load the first PDF document using PyPDFLoader
        pdf_loader = PyPDFLoader(st.session_state.pdfs[0])
        pdf_pages = pdf_loader.load_and_split()

        # Initialize Chroma vector store with OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(documents=pdf_pages, embedding=embeddings)

        # RAG chain setup
        rag_chain = (
            {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
            | hub.pull("rlm/rag-prompt")
            | llm
            | StrOutputParser()
        )

        # Generate response using RAG chain
        with st.spinner("Processing your request..."):
            response = rag_chain.invoke(query_text)

        # Show the bot response
        st.chat_message("assistant").markdown(response)
        st.session_state.conversation.append({"role": "bot", "content": response})

# Display conversation history
if st.session_state.conversation:
    st.subheader("🔍 Conversation History:")
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])

# Add a footer
st.markdown("---")
st.markdown("### About This App")
st.markdown("This application leverages the power of Langchain and OpenAI to provide intelligent responses based on your uploaded documents. Feel free to upload your PDFs and ask questions!")

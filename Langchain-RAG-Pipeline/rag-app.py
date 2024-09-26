import os
import streamlit as st
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from langchain_community.vectorstores import Chroma
import openai

from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI instance (assuming ChatOpenAI is your intended usage)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Set Streamlit app title
st.title('Frankline & Co. LP. Self Rostering RAG for Langchain & GPT3.xx LLM')

# Ensure temp_files directory exists or create it
temp_files_dir = "temp_files"
if not os.path.exists(temp_files_dir):
    os.makedirs(temp_files_dir)

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to the temp_files directory
    file_path = os.path.join(temp_files_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF document using PyPDFLoader
    pdf_loader = PyPDFLoader(file_path)
    pdf_pages = pdf_loader.load_and_split()

    # Initialize Chroma vector store with OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(documents=pdf_pages, embedding=embeddings)

    # Function to format documents for RAG
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG chain setup
    rag_chain = (
        {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | hub.pull("rlm/rag-prompt")
        | llm
        | StrOutputParser()
    )

    # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Create a text input box for the user
    prompt = st.text_input(f'Input your question or query here (Type "exit" to quit)')
    st.warning("Wait for the loader to process your prompt before clicking submit")

    # Check if user submitted input
    if st.button('Submit'):
        # Exit condition
        if prompt.lower() == 'exit':
            st.write('Exiting the application. Thank you!')
        else:
            # Generate response using RAG chain
            response = rag_chain.invoke(prompt)

            # Append user input and bot response to conversation history
            st.session_state.conversation.append({"role": "user", "content": prompt})
            st.session_state.conversation.append({"role": "bot", "content": response})

            # Remove the temporary file after processing
            if os.path.exists(file_path):
                os.remove(file_path)

    # Display conversation history
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Bot:** {message['content']}")
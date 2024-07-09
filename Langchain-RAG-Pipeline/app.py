import os
import streamlit as st
from config import OPEN_AI_API_KEY
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from langchain_community.vectorstores import Chroma
import openai


os.environ['OPENAI_API_KEY'] = OPEN_AI_API_KEY

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Set Streamlit app title
st.title('Self Rostering RAG Langchain LLM')
st.success('This app allows you to interact with Aquila, your financial digital assistant.')

# Ensure temp_files directory exists or create it
temp_files_dir = "temp_files"
if not os.path.exists(temp_files_dir):
    os.makedirs(temp_files_dir)

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to the temp_files directory
    with open(os.path.join(temp_files_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Get the file path of the uploaded file
    file_path = os.path.join(temp_files_dir, uploaded_file.name)

    # Load the PDF document using PyPDFLoader
    loader = PyPDFLoader(file_path)

    # Split pages from the PDF
    pages = loader.load_and_split()

    # Load documents into the vector database (ChromaDB)
    store = Chroma.from_documents(pages, embeddings, collection_name='uploaded_document')

    # Create a vectorstore info object
    vectorstore_info = VectorStoreInfo(
        name="uploaded_document",
        description="Uploaded financial document as a PDF",
        vectorstore=store
    )

    # Create a vectorstore toolkit with llm parameter
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)


# Create a text input box for the user
prompt = st.text_input(f'Input your question or query here (Type "exit" to quit)')

# Check if user submitted input
if st.button('Submit'):
    # Exit condition
    if prompt.lower() == 'exit':
        st.write('Exiting the application. Thank you!')

    # Create a vectorstore agent
    agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

    # Pass the prompt to the agent
    response = agent_executor.run(prompt)

    # Write the response to the screen
    st.write(response)

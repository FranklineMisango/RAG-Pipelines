from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pypdfium2 as pdfium
from constants import chunk_size, chunk_overlap, number_snippets_to_retrieve
import os

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')

def __update_metadata(pages, pdf_name):
    """
    Add to the document metadata the title and original PDF name
    """
    for page in pages:
        pdf = pdfium.PdfDocument(page.metadata['source'])
        title = pdf.get_metadata_dict().get('Title', pdf_name)
        page.metadata['source'] = pdf_name
        page.metadata['title'] = title
    return pages

def index_pdfs(pdfs, from_paths=False) -> FAISS:
    """
    Index a list of uploaded PDFs or PDFs from file paths
    """
    all_pages = []
    for pdf in pdfs:
        if from_paths:
            loader = PyPDFium2Loader(pdf)
            pdf_name = pdf.split('/')[-1]
        else:
            loader = PyPDFium2Loader(pdf)
            pdf_name = pdf.name

        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pages = loader.load_and_split(splitter)
        pages = __update_metadata(pages, pdf_name)
        all_pages += pages

    faiss_index = FAISS.from_documents(all_pages, OpenAIEmbeddings())
    return faiss_index

def search_faiss_index(faiss_index: FAISS, query: str, top_k: int = number_snippets_to_retrieve) -> list:
    """
    Search a FAISS index, using the passed query
    """
    docs = faiss_index.similarity_search(query, k=top_k)
    return docs

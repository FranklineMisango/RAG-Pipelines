from langchain import FAISS
from langchain.document_loaders import PyPDFium2Loader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pypdfium2 as pdfium
from constants import chunk_size, chunk_overlap, number_snippets_to_retrieve


def index_uploaded_pdfs(pdfs: list) -> FAISS:
    """
    Index a list of uploaded PDFs
    """

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

    all_pages = []
    for pdf in pdfs:
        loader = PyPDFium2Loader(pdf)
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pages = loader.load_and_split(splitter)
        pages = __update_metadata(pages, pdf.name)
        all_pages += pages

    faiss_index = FAISS.from_documents(all_pages, OpenAIEmbeddings())

    return faiss_index


def search_faiss_index(faiss_index: FAISS, query: str, top_k: int = number_snippets_to_retrieve) -> list:
    """
    Search a FAISS index, using the passed query
    """

    docs = faiss_index.similarity_search(query, k=top_k)

    return docs
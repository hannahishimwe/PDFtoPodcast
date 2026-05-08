from urllib.parse import urlparse
from langchain_community.vectorstores import FAISS
from app.settings import embeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import requests

class PDFLoader:
    MIN_ATTRIBUTES = ('scheme', 'netloc')
    def init(self, link):
        try: 
            self.pdf = self.is_valid(link)
            print("Valid URL")
        except ValueError:
            print("Invalid URL")
            self.pdf = ""
        self.split_docs = None
        self.vector_store = None
        self.docs = None
    def handle_pdf(self, link):
        self.load_pdf(link)
        self.split_documents(self.docs)
        self.create_vector_store(self.split_docs)

    def load_pdf(self, link):
        loader = PDFMinerLoader(link)
        self.docs = loader.load()
    
    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.split_docs = text_splitter.split_documents(documents)

    def create_vector_store(self, split_docs):
        self.vector_store = FAISS.from_documents(split_docs, embedding=embeddings)

    def is_valid(self, url, qualifying=MIN_ATTRIBUTES):
        tokens = urlparse(url)
        if all(getattr(tokens, qualifying_attr) for qualifying_attr in qualifying):
            r = requests.get(url)
            content_type = r.headers.get('content-type', '')
            if 'application/pdf' in content_type.lower():
                return True
        else:
            raise ValueError("Invalid PDF URL")
        





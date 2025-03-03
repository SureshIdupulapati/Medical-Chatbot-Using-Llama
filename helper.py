from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls = PyPDFLoader
                             )
    documents = loader.load()
    return documents


def text_chunking(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    
    return text_chunks


def download_hugging_face_embeddings():
    
    model = "all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(model_name=model,model_kwargs={"device": "cpu"})
    
    return embeddings


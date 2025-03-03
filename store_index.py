from src.helper import load_pdf, text_chunking, download_hugging_face_embeddings
import os

extracted_data = load_pdf("data")

text_chunks = text_chunking(extracted_data)
print(len(text_chunks))

embeddings = download_hugging_face_embeddings()

import os
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

# Ensure correct absolute path (use forward slashes)
persist_path = os.path.abspath("C:/Users/Sures/jupyter_programs/Open-AI/Medical-Chatbot-Using-Llama2/chroma_db")

# Initialize Chroma with persistence enabled
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory=persist_path,  # Ensure this is set
)


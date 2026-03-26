# app/database.py
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Centralizamos los embeddings y el cliente
embeddings = HuggingFaceEmbeddings()
client = chromadb.PersistentClient(path="app/chroma_data")

vector_store = Chroma(
    client=client,
    collection_name="knowledge_base",
    embedding_function=embeddings
)
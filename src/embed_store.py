# Vector store builder
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

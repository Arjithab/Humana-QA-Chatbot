# Unstructured PDF loader
from langchain_unstructured import UnstructuredLoader

def extract_text_from_pdf(pdf_path):
    loader = UnstructuredLoader(pdf_path)
    documents = loader.load()
    text = "\n".join([doc.page_content for doc in documents])
    return text

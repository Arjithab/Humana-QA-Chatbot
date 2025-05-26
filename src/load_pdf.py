# Unstructured PDF loader
from langchain_community.document_loaders import UnstructuredFileLoader

def extract_text_from_pdf(pdf_path):
    loader = UnstructuredFileLoader(pdf_path)
    documents = loader.load()
    text = "\n".join([doc.page_content for doc in documents])
    return text

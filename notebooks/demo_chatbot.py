# Script version of chatbot demo
# # 💬 HER-2/neu Chatbot Demo
# This notebook demonstrates the full end-to-end pipeline for document-based Q/A using RAG.

# Step 1: Extract text from PDF
from src.load_pdf import extract_text_from_pdf

pdf_path = 'data/her2_paper.pdf'
raw_text = extract_text_from_pdf(pdf_path)
print(raw_text[:500])

# Step 2: Chunk the text
from src.chunking import chunk_text

documents = chunk_text(raw_text)
print(f'Created {len(documents)} chunks')

# Step 3: Build FAISS vector store
from src.embed_store import build_vector_store

vector_store = build_vector_store(documents)
print('Vector store created')

# Step 4: Setup RAG pipeline
from src.qa_pipeline import setup_qa_pipeline

qa = setup_qa_pipeline(vector_store)
print('QA system ready')

# Step 5: Ask a question
query = 'What is HER-2/neu and why is it important in breast cancer?'
response = qa.run(query)
print(response)

# Step 6: Evaluate using test set
from src.evaluate import evaluate_model
evaluate_model(qa.run, 'data/sample_questions.json')

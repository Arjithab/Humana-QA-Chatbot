# 🧬 HER-2/neu Q&A Chatbot

A document-grounded chatbot that answers questions about HER-2/neu oncogene amplification and prognosis in breast cancer, using a research paper and an open-source LLM (BioGPT). The chatbot supports continuous improvement via feedback logging, automated evaluation, and human-in-the-loop correction.

---

## 🚀 Features

- 🔍 Answers biomedical questions from a specific research paper (RAG + BioGPT)
- 📄 Built on a PDF publication from *Science, 1987* (Slamon et al.)
- 👍 Accepts thumbs-up/down feedback from users
- 🧪 Logs and evaluates semantic accuracy using BERTScore and cosine similarity
- 🔁 Supports continuous improvement via labeled corrections and versioning

---

## 🛠️ Technologies Used

- Python, Streamlit, LangChain
- `microsoft/BioGPT-Large` for biomedical Q&A
- FAISS for semantic retrieval
- `bert-score`, `sentence-transformers` for evaluation
- Remote logging via `.streamlit/secrets.toml`

---




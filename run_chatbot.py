import streamlit as st
import json
import time
from datetime import datetime
import uuid
from src.load_pdf import extract_text_from_pdf
from src.chunking import chunk_text
from src.embed_store import build_vector_store
from src.qa_pipeline import setup_qa_chain
import os

os.environ["TORCH_DISABLE_TELEMETRY"] = "1"

# First Streamlit call
st.set_page_config(page_title="HER-2/neu Q&A Chatbot", page_icon="🧬")

# Session + version tracking
MODEL_VERSION = "BioGPT-v1.1"
SESSION_ID = str(uuid.uuid4())

@st.cache_resource
def load_chain():
    text = extract_text_from_pdf("data/her2_paper.pdf")
    chunks = chunk_text(text)
    vectorstore = build_vector_store(chunks)
    return setup_qa_chain(vectorstore)

qa_chain = load_chain()

# Streamlit layout
st.title("🧬 HER-2/neu Biomedical Chatbot")
st.markdown("Ask questions based on the HER-2/neu research paper. The chatbot will retrieve relevant sections and generate an answer.")

# Chat input
question = st.text_input("🔎 Ask a question:")

if question:
    start_time = time.time()
    response = qa_chain.run(question)
    latency = round(time.time() - start_time, 2)

    st.markdown("### 🤖 Answer")
    st.write(response)
    st.caption(f"⏱️ Responded in {latency:.2f} seconds")

    # Feedback
    st.markdown("### 🙋 Was this answer helpful?")
    col1, col2 = st.columns(2)
    thumbs_up = col1.button("👍 Yes")
    thumbs_down = col2.button("👎 No")
    comment = st.text_area("💬 Additional feedback", "")

    if thumbs_up or thumbs_down:
        feedback = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": SESSION_ID,
            "question": question,
            "response": response,
            "latency": latency,
            "model_version": MODEL_VERSION,
            "feedback": {
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "comment": comment
            }
        }
        with open("chatbot_logs.jsonl", "a") as f:
            f.write(json.dumps(feedback) + "\n")
        st.success("✅ Feedback saved. Thank you!")


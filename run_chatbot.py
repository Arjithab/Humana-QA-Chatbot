import streamlit as st
import json
import time
from datetime import datetime
import uuid
from src.load_pdf import extract_text_from_pdf
from src.chunking import chunk_text
from src.embed_store import build_vector_store
from src.qa_pipeline import setup_qa_chain

MODEL_VERSION = "BioGPT-v1.1"
SESSION_ID = str(uuid.uuid4())

@st.cache_resource
def load_qa_chain():
    text = extract_text_from_pdf("data/her2_paper.pdf")
    chunks = chunk_text(text)
    vectorstore = build_vector_store(chunks)
    return setup_qa_chain(vectorstore), chunks

qa_chain, all_chunks = load_qa_chain()

st.set_page_config(page_title="HER-2/neu Chatbot", page_icon="🧬")
st.title("🧬 HER-2/neu Biomedical Chatbot")
st.markdown("Ask research-based questions about HER-2/neu in breast cancer.")

# Captures thumbs-up/down feedback, latency, model version, and session ID for each question

if question:
    start_time = time.time()
    response = qa_chain.run(question)
    latency = time.time() - start_time

    st.markdown(f"**🤖 Answer:** {response}")
    st.caption(f"⏱️ {latency:.2f}s")

    col1, col2 = st.columns(2)
    thumbs_up = col1.button("👍 Helpful")
    thumbs_down = col2.button("👎 Not helpful")
    comment = st.text_area("💬 Any feedback?", "")

    if thumbs_up or thumbs_down:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": SESSION_ID,
            "question": question,
            "response": response,
            "model_version": MODEL_VERSION,
            "latency_sec": round(latency, 2),
            "feedback": {
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "comment": comment
            }
        }
        with open("chatbot_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        st.success("Feedback saved.")

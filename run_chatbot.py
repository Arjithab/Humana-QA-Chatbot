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
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pytorch_utils")
os.environ["TORCH_DISABLE_TELEMETRY"] = "1"

# First Streamlit call
st.set_page_config(page_title="HER-2/neu Q&A Chatbot", page_icon="🧬")

# Session + version tracking
#MODEL_VERSION = "BioGPT-v1.1"
#SESSION_ID = str(uuid.uuid4())

##Session tracking
MODEL_VERSION = "TinyGPT-Test"
SESSION_ID = str(uuid.uuid4())

# Init state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "question" not in st.session_state:
    st.session_state["question"] = ""

# Load model + vectorstore
@st.cache_resource
def load_chain():
    text = extract_text_from_pdf("data/her2_paper.pdf")
    chunks = chunk_text(text)
    vectorstore = build_vector_store(chunks)
    return setup_qa_chain(vectorstore)

qa_chain = load_chain()

# Show chat history
for entry in st.session_state["messages"]:
    st.markdown(f"**👤 You:** {entry['question']}")
    st.markdown(f"**🤖 Chatbot:** {entry['answer']}")
    st.divider()

# Input box (persistent key, empty after submit)
question = st.text_input("🔎 Ask a question about the paper:", value="", key="question_input")

# When user submits a question
if question:
    with st.spinner("🤖 Thinking..."):
        try:
            start_time = time.time()
            response = qa_chain.invoke({"query": question})
            duration = round(time.time() - start_time, 2)

            if isinstance(response, dict):
                answer = response.get("result", "")
                sources = response.get("source_documents", [])
                source_texts = [doc.page_content for doc in sources if hasattr(doc, "page_content")]
            else:
                answer = str(response)
                source_texts = []

            # Add to conversation memory
            st.session_state["messages"].append({
                "question": question,
                "answer": answer,
                "sources": source_texts,
                "latency": duration
            })

            # Re-render this entry immediately
            st.markdown(f"**👤 You:** {question}")
            st.markdown(f"**🤖 Chatbot:** {answer}")
            st.caption(f"⏱️ Responded in {duration} seconds")
            st.divider()

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    # ✅ Clear input box after response (simulating ChatGPT behavior)
    st.session_state["question_input"] = ""

# Optional Feedback (at bottom, always available)
with st.expander("🙋 Give feedback on the last answer"):
    if st.session_state["messages"]:
        last = st.session_state["messages"][-1]
        col1, col2 = st.columns(2)
        thumbs_up = col1.button("👍 Helpful", key="up")
        thumbs_down = col2.button("👎 Not helpful", key="down")
        comment = st.text_area("💬 Additional feedback", "", key="feedback_comment")

        if thumbs_up or thumbs_down:
            feedback = {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": SESSION_ID,
                "question": last["question"],
                "response": last["answer"],
                "sources": last["sources"],
                "latency": last["latency"],
                "model_version": MODEL_VERSION,
                "feedback": {
                    "thumbs_up": thumbs_up,
                    "thumbs_down": thumbs_down,
                    "comment": comment
                }
            }
            with open("chatbot_logs.jsonl", "a") as f:
                f.write(json.dumps(feedback) + "\n")
            st.success("✅ Feedback saved!")

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

# Streamlit layout
st.title("🧬 HER-2/neu Biomedical Chatbot")
st.markdown("Ask questions based on the HER-2/neu research paper. The chatbot will retrieve relevant sections and generate an answer.")

# Debug step 1: load vectorstore and chain
st.info("🔄 Initializing chatbot backend...")
start_load = time.time()

@st.cache_resource
def load_chain():
    text = extract_text_from_pdf("data/her2_paper.pdf")
    chunks = chunk_text(text)
    vectorstore = build_vector_store(chunks)
    return setup_qa_chain(vectorstore)

qa_chain = load_chain()

st.success(f"Chatbot ready in {round(time.time() - start_load, 2)} seconds")

# Chat input
question = st.text_input("🔎 Ask a question:")

if question:
    st.write("📩 Question received:", question)
    with st.spinner("🤖 Generating answer..."):
        start_time = time.time()    

    try:
        response = qa_chain.invoke({"query": question})
        
        if isinstance(response, dict):
            answer = response.get("result", "")
            sources = response.get("source_documents", [])
            # Extract only the source text, not full Document objects
            source_texts = [ doc.page_content for doc in sources if hasattr(doc, "page_content")]
        else:
            answer = str(response)
            source_texts = []      
            
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
    
        # 🔧 ✅ Only write feedback if button clicked
        if thumbs_up or thumbs_down:
            feedback = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "session_id": SESSION_ID,
                        "question": question,
                        "response": answer,       # JSON-safe string
                        "sources": source_texts,  # SON-safe list of strings
                        "latency": duration,
                        "model_version": MODEL_VERSION,
                        "feedback": {
                            "thumbs_up": thumbs_up,
                            "thumbs_down": thumbs_down,
                            "comment": comment
                        }
                    }
    
            with open("chatbot_logs.jsonl", "a") as f:
                f.write(json.dumps(feedback) + "\n")
    
            st.success("Feedback saved!")

        except Exception as e:
            st.error(f"⚠️ Error generating response: {e}")


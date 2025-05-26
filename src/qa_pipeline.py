# RAG QA pipeline

#For local testing
import torch
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

def setup_qa_chain(vectorstore, model_name="microsoft/BioGPT-Large"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = 0 if torch.backends.mps.is_available() else -1
    
    # Define pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        return_full_text=False
    )

    # Wrap in LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    # Setup retrieval-augmented QA
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    return qa_chain

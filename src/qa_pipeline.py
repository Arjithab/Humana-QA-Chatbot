# RAG QA pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_qa_chain(vectorstore):
    # Load BioGPT model
    model_name = "microsoft/BioGPT-Large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Select appropriate device
    if torch.backends.mps.is_available():
        device_index = 0
    elif torch.cuda.is_available():
        device_index = 0
    else:
        device_index = -1  # CPU

    # Build text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_index,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Construct proper LangChain PromptTemplate
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Return a RetrievalQA chain using BioGPT
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

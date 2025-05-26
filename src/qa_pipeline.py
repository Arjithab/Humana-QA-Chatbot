# RAG QA pipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

def setup_qa_chain(vectorstore, model_name="microsoft/BioGPT-Large"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        return_full_text=False
    )

    # Wrap in LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    # Setup retrieval-augmented QA
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain

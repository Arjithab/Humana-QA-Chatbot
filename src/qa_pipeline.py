# RAG QA pipeline
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def setup_qa_pipeline(vector_store):
    model_name = "tiiuae/falcon-7b-instruct"  # or mistralai/Mistral-7B-Instruct
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

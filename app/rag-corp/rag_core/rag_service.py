from langchain.chains import RetrievalQA

def build_qa_chain(vectorstore, llm, k: int):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

def answer(qa_chain, query: str) -> str:
    return qa_chain.run(query)

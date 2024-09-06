import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain import hub

# LLM
llm = ChatGroq(model=st.session_state.llm, api_key=os.getenv("GROQ_API_KEY"))
retriever = (st.session_state.vectorstore).as_retriever(search_kwargs={"k": 20})

# Reranker and retriever
compressor = CohereRerank(top_n=5, model="rerank-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Generate response (Reranker)
def reranker_rag_response(question):
    prompt = hub.pull("rlm/rag-prompt")

    reranked_rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return reranked_rag_chain.invoke(question)

# Final response
if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        response = reranker_rag_response(st.session_state.messages[-1]["content"])
        print("\n\compression_retriever: ", compression_retriever.invoke(st.session_state.messages[-1]["content"]))
        st.chat_message("assistant", avatar="assets/bot.png").markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
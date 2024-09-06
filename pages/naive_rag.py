import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatGroq(model=st.session_state.llm, api_key=os.getenv("GROQ_API_KEY"))
retriever = (st.session_state.vectorstore).as_retriever()

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Generate response (Naive RAG)
def naive_rag_response(question):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        response = naive_rag_response(st.session_state.messages[-1]["content"])
        st.chat_message("assistant", avatar="assets/bot.png").markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
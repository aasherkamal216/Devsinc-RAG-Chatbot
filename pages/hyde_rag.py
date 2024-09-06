import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub


# HyDE document genration
template = """Please write a scientific paper passage to answer the question below.
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatGroq(model=st.session_state.llm, api_key=os.getenv("GROQ_API_KEY"))
retriever = (st.session_state.vectorstore).as_retriever()

# Genera a scientific paper passage
generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser() 
)

# HyDe chain
retrieval_chain = generate_docs_for_retrieval | retriever 

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Generate response (Hypothetical Document Embeddings)
def hyde_rag_response(question):
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

# Final response
if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        response = hyde_rag_response(st.session_state.messages[-1]["content"])
        print("\n\nHyDe retrieval_chain: ", retrieval_chain.invoke(st.session_state.messages[-1]["content"]))
        st.chat_message("assistant", avatar="assets/bot.png").markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain import hub


# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate
three different versions of the given user question to retrieve relevant
documents from a vector database. By generating multiple perspectives on the user question,
your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
Provide three alternative questions separated by newlines.
Original question: 
{question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatGroq(model=st.session_state.llm, api_key=os.getenv("GROQ_API_KEY"))
retriever = (st.session_state.vectorstore).as_retriever()

# Get unique union of retrieved docs
def get_unique_union(documents: list[list]):
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))

    return [loads(doc) for doc in unique_docs]


# Generate 3 queries
generate_queries = (
    prompt_perspectives
    | ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Generate response (Multi-Query Perspective)
def multi_query_rag_response(question):
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
        response = multi_query_rag_response(st.session_state.messages[-1]["content"])
        print("\n\nMulti-Query retrieval_chain: ", retrieval_chain.invoke(st.session_state.messages[-1]["content"]))
        st.chat_message("assistant", avatar="assets/bot.png").markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
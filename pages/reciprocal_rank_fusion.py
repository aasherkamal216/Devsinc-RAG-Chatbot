import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain import hub

# LLM
llm = ChatGroq(model=st.session_state.llm, api_key=os.getenv("GROQ_API_KEY"))
retriever = (st.session_state.vectorstore).as_retriever()

# Prompt
template = """You are an assistant that generates multiple search queries based on a single input query.
Generate four search queries related to the question: {question} \n
Return only the search queries.
Output:
"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

# Generate search queries
generate_search_queries = (
    prompt_rag_fusion 
    | ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

# RRF function
def reciprocal_rank_fusion(results: list[list], k=60):

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

# RRF chain
retrieval_chain_rag_fusion = generate_search_queries | retriever.map() | reciprocal_rank_fusion

# Generate response (Reciprocal Rank Fusion)
def reciprocal_rag_fusion(question):
    prompt = hub.pull("rlm/rag-prompt")

    rrf_rag_chain = (
        {"context": retrieval_chain_rag_fusion, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rrf_rag_chain.invoke(question)

# Final response
if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        response = reciprocal_rag_fusion(st.session_state.messages[-1]["content"])
        print("\n\RRF_rag_chain: ", retrieval_chain_rag_fusion.invoke(st.session_state.messages[-1]["content"]))
        st.chat_message("assistant", avatar="assets/bot.png").markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
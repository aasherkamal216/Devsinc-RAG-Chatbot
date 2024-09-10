import streamlit as st
from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from utils.self_rag_utils import retrieve
from utils.crag_utils import (contextualize_question,
                              grade_documents,
                              transform_query, web_search,
                              decide_to_generate, generate)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        chat_history: previous conversation history
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    chat_history: List[Dict[str, str]]
    generation: str
    web_search: str
    documents: List[str]


if "crag" not in st.session_state:
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("contextualize_question", contextualize_question)  # contextualize question
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search_node", web_search)  # web search

    # Build graph
    workflow.add_edge(START, "contextualize_question")
    workflow.add_edge("contextualize_question", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    # Compile
    st.session_state.crag = workflow.compile()

# Run the app
if st.session_state.messages[-1]["role"] == "user":
    inputs = {"question": st.session_state.messages[-1]["content"],
              "chat_history": st.session_state.messages[:-1]}

    with st.spinner("Thinking..."):
        print("=========Corrective-RAG=========")
        for output in st.session_state.crag.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"\n\nNode '{key.upper()}':")
                print("\n", value)
        # Final generation
        response = value["generation"]
        st.chat_message("assistant", avatar="assets/bot.png").markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

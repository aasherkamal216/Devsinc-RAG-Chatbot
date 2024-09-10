import streamlit as st
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from utils.self_rag_utils import (retrieve, generate,
                                  decide_to_generate,
                                  grade_documents,
                                  transform_query,
                                  grade_generation_vs_documents_and_question)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        iteration: number of iterations
    """

    question: str
    generation: str
    documents: List[str]
    iteration: int


if "self_rag" not in st.session_state:
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_vs_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
            "stop": END
        },
    )

    st.session_state.self_rag = workflow.compile()


# Run the app
if st.session_state.messages[-1]["role"] == "user":
    inputs = {"question": st.session_state.messages[-1]["content"], "iteration": 0}

    with st.spinner("Thinking..."):
        print("=========Self-RAG=========")
        for output in st.session_state.self_rag.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"\n\nNode '{key.upper()}':")
                print("\n", value)
        # Final response
        response = value["generation"]
        st.chat_message("assistant", avatar="assets/bot.png").markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

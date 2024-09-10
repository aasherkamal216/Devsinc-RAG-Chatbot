import streamlit as st
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from utils.self_rag_utils import retrieve
from utils.agent_rag_utils import rewrite, agent, grade_document, generate_answer, retriever_tool

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

if "agentic_rag" not in st.session_state:

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate_answer
    )  # Generating a response after we know the documents are relevant

    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_document,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    st.session_state.agentic_rag = workflow.compile()

# Run the app
if st.session_state.messages[-1]["role"] == "user":
    inputs = {
    "messages": [
        ("user", st.session_state.messages[-1]["content"]),
    ]
}

    with st.spinner("Thinking..."):
        print("=========Agentic-RAG=========")
        for output in st.session_state.agentic_rag.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"\n\nOutput from node '{key.upper()}':")

                print("\n", value)
        # Final generation
        response = value["messages"][0].content if hasattr(value["messages"][0], "content") else value["messages"][0]
        st.chat_message("assistant", avatar="assets/bot.png").markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

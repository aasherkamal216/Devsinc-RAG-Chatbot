import streamlit as st

# Devsinc logo
st.logo("logo.png")

st.set_page_config(page_title="RAG Techniques")
st.title("RAG Techniques")

if "messages" not in st.session_state:
    st.session_state.messages = []

if question := st.chat_input("Write your message here"):
    st.session_state.messages.append({"role": "user", "content": question})

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


p1 = st.Page("pages/naive_rag.py", title="Naive RAG", icon=":material/description:")
p2 = st.Page("pages/multi_query_rag.py", title="Multi-Query Perspective", icon=":material/quiz:")
p3 = st.Page("pages/hyde_rag.py", title="HyDe", icon=":material/science:")
p4 = st.Page("pages/reranker_rag.py", title="Reranker", icon=":material/trending_up:")
p5 = st.Page("pages/rag_fusion.py", title="RAG Fusion", icon=":material/join:")
p6 = st.Page("pages/self_rag.py", title="Self-RAG", icon=":material/emoji_objects:")
p7 = st.Page("pages/corrective_rag.py", title="Corrective RAG", icon=":material/all_match:")
p8 = st.Page("pages/agentic_rag.py", title="Agentic RAG", icon=":material/smart_toy:")
p9 = st.Page("pages/adaptive_rag.py", title="Adaptive RAG", icon=":material/compare_arrows:")

pg = st.navigation({"Rag Models:":[p1, p2, p3, p4, p5, p6, p7, p8]})

pg.run()
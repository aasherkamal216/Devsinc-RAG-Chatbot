import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Devsinc logo
st.logo("assets/logo.png")

st.set_page_config(page_title="RAG Techniques", layout="wide")

###--- Title ---###
st.markdown("""
    <h1 style='text-align: center;'>
        <span style='color: #00EADE;'>Devsinc</span> 
        <span style='color: #ffffff;'>AI Assistant</span>
    </h1>
""", unsafe_allow_html=True)

# Initialize session state variables
if "vectorstore" not in st.session_state:
    with st.spinner(":green[Loading Vector Store. This may take a moment.]"):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        st.session_state.vectorstore = FAISS.load_local("devsinc_vectorstore", embeddings=embeddings, allow_dangerous_deserialization=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

if "llm" not in st.session_state:
    st.session_state.llm = "llama-3.1-8b-instant"

# Function to reset chat
def reset_chat():
    if len(st.session_state.messages) > 1:
        del st.session_state.messages

# LLM selection and Reset button
col1, col2 = st.columns([1,3], vertical_alignment="bottom")
with col1:
    st.button("🗑️ Reset Chat", on_click=reset_chat, use_container_width=True)
with col2:
    st.session_state.llm = st.selectbox("Select an LLM:", ["llama-3.1-8b-instant", "gemma2-9b-it", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"])

# Chat input
if question := st.chat_input("Write your message here"):
    st.session_state.messages.append({"role": "user", "content": question})

# Displaying chat messages
for message in st.session_state.messages:
    avatar = "assets/user.png" if message["role"] == "user" else "assets/bot.png"
    st.chat_message(message["role"], avatar=avatar).write(message["content"])

# Pages
p1 = st.Page("pages/naive_rag.py", title="Naive RAG", icon=":material/description:")
p2 = st.Page("pages/multi_query_rag.py", title="Multi-Query Perspective", icon=":material/quiz:")
p3 = st.Page("pages/hyde_rag.py", title="HyDe", icon=":material/science:")
p4 = st.Page("pages/reranker_rag.py", title="Reranker", icon=":material/trending_up:")
p5 = st.Page("pages/reciprocal_rank_fusion.py", title="Reciprocal Rank Fusion", icon=":material/join:")
p6 = st.Page("pages/self_rag.py", title="Self-RAG", icon=":material/emoji_objects:")
p7 = st.Page("pages/corrective_rag.py", title="Corrective RAG", icon=":material/all_match:")
p8 = st.Page("pages/agentic_rag.py", title="Agentic RAG", icon=":material/smart_toy:")
p9 = st.Page("pages/adaptive_rag.py", title="Adaptive RAG", icon=":material/compare_arrows:")

pg = st.navigation({"Rag Techniques:":[p1, p2, p3, p4, p5, p6, p7, p8, p9]})

pg.run()

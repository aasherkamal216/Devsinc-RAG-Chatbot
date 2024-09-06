# Devsinc AI Assistant: RAG Techniques Showcase

This Streamlit application demonstrates various Retrieval-Augmented Generation (RAG) techniques using the Devsinc AI Assistant. It provides an interactive interface to explore different RAG methods and their applications in question-answering tasks.

## Features

- Interactive chat interface with AI assistant
- Multiple RAG techniques implemented:
  - Naive RAG
  - Multi-Query Perspective RAG
  - Hypothetical Document Embeddings (HyDE)
  - Reranker RAG
  - Reciprocal Rank Fusion
  - Self-RAG (placeholder)
  - Corrective RAG (placeholder)
  - Agentic RAG (placeholder)
  - Adaptive RAG (placeholder)
- LLM selection option
- Vector store integration using FAISS

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/aasherkamal216/Devsinc-RAG-Chatbot.git
   cd Devsinc-RAG-Chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Get your API key:
   - [Groq API key](https://console.groq.com/keys)
   - [Cohere API key](https://dashboard.cohere.com/api-keys)
   
4. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   COHERE_API_KEY=your_cohere_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

## Running the App

Run the Streamlit app:
```
streamlit run app.py
```


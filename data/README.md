# Devsinc RAG Chatbot: Data Preparation

This document outlines the process used to prepare data for the Devsinc AI Assistant RAG application.

## Data Processing Steps

1. **Web Scraping**: 
   - Used Langchain's RecursiveURLloader to scrape data from Devsinc's website.

2. **Manual Cleaning**:
   - Performed manual data cleaning to improve quality.

3. **Initial Chunking**:
   - Split data into 33 large chunks (~10,000 characters each) using Recursive Character Text Splitter.

4. **Pre-processing with AI**:
   - Utilized Google Gemini to pre-process each large chunk.

5. **Refined Chunking**:
   - Concatenated pre-processed chunks.
   - Re-split into 402 smaller chunks (~700 characters each).

6. **Title Generation**:
   - Used a Huggingface pre-trained 'Title Generator' model.
   - Created a pipeline to generate titles for each chunk.
   - Converted chunks to 'Document' objects.

7. **Vectorstore Creation**:
   - Created a FAISS vectorstore using the "all-mpnet-base-v2" embedding model.

8. **Application Integration**:
   - Integrated the saved vectorstore into the Streamlit application.

## Note
This data preparation process aims to structure information for the RAG system efficiently. However, the relevance of retrieved data may vary depending on the specific query and context.
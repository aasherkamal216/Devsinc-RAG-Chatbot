import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from typing import Literal

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

###----Question Router ----###
# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

llm = ChatGroq(model="gemma2-9b-it", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
route_system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to 'Devsinc' tech company, its services, staff,
blogs written on their website, terms and conditions, privacy policy and other relevant information.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system_prompt),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("###----ROUTE QUESTION----###")
    question = state["question"]
    source = question_router.invoke({"question": question})

    if source.datasource == "web_search":
        print("###---ROUTE QUESTION TO WEB SEARCH---###")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("###---ROUTE QUESTION TO RAG---###")
        return "vectorstore"
    
    
### --- Web Search --- ###
web_search_tool = TavilySearchResults(k=3)

###---- Adaptive-RAG Web Search ---###
def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("###---WEB SEARCH---###")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
from typing import Literal

os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

retriever = (st.session_state.vectorstore).as_retriever()

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


### --- Retrieval Grader --- ###

llm = ChatGroq(model="gemma2-9b-it", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader


### ---- Generate ---- ###
prompt = hub.pull("rlm/rag-prompt")
llm = ChatGroq(model=st.session_state.llm)
# Chain
rag_chain = prompt | llm | StrOutputParser()


### --- Question Contextualizer --- ###
contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is. RETURN Only the question."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("human", "Question: {question}\n\nChat History: {chat_history}"),
    ]
)
contextualize_q_chain = (
    contextualize_q_prompt 
    | ChatGroq(model='gemma2-9b-it', temperature=0) 
    | StrOutputParser()
    )


### --- Question Re-writer --- ###
q_rewriter_system = """You a question re-writer that converts an input question to a better version that is optimized
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
     Return only the re-written question."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", q_rewriter_system),
        ("human", "Here is the initial question:\n{question}"),
    ]
)
question_rewriter = re_write_prompt | ChatGroq(model='gemma2-9b-it', temperature=0)  | StrOutputParser()


### --- Web Search --- ###
web_search_tool = TavilySearchResults(k=3)


#####---- Functions for Graph State----#####

def contextualize_question(state):
    """
    Contextualize the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with contextualized question
    """
    print("###-----Contextualizing Question-----###")
    question = state["question"]
    chat_history = state["chat_history"]

    # Contextualize question
    contextualized_question = contextualize_q_chain.invoke({"question": question, "chat_history": chat_history})
    return {"question": contextualized_question}


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("###-----Retrieving Documents-----###")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("###-----Generating Answer-----###")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("###-----Checking Document Relevance to Question-----###")
    question = state["question"]
    documents = state["documents"]

    # Score each document
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("### -----GRADE: DOCUMENT RELEVANT----- ###")
            filtered_docs.append(d)
        else:
            print("### -----GRADE: DOCUMENT NOT RELEVANT----- ###")
            continue

    if len(filtered_docs) < 3:
        web_search = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("###-----Transforming Query-----###")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("###-----Web Search-----###")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}


### Edges

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("###-----Assessing Graded Documents-----###")
    web_search = state["web_search"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("### ---Decision: Documents Are Not Relevant to Question, Transform Query--- ###")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("### ---Decision: Generate--- ###")
        return "generate"



### --- Retriever Tool --- ###
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_devsinc_information",
    """Search and return information about Devsinc company,
    its services, staff, blogs written on their website, terms and conditions,
    privacy policy and other relevant information."""
)
tools = [retriever_tool]

llm = ChatGroq(model=st.session_state.llm, temperature=0)
### --- Agent --- ###
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("###-----Calling Agent-----###")
    messages = state["messages"]

    model = llm.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


### --- Grade Documents --- ###
def grade_document(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("###-----Checking Document Relevance to Question-----###")

    # LLM
    model = ChatGroq(model="gemma2-9b-it", temperature=0)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(GradeDocuments)

    # Chain
    chain = grade_prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "document": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("###-----DECISION: DOCS RELEVANT-----###")
        return "generate"

    else:
        print("###-----DECISION: DOCS NOT RELEVANT-----###")
        return "rewrite"


### --- Rewrite Question --- ###
def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("###-----Transforming Query-----###")
    messages = state["messages"]
    question = messages[0].content

    response = question_rewriter.invoke({"question": question})
    return {"messages": [response]}


### --- Generate --- ###
def generate_answer(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with LLM generation
    """
    print("###-----Generating Answer-----###")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}
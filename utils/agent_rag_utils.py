import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from typing import Literal

retriever = (st.session_state.vectorstore).as_retriever(search_kwargs={"k": 5})


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


### --- Retriever Tool --- ###
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_devsinc_information",
    """Search and return information about Devsinc company,
    its services, staff, blogs written on their website, terms and conditions,
    work history, privacy policy and other relevant information."""
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
# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

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
    


### --- Generate --- ###
prompt = hub.pull("rlm/rag-prompt")
llm = ChatGroq(model=st.session_state.llm)
# Chain
rag_chain = prompt | llm | StrOutputParser()


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
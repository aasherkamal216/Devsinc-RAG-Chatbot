import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

retriever = (st.session_state.vectorstore).as_retriever(search_kwargs={"k": 5})

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



### ---- Generate ---- ###
prompt = hub.pull("rlm/rag-prompt")
llm = ChatGroq(model=st.session_state.llm)
# Chain
rag_chain = prompt | llm | StrOutputParser()


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
    iteration = state["iteration"]
    print("\nIteration: ", iteration)
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "iteration": iteration+1}


### --- Retrieval Grader --- ###
# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatGroq(model="gemma2-9b-it", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

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
retrieval_grader = grade_prompt | structured_llm_grader


### --- Self-RAG Document Grader --- ###
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("###-----CHECK DOCUMENT RELEVANCE TO QUESTION----###")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("###-----GRADE: DOCS RELEVANT-----###")
            filtered_docs.append(d)
        else:
            print("###-----GRADE: DOCS NOT RELEVANT-----###")
            continue
    return {"documents": filtered_docs, "question": question}


### --- Question Rewriter for Retrieval--- ###
retrieval_system_prompt = """You are a question re-writer that converts an input question to a better version that is optimized
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
     Formulate and provide an improved question. Don't say anything else."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_system_prompt),
        ("human", "Here is the initial question: \n\n {question} \n "),
    ]
)

q_rewriter_for_retrieval = re_write_prompt | ChatGroq(model="llama-3.1-8b-instant", temperature=0) | StrOutputParser()



###--- Self-RAG Query Transformation ---###
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("###---TRANSFORM QUERY---###")
    question = state["question"]
    documents = state["documents"]
    iteration = state["iteration"]
    print("\nIteration: ", iteration)

    # Re-write question
    better_question = q_rewriter_for_retrieval.invoke({"question": question})
    return {"documents": documents, "question": better_question, "iteration": iteration+1}



###--- Self-RAG Decision to Generate ---###
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("###---ASSESS GRADED DOCUMENTS---###")
    filtered_documents = state["documents"]
    print("Iteration: ", state["iteration"])
    
    if not filtered_documents and state["iteration"] < 4:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "###---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, TRANSFORM QUERY---###"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("###---DECISION: GENERATE---###")
        return "generate"
    


### --- Hallucination Checker --- ###
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_hallucination_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader = hallucination_prompt | structured_hallucination_llm_grader


### --- Answer Grader --- ###
# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader = answer_prompt | structured_llm_answer_grader


def grade_generation_vs_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("###---CHECK HALLUCINATIONS---###")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("###---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---###")
        # Check question-answering
        print("###---GRADE GENERATION vs QUESTION---###")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score

        if grade == "yes":
            print("###---DECISION: GENERATION ADDRESSES QUESTION---###")
            return "useful"
        elif grade == "no" and state["iteration"] < 4:
            print("###---DECISION: GENERATION DOES NOT ADDRESS QUESTION, TRANSFORM QUESTION---###")
            return "not useful"
        else:
            print("###---DECISION: GENERATION DOES NOT ADDRESS QUESTION---###")
            return "stop"
        
    else:
        if state["iteration"] >= 4:
            print("###---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, AND MAX REVISION COUNT REACHED---###")
            return "stop"
        
        else:
            print("###---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---###")
            return "not supported"
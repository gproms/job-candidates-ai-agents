from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import List, Dict
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Prompts for the agents
CV_PROMPT = """You are an expert in parsing CVs. Extract the following sections from the provided CV: Experience, Education, Skills.
Return the extracted information as a JSON object with keys: Experience, Education, and Skills."""
LINKEDIN_PROMPT = """You are an expert in parsing LinkedIn profiles. Extract the following details: Current Role, Connections, Endorsements, and Skills.
Return the extracted information as a JSON object with keys: Experience, Education, and Skills."""
INTERVIEW_PROMPT = """You are an expert in summarizing interviews. Extract the key takeaways related to the candidate's skills, experiences, and motivations.
Return the extracted information as a JSON object with keys: Experience, Education, and Skills."""
SYNTHESIS_PROMPT = """You are tasked with synthesizing structured profiles from multiple data sources. Combine the CV, LinkedIn, and interview data into a single, comprehensive profile.
Return the extracted information as a JSON object with keys: Experience, Education, and Skills."""

# Define the agent state
class AgentState(dict):
    cv: str
    linkedin: str
    interview: str
    cv_data: Dict[str, str]
    linkedin_data: Dict[str, str]
    interview_data: str
    profile: Dict[str, str]

# Define functions for each node
def cv_parser_node(state: AgentState):
    messages = [
        SystemMessage(content=CV_PROMPT),
        HumanMessage(content=state['cv'])
    ]
    response = model.invoke(messages)

    try:
        # Expect the response to be JSON-formatted
        parsed_data = json.loads(response.content)
        return {"cv_data": parsed_data}
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from CV parser response: {response.content}") from e


def linkedin_parser_node(state: AgentState):
    messages = [
        SystemMessage(content=LINKEDIN_PROMPT),
        HumanMessage(content=state['linkedin'])
    ]
    response = model.invoke(messages)

    try:
        parsed_data = json.loads(response.content)
        return {"linkedin_data": parsed_data}
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LinkedIn parser response: {response.content}") from e


def interview_summarizer_node(state: AgentState):
    messages = [
        SystemMessage(content=INTERVIEW_PROMPT),
        HumanMessage(content=state['interview'])
    ]
    response = model.invoke(messages)
    return {"interview_data": response.content}


def synthesis_node(state: AgentState):
    messages = [
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(
            content=f"CV Data:\n{state['cv_data']}\n\nLinkedIn Data:\n{state['linkedin_data']}\n\nInterview Data:\n{state['interview_data']}")
    ]
    response = model.invoke(messages)

    try:
        parsed_data = json.loads(response.content)
        return {"profile": parsed_data}
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from synthesis response: {response.content}") from e


# Build the state graph
builder = StateGraph(AgentState)

builder.add_node("cv_parser", cv_parser_node)
builder.add_node("linkedin_parser", linkedin_parser_node)
builder.add_node("interview_summarizer", interview_summarizer_node)
builder.add_node("synthesis", synthesis_node)

builder.set_entry_point("cv_parser")

builder.add_edge("cv_parser", "linkedin_parser")
builder.add_edge("linkedin_parser", "interview_summarizer")
builder.add_edge("interview_summarizer", "synthesis")
# builder.add_edge("synthesis", END)

if __name__ == "__main__":
    # Compile the graph
    graph = builder.compile()

    # Visualize the graph
    from IPython.display import Image

    Image(graph.get_graph().draw_png())

    # Sample execution
    thread = {"configurable": {"thread_id": "1"}}
    for state in graph.stream({
        "cv": "John Doe\nExperience: 5 years in Software Development\nEducation: BSc in Computer Science\nSkills: Python, Java, SQL",
        "linkedin": "John Doe\nCurrent Role: Senior Developer\nConnections: 500+\nEndorsements: Python, Java",
        "interview": "John highlighted his ability to work in teams, solve complex problems, and his passion for learning new technologies.",
    }, thread):
        print(state)

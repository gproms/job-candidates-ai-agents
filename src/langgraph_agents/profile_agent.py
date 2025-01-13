from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import Dict
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
    parsed_cvs = {}
    for file_name, cv_content in state['cv_data'].items():
        messages = [
            SystemMessage(content=CV_PROMPT),
            HumanMessage(content=cv_content)
        ]
        response = model.invoke(messages)
        try:
            # Expect the response to be JSON-formatted
            parsed_data = json.loads(response.content)
            candidate_name = os.path.splitext(file_name)[0]  # Extract candidate name from file name
            parsed_cvs[candidate_name] = parsed_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from CV parser response for {file_name}: {response.content}") from e
    return {"cv_data": parsed_cvs}

def linkedin_parser_node(state: AgentState):
    parsed_linkedin_profiles = {}
    for profile in state['linkedin_data']:
        messages = [
            SystemMessage(content=LINKEDIN_PROMPT),
            HumanMessage(content=json.dumps(profile))  # Pass profile as JSON string
        ]
        response = model.invoke(messages)
        try:
            parsed_data = json.loads(response.content)
            candidate_name = profile['name']  # Assuming 'name' is the identifier
            parsed_linkedin_profiles[candidate_name] = parsed_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LinkedIn parser response for {profile['name']}: {response.content}") from e
    return {"linkedin_data": parsed_linkedin_profiles}

def interview_summarizer_node(state: AgentState):
    summarized_interviews = {}
    for file_name, interview_content in state['interview_data'].items():
        messages = [
            SystemMessage(content=INTERVIEW_PROMPT),
            HumanMessage(content=interview_content)
        ]
        response = model.invoke(messages)
        try:
            # Parse the response content
            parsed_data = json.loads(response.content)
            candidate_name = os.path.splitext(file_name)[0]  # Extract candidate name from the file name
            summarized_interviews[candidate_name] = parsed_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from interview summarizer response for {file_name}: {response.content}") from e
    return {"interview_data": summarized_interviews}


def synthesis_node(state: AgentState):
    print(f"State Before Synthesis Node: {state}")  # Debugging state input

    messages = [
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(
            content=f"CV Data:\n{state['cv_data']}\n\nLinkedIn Data:\n{state['linkedin_data']}\n\nInterview Data:\n{state['interview_data']}")
    ]
    response = model.invoke(messages)
    try:
        parsed_data = json.loads(response.content)
        state['profiles'] = parsed_data  # Update the state with synthesized profiles
        print(f"Synthesized Profiles in synthesis_node: {state['profiles']}")  # Debugging synthesized profiles
        print(f"State After Updating Profiles in synthesis_node: {state}")  # Debugging updated state
        return {"profiles": parsed_data}  # Return the updated portion of the state
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from synthesis response: {response.content}") from e

# Function to create the profile graph
def create_profile_graph() -> StateGraph:
    # Build the state graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("cv_parser", cv_parser_node)
    builder.add_node("linkedin_parser", linkedin_parser_node)
    builder.add_node("interview_summarizer", interview_summarizer_node)
    builder.add_node("synthesis", synthesis_node)

    # Define edges
    builder.set_entry_point("cv_parser")
    builder.add_edge("cv_parser", "linkedin_parser")
    builder.add_edge("linkedin_parser", "interview_summarizer")
    builder.add_edge("interview_summarizer", "synthesis")
    builder.add_edge("synthesis", END)  # Add an END state here

    return builder

if __name__ == "__main__":
    # For testing the graph, uncomment the following block
    print("Testing Profile Agent...")
    profile_graph = create_profile_graph()
    compiled_graph = profile_graph.compile()

    # Sample execution
    initial_state = {
        "cv": "John Doe\nExperience: 5 years in Software Development\nEducation: BSc in Computer Science\nSkills: Python, Java, SQL",
        "linkedin": "John Doe\nCurrent Role: Senior Developer\nConnections: 500+\nEndorsements: Python, Java",
        "interview": "John highlighted his ability to work in teams, solve complex problems, and his passion for learning new technologies.",
    }
    thread = {"configurable": {"thread_id": "1"}}
    for state in compiled_graph.stream(initial_state, thread):
        print(state)

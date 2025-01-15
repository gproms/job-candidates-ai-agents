from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import Dict, TypedDict
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the agent state
class AgentState(TypedDict):
    cv_data: Dict[str, Dict]
    linkedin_data: Dict[str, Dict]
    interview_data: Dict[str, Dict]
    profiles: Dict[str, Dict]

# Define functions for each node
def cv_parser_node(state: AgentState):
    parsed_cvs = {}
    for file_name, cv_content in state["cv_data"].items():
        # Serialize the CV content to a string
        serialized_content = json.dumps(cv_content)  # Convert the dictionary to a JSON string

        messages = [
            SystemMessage(content=CV_PROMPT),
            HumanMessage(content=serialized_content),  # Use the serialized string
        ]
        response = model.invoke(messages)
        try:
            # Expect the response to be JSON-formatted
            parsed_data = json.loads(response.content)
            candidate_name = os.path.splitext(file_name)[0]  # Extract candidate name from file name
            parsed_cvs[candidate_name] = parsed_data
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from CV parser response for {file_name}: {response.content}"
            ) from e
    return {"cv_data": parsed_cvs}


def linkedin_parser_node(state: AgentState):
    state["linkedin_data"] = {
        "john_doe": {
            "Experience": {"Current Role": "Senior Developer"},
            "Skills": ["Python", "Java"]
        },
        "jane_smith": {
            "Experience": {"Current Role": "Data Analyst"},
            "Skills": ["Python", "R"]
        }
    }
    return {"linkedin_data": state["linkedin_data"]}

def interview_summarizer_node(state: AgentState):
    state["interview_data"] = {
        "john_doe_interview": {
            "Skills": {"Problem Solving": "Strong problem-solving skills"}
        },
        "jane_smith_interview": {
            "Skills": {"Data Visualization": "Proficient in visualization tools"}
        }
    }
    return {"interview_data": state["interview_data"]}

def synthesis_node(state: AgentState):
    profiles = {}
    cv_data = state.get("cv_data", {})
    linkedin_data = state.get("linkedin_data", {})
    interview_data = state.get("interview_data", {})

    for name, cv in cv_data.items():
        formatted_name = name.replace("_", " ").title()
        profiles[formatted_name] = {
            "Experience": linkedin_data.get(name, {}).get("Experience", {}),
            "Skills": list(set(cv["Skills"] + linkedin_data.get(name, {}).get("Skills", []))),
            "Interview Insights": interview_data.get(f"{name}_interview", {}).get("Skills", {})
        }
    state["profiles"] = profiles
    return {"profiles": profiles}

# Create the profile graph
def create_profile_graph():
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
    builder.add_edge("synthesis", END)

    return builder

# Main execution
if __name__ == "__main__":
    print("Executing Profile Agent Graph...")
    profile_graph = create_profile_graph()
    compiled_graph = profile_graph.compile()

    initial_state = AgentState(
        cv_data={},
        linkedin_data={},
        interview_data={},
        profiles={}
    )

    thread = {"configurable": {"thread_id": "1"}}
    for state in compiled_graph.stream(initial_state, thread):
        print("Current State:", state)

    final_state = compiled_graph.invoke(initial_state)
    print("Final State:", final_state)
    print("Profiles Created:", final_state["profiles"])

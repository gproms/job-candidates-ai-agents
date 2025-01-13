from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from utils.file_loader import load_json, load_files_from_directory
from utils.preprocessor import extract_name_from_file_name


# Define State Schema
class ProfileState(BaseModel):
    linkedin_data: list
    cv_data: dict
    interview_data: dict
    profiles: dict = {}


# Node Functions
def parse_linkedin(state: ProfileState) -> ProfileState:
    linkedin_profiles = state.linkedin_data
    profiles = {profile["name"]: profile for profile in linkedin_profiles}
    state.profiles = profiles
    return state


def parse_cvs(state: ProfileState) -> ProfileState:
    cvs = state.cv_data
    profiles = state.profiles

    for file_name, content in cvs.items():
        candidate_name = extract_name_from_file_name(file_name)
        if candidate_name in profiles:
            profiles[candidate_name]["cv_summary"] = content
    state.profiles = profiles
    return state


def parse_interviews(state: ProfileState) -> ProfileState:
    interviews = state.interview_data
    profiles = state.profiles

    for file_name, content in interviews.items():
        candidate_name = extract_name_from_file_name(file_name)
        if candidate_name in profiles:
            profiles[candidate_name]["interview_notes"] = content
    state.profiles = profiles
    return state


# Create the Profile Graph
def create_profile_graph() -> StateGraph:
    graph = StateGraph(state_schema=ProfileState)

    # Add nodes
    graph.add_node("ParseLinkedIn", parse_linkedin)
    graph.add_node("ParseCVs", parse_cvs)
    graph.add_node("ParseInterviews", parse_interviews)

    # Define edges
    graph.add_edge(START, "ParseLinkedIn")
    graph.add_edge("ParseLinkedIn", "ParseCVs")
    graph.add_edge("ParseCVs", "ParseInterviews")
    graph.add_edge("ParseInterviews", END)

    return graph

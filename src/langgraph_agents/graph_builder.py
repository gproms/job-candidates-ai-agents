# graph_builder.py

from langgraph.graph import StateGraph
from langgraph_agents.nodes import cv_parser_node, linkedin_parser_node, interview_summarizer_node, synthesis_node

# Function to create the profile graph
def create_profile_graph() -> StateGraph:
    builder = StateGraph(dict)

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

    return builder

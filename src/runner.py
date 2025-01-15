import os
import json
from utils.file_loader import load_text_files, load_json
from langgraph_agents.nodes import cv_parser_node, linkedin_parser_node, interview_summarizer_node, synthesize_profiles
from src import DATA_DIR

def export_to_json(data, filename="synthesized_profiles.json"):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=2)
    print(f"Data exported to {filename}")

# Load data
cv_data = load_text_files(os.path.join(DATA_DIR, "cvs"))
interview_data = load_text_files(os.path.join(DATA_DIR, "interviews"))
linkedin_data = load_json(os.path.join(DATA_DIR, "linkedin_profiles.json"))

state = {
    "cv_data": cv_data,
    "linkedin_data": linkedin_data,
    "interview_data": interview_data,
}

# Run nodes
print("\n===== Running CV Parser Node =====")
cv_results = cv_parser_node(state)
assert cv_results["cv_data"] is not None, "CV parsing failed!"

print("\n===== Running LinkedIn Parser Node =====")
linkedin_results = linkedin_parser_node(state)
assert linkedin_results["linkedin_data"] is not None, "LinkedIn parsing failed!"

print("\n===== Running Interview Summarizer Node =====")
interview_results = interview_summarizer_node(state)
assert interview_results["interview_data"] is not None, "Interview summarization failed!"

print("\n===== Running Synthesis Node =====")
synthesized_profiles = synthesize_profiles(
    cv_results["cv_data"],
    linkedin_results["linkedin_data"],
    interview_results["interview_data"]
)

export_to_json(synthesized_profiles, "synthesized_profiles.json")

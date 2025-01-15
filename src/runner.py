import os
import json
from utils.file_loader import load_text_files, load_json
from langgraph_agents.profile_agent import load_or_generate_profiles
from langgraph_agents.search_agent import search_candidates
# from langgraph_agents.nodes import cv_parser_node, linkedin_parser_node, interview_summarizer_node, synthesize_profiles
from src import DATA_DIR

def export_to_json(data, filename="search_results.json"):
    """
    Export data to a JSON file.
    """
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=2)
    print(f"Data exported to {filename}")


# Load data
cv_data = load_text_files(os.path.join(DATA_DIR, "cvs"))
interview_data = load_text_files(os.path.join(DATA_DIR, "interviews"))
linkedin_data = load_json(os.path.join(DATA_DIR, "linkedin_profiles.json"))


# state = {
#     "cv_data": cv_data,
#     "linkedin_data": linkedin_data,
#     "interview_data": interview_data,
# }
#
# # Run nodes
# print("\n===== Running CV Parser Node =====")
# cv_results = cv_parser_node(state)
# assert cv_results["cv_data"] is not None, "CV parsing failed!"
#
# print("\n===== Running LinkedIn Parser Node =====")
# linkedin_results = linkedin_parser_node(state)
# assert linkedin_results["linkedin_data"] is not None, "LinkedIn parsing failed!"
#
# print("\n===== Running Interview Summarizer Node =====")
# interview_results = interview_summarizer_node(state)
# assert interview_results["interview_data"] is not None, "Interview summarization failed!"


# Step 1: Generate or Load Profiles
print("\n===== Running Synthesis Node =====")
# profiles_candidates = synthesize_profiles(
#     cv_results["cv_data"],
#     linkedin_results["linkedin_data"],
#     interview_results["interview_data"]
# )
# export_to_json(profiles_candidates, "profiles_candidates.json")

profiles_candidates = load_or_generate_profiles(
    cv_data, linkedin_data, interview_data, regenerate=False
)

# Step 2: Search Candidates
print("\n===== Running Search =====")
query = "Find candidates with up to 4 years of experience in Python."
results = search_candidates(query, profiles_candidates, top_n=10)

# Step 3: Export Results
export_to_json(results, "search_results.json")

# Display Results
print("Search Results:")
for candidate_id, profile in results.items():
    print(f"Candidate ID: {candidate_id}")
    print(f"Summary: {profile['Summary']}")
    print(json.dumps(profile, indent=2))

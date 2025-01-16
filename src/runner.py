import os
import json
from langgraph_agents.query_agent import interpret_and_filter_profiles, validate_and_convert_profiles
from utils.file_loader import export_to_json
from src import DATA_DIR

# Paths
PROFILES_JSON_PATH = os.path.join(DATA_DIR, "profiles_candidates.json")
QUERY_RESULTS_PATH = os.path.join(DATA_DIR, "query_results.json")

# Load profiles
print("Loading profiles from cached JSON...")
with open(PROFILES_JSON_PATH, "r") as f:
    profiles_candidates = validate_and_convert_profiles(json.load(f))

# Run query agent
print("\n===== Running Query Agent =====")
user_query = "Find candidates having either Java skills less than 5 years work experience. Also find any candidates having Scrum and Agile"
filtered_results = interpret_and_filter_profiles(user_query, profiles_candidates)

if filtered_results:
    print("\n=== Query Results ===")
    print(json.dumps(filtered_results, indent=2))
else:
    print("\n=== No Results Found ===")

export_to_json(filtered_results, QUERY_RESULTS_PATH)

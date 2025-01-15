import os
import json
from langgraph_agents.query_agent import interpret_and_filter_profiles, validate_and_convert_profiles
from utils.file_loader import export_to_json
from src import DATA_DIR

# Paths
PROFILES_JSON_PATH = os.path.join(DATA_DIR, "profiles_candidates.json")
QUERY_RESULTS_PATH = os.path.join(DATA_DIR, "query_results.json")

try:
    # Load profiles
    print("Loading profiles from cached JSON...")
    with open(PROFILES_JSON_PATH, "r") as f:
        profiles_candidates = validate_and_convert_profiles(json.load(f))
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading profiles: {e}")
    profiles_candidates = {}

# Run query agent
if profiles_candidates:
    print("\n===== Running Query Agent =====")
    user_query = "Find 3 candidates with 1+ years of experience."
    filtered_results = interpret_and_filter_profiles(user_query, profiles_candidates)

    if filtered_results:
        print("\n=== Query Results ===")
        print(json.dumps(filtered_results, indent=2))
    else:
        print("\n=== No Results Found ===")

    export_to_json(filtered_results, QUERY_RESULTS_PATH)
else:
    print("No profiles available for querying.")

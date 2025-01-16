import os
import json
from langgraph_agents.embeddings_agent import (
    preprocess_and_embed_profiles,
    embed_query,
    search_profiles,
)
from utils.file_loader import load_json, export_to_json
from src import DATA_DIR

# Paths
PROFILES_JSON_PATH = os.path.join(DATA_DIR, "profiles_candidates.json")
QUERY_RESULTS_PATH = os.path.join(DATA_DIR, "query_results_embeddings.json")

def main():
    # Load profiles
    print("Loading profiles from cached JSON...")
    profiles = load_json(PROFILES_JSON_PATH)

    # Preprocess and embed profiles
    print("Generating profile embeddings...")
    profile_embeddings = preprocess_and_embed_profiles(profiles)

    # User query
    user_query = input("Enter your query: ")
    print(f"Processing query: {user_query}")

    # Embed query
    query_embedding = embed_query(user_query)

    # Search profiles
    print("Searching profiles...")
    top_results = search_profiles(query_embedding, profile_embeddings, top_k=1)

    # Display results
    if top_results:
        print("\n=== Query Results ===")
        print(json.dumps(top_results, indent=2))
    else:
        print("\n=== No Results Found ===")

    # Save results
    export_to_json(top_results, QUERY_RESULTS_PATH)

if __name__ == "__main__":
    main()

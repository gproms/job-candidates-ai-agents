import os
import json
from langgraph_agents.embeddings_agent import EmbeddingsAgent
from utils.file_loader import export_to_json
from src import DATA_DIR
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
PROFILES_JSON_PATH = os.path.join(DATA_DIR, "profiles_candidates.json")
QUERY_RESULTS_PATH = os.path.join(DATA_DIR, "query_results_embeddings.json")

# Load profiles
print("Loading profiles from cached JSON...")
with open(PROFILES_JSON_PATH, "r") as f:
    profiles_candidates = json.load(f)

# Initialize the EmbeddingsAgent
agent = EmbeddingsAgent()

# Define the query
# user_query = "Find candidates with 4 years work experience"
user_query = "Find a candidate with 4 years work experience"

# Run the embedding search
print("\n===== Running Embeddings Search =====")
top_k = 1  # Number of results to retrieve
filtered_results = agent.search(user_query, profiles_candidates, top_k=top_k)

# Output results
if filtered_results:
    print("\n=== Query Results ===")
    print(json.dumps(filtered_results, indent=2))
else:
    print("\n=== No Results Found ===")

# Export results
export_to_json(filtered_results, QUERY_RESULTS_PATH)
print(f"Results saved to {QUERY_RESULTS_PATH}")

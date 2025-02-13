import os
import json
import logging
from utils.file_loader import load_text_files, load_json
from langgraph_agents.nodes import cv_parser_node, linkedin_parser_node, interview_summarizer_node, synthesize_profiles
from langgraph_agents.profile_agent import query_profiles, refine_profiles
from src import DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def export_to_json(data, filename="synthesized_profiles.json"):
    """
    Export data to a JSON file.
    """
    try:
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=2)
        logging.info(f"Data exported to {filename}")
    except Exception as e:
        logging.error(f"Failed to export data: {e}")

def load_data():
    try:
        cv_data = load_text_files(os.path.join(DATA_DIR, "cvs"))
        interview_data = load_text_files(os.path.join(DATA_DIR, "interviews"))
        linkedin_data = load_json(os.path.join(DATA_DIR, "linkedin_profiles.json"))
        return {
            "cv_data": cv_data,
            "linkedin_data": linkedin_data,
            "interview_data": interview_data,
        }
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def main():
    state = load_data()
    if not state:
        return

    # Run nodes
    logging.info("Running CV Parser Node")
    cv_results = cv_parser_node(state)
    assert cv_results["cv_data"] is not None, "CV parsing failed!"

    logging.info("Running LinkedIn Parser Node")
    linkedin_results = linkedin_parser_node(state)
    assert linkedin_results["linkedin_data"] is not None, "LinkedIn parsing failed!"

    logging.info("Running Interview Summarizer Node")
    interview_results = interview_summarizer_node(state)
    assert interview_results["interview_data"] is not None, "Interview summarization failed!"

    logging.info("Running Synthesis Node")
    profiles_candidates = synthesize_profiles(
        cv_results["cv_data"],
        linkedin_results["linkedin_data"],
        interview_results["interview_data"]
    )

    export_to_json(profiles_candidates, f"{DATA_DIR}/profiles_candidates2.json")

    # Step 2: Search Candidates
    logging.info("Running Search")
    query = "Find one candidate with a PhD"
    results = query_profiles(query, profiles_candidates)

    # Display Results
    logging.info("Search Results:")
    for candidate_id, profile in results.items():
        logging.info(f"Candidate ID: {candidate_id}")
        logging.info(f"Summary: {profile['Summary']}")
        logging.info(json.dumps(profile, indent=2))

if __name__ == "__main__":
    main()


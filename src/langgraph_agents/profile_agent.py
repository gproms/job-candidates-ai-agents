import json
import logging
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph_agents.prompts import SYNTHESIS_PROMPT
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

DEFAULT_JSON_PATH = "profiles_candidates.json"

def synthesize_profiles(cv_data, linkedin_data, interview_data):
    """
    Combines CVs, LinkedIn profiles, and interviews into unified profiles.
    """
    logging.info("Starting Profile Synthesis.")
    synthesized_profiles = {}

    for candidate_id, cv_entry in cv_data.items():
        linkedin_entry = linkedin_data.get(candidate_id, {})
        interview_entry = interview_data.get(candidate_id, {})

        content = {
            "cv": cv_entry,
            "linkedin": linkedin_entry,
            "interview": interview_entry
        }

        messages = [
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(content=json.dumps(content))
        ]

        response = llm.invoke(messages)
        try:
            synthesized_profiles[candidate_id] = json.loads(response.content)
        except json.JSONDecodeError:
            logging.error(f"Failed to synthesize profile for {candidate_id}. Response: {response.content}")
            synthesized_profiles[candidate_id] = {"error": "Failed to synthesize profile"}

    logging.info("Profile Synthesis complete.")
    return synthesized_profiles


def load_or_generate_profiles(cv_data, linkedin_data, interview_data, regenerate=False, json_path=DEFAULT_JSON_PATH):
    """
    Loads profiles from a JSON file or regenerates them if specified.
    """
    if not regenerate and os.path.exists(json_path):
        logging.info(f"Loading profiles from {json_path}")
        with open(json_path, "r") as f:
            return json.load(f)

    logging.info("Generating new profiles...")
    profiles = synthesize_profiles(cv_data, linkedin_data, interview_data)
    with open(json_path, "w") as f:
        json.dump(profiles, f, indent=2)
    logging.info(f"Profiles saved to {json_path}")
    return profiles

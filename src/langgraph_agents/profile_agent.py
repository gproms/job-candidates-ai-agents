import json
import logging
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

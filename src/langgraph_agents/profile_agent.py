import json
import logging
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def query_profiles(query: str, profiles: Dict) -> Dict:
    """
    Query the synthesized profiles based on a user query.
    """
    logging.info(f"Querying profiles with query: {query}")
    filtered_profiles = {}

    for candidate_id, profile in profiles.items():
        # Check if the profile matches the query
        messages = [
            SystemMessage(content="Answer True if the profile matches the query, else False."),
            HumanMessage(content=f"Query: {query}\nProfile: {json.dumps(profile)}")
        ]
        response = llm.invoke(messages)
        is_match = response.content.strip().lower() == "true"
        if is_match:
            filtered_profiles[candidate_id] = profile

    logging.info(f"Found {len(filtered_profiles)} matching profiles.")
    return filtered_profiles

def refine_profiles(profiles: Dict) -> Dict:
    """
    Refine the synthesized profiles (e.g., deduplicate, normalize).
    """
    logging.info("Refining profiles.")
    refined_profiles = {}

    for candidate_id, profile in profiles.items():
        # Deduplicate skills
        if "Skills" in profile:
            profile["Skills"] = list(set(profile["Skills"]))
        # Normalize experience and education entries
        if "Experience" in profile:
            profile["Experience"] = [dict(t) for t in {tuple(d.items()) for d in profile["Experience"]}]
        if "Education" in profile:
            profile["Education"] = [dict(t) for t in {tuple(d.items()) for d in profile["Education"]}]
        refined_profiles[candidate_id] = profile

    logging.info("Profiles refined.")
    return refined_profiles


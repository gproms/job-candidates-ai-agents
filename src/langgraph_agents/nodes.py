import json
import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from langgraph_agents.prompts import CV_PROMPT, LINKEDIN_PROMPT, INTERVIEW_PROMPT, SYNTHESIS_PROMPT

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()

# Initialize OpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# CV Parser Node
def cv_parser_node(state):
    logging.info(f"Starting CV Parser with {len(state.get('cv_data', {}))} entries.")
    parsed_cvs = {}

    for key, content in state.get("cv_data", {}).items():
        key = key.replace("cv_", "candidate_").split(".txt")[0]
        messages = [
            SystemMessage(content=CV_PROMPT),
            HumanMessage(content=json.dumps(content))
        ]
        response = model.invoke(messages)
        try:
            parsed_cvs[key] = json.loads(response.content)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse CV data for {key}. Response: {response.content}")
            parsed_cvs[key] = {"error": "Failed to parse on cv_parser_node"}

    state["cv_data"] = parsed_cvs
    logging.info("CV Parsing complete.")
    return {"cv_data": parsed_cvs}

# LinkedIn Parser Node
def linkedin_parser_node(state):
    linkedin_data = state.get("linkedin_data", [])
    logging.info(f"Starting LinkedIn Parser with {len(linkedin_data)} profiles.")
    parsed_linkedin_profiles = {}

    for i, profile in enumerate(linkedin_data):
        key = f"candidate_{i + 1}"
        profile_name = profile.get("name", "Unknown")
        messages = [
            SystemMessage(content=LINKEDIN_PROMPT),
            HumanMessage(content=json.dumps(profile))
        ]
        response = model.invoke(messages)
        try:
            parsed_linkedin_profiles[key] = json.loads(response.content)
            parsed_linkedin_profiles[key]["name"] = profile_name
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LinkedIn data for {profile_name}. Response: {response.content}")
            parsed_linkedin_profiles[key] = {"error on linkedin_parser_node": "Failed to parse"}

    state["linkedin_data"] = parsed_linkedin_profiles
    logging.info("LinkedIn Parsing complete.")
    return {"linkedin_data": parsed_linkedin_profiles}

# Interview Summarizer Node
def interview_summarizer_node(state):
    interview_data = state.get("interview_data", {})
    logging.info(f"Starting Interview Summarizer with {len(interview_data)} entries.")
    summarized_interviews = {}

    for key, content in interview_data.items():
        key = key.replace("interview_", "candidate_").split(".txt")[0]
        messages = [
            SystemMessage(content=INTERVIEW_PROMPT),
            HumanMessage(content=json.dumps(content))
        ]
        response = model.invoke(messages)
        try:
            summarized_interviews[key] = json.loads(response.content)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse interview data for {key}. Response: {response.content}")
            summarized_interviews[key] = {"error on interview_summarizer_node": "Failed to parse"}

    state["interview_data"] = summarized_interviews
    logging.info("Interview Summarization complete.")
    return {"interview_data": summarized_interviews}

def normalize_skills(skills: List[str]) -> List[str]:
    """
    Normalize skill names to ensure consistency.
    """
    skill_mapping = {
        "ai": "Artificial Intelligence",
        "ml": "Machine Learning",
        "dl": "Deep Learning",
        "nlp": "Natural Language Processing",
        "ux": "User Experience",
        "ui": "User Interface",
    }
    normalized_skills = set()
    for skill in skills:
        skill = skill.strip().lower()
        if skill in skill_mapping:
            normalized_skills.add(skill_mapping[skill])
        else:
            normalized_skills.add(skill.title())  # Capitalize first letter
    return sorted(normalized_skills)

def add_source_to_field(field_data, cv_entry, linkedin_entry, interview_entry, field_name):
    """
    Add source tracking to a field (e.g., Experience or Education).
    """
    updated_field = []
    for entry in field_data:
        sources = []
        # Check if the entry exists in the CV data
        cv_items = cv_entry.get(field_name, [])
        if isinstance(cv_items, list):
            for cv_item in cv_items:
                if isinstance(cv_item, dict) and isinstance(entry, dict) and entry.items() <= cv_item.items():
                    sources.append("CV")
                elif isinstance(cv_item, str) and isinstance(entry, dict) and str(entry) == cv_item:
                    sources.append("CV")
                elif isinstance(cv_item, str) and isinstance(entry, str) and entry == cv_item:
                    sources.append("CV")
        # Check if the entry exists in the LinkedIn data
        linkedin_items = linkedin_entry.get(field_name, [])
        if isinstance(linkedin_items, list):
            for linkedin_item in linkedin_items:
                if isinstance(linkedin_item, dict) and isinstance(entry, dict) and entry.items() <= linkedin_item.items():
                    sources.append("LinkedIn")
                elif isinstance(linkedin_item, str) and isinstance(entry, dict) and str(entry) == linkedin_item:
                    sources.append("LinkedIn")
                elif isinstance(linkedin_item, str) and isinstance(entry, str) and entry == linkedin_item:
                    sources.append("LinkedIn")
        # Check if the entry exists in the Interview data
        interview_items = interview_entry.get(field_name, [])
        if isinstance(interview_items, list):
            for interview_item in interview_items:
                if isinstance(interview_item, dict) and isinstance(entry, dict) and entry.items() <= interview_item.items():
                    sources.append("Interview")
                elif isinstance(interview_item, str) and isinstance(entry, dict) and str(entry) == interview_item:
                    sources.append("Interview")
                elif isinstance(interview_item, str) and isinstance(entry, str) and entry == interview_item:
                    sources.append("Interview")
        entry["source"] = sources
        updated_field.append(entry)
    return updated_field

def add_source_to_skills(skills, cv_entry, linkedin_entry, interview_entry):
    """
    Add source tracking to skills.
    """
    updated_skills = []
    for skill in skills:
        sources = []
        if skill in cv_entry.get("Skills", []):
            sources.append("CV")
        if skill in linkedin_entry.get("Skills", []):
            sources.append("LinkedIn")
        if skill in interview_entry.get("Skills", []):
            sources.append("Interview")
        updated_skills.append({"skill": skill, "source": sources})
    return updated_skills

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
        response = model.invoke(messages)
        try:
            profile = json.loads(response.content)
            # Normalize skills
            if "Skills" in profile:
                profile["Skills"] = normalize_skills(profile["Skills"])
            # Ensure Education is a list of dictionaries
            if "Education" in profile and isinstance(profile["Education"], dict):
                profile["Education"] = [profile["Education"]]
            # Add source tracking for each field
            profile["Experience"] = add_source_to_field(profile.get("Experience", []), cv_entry, linkedin_entry, interview_entry, "Experience")
            profile["Education"] = add_source_to_field(profile.get("Education", []), cv_entry, linkedin_entry, interview_entry, "Education")
            profile["Skills"] = add_source_to_skills(profile.get("Skills", []), cv_entry, linkedin_entry, interview_entry)
            synthesized_profiles[candidate_id] = profile
        except json.JSONDecodeError:
            logging.error(f"Failed to combine the 3 sources for {candidate_id}. Response: {response.content}")
            synthesized_profiles[candidate_id] = {"error": "Failed to combine the 3 sources on synthesize_profiles"}

    logging.info("ProfilesSynthesis of 3 sources complete.")
    return synthesized_profiles


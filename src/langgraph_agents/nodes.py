import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from langgraph_agents.prompts import CV_PROMPT, LINKEDIN_PROMPT, INTERVIEW_PROMPT, SYNTHESIS_PROMPT

logging.basicConfig(
    # filename="debug_log.txt",
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
            # parsed_data = json.loads(response.content)
            # summarized_interviews[key] = {
            #     "name": parsed_data.get("name", "Unknown Candidate"),
            #     "Experience": parsed_data.get("Experience", "No Experience"),
            #     "Education": parsed_data.get("Education", {}),
            #     "Skills": parsed_data.get("Skills", [])
            # }
        except json.JSONDecodeError:
            logging.error(f"Failed to parse interview data for {key}. Response: {response.content}")
            # summarized_interviews[key] = {
            #     "name": "Unknown Candidate on interview_summarizer_node",
            #     "Experience": ["Parsing failed on interview_summarizer_node."],
            #     "Education": {"error": "Parsing failed on interview_summarizer_node."},
            #     "Skills": []
            # }
            summarized_interviews[key] = {"error on interview_summarizer_node": "Failed to parse"}

    state["interview_data"] = summarized_interviews
    logging.info("Interview Summarization complete.")
    return {"interview_data": summarized_interviews}



# Synthesize Profiles Node
def synthesize_profiles(cv_data, linkedin_data, interview_data):
    logging.info("Starting Profile Synthesis.")
    synthesized_profiles = {}

    # for key, content in state.get("cv_data", {}).items():
    for candidate_id, cv_entry in cv_data.items():
        linkedin_entry = linkedin_data[candidate_id]
        interview_entry = interview_data[candidate_id]

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
            synthesized_profiles[candidate_id] = json.loads(response.content)
        except json.JSONDecodeError:
            logging.error(f"Failed to combine the 3 sources for {candidate_id}. Response: {response.content}")
            synthesized_profiles[candidate_id] = {"error": "Failed to combine the 3 sources on synthesize_profiles"}

    # state["cv_data"] = parsed_cvs
    logging.info("ProfilesSynthesis of 3 sources complete.")
    return synthesized_profiles


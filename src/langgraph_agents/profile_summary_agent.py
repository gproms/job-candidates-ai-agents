import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI LLM
summarizer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def generate_structured_summary(profile: Dict) -> str:
    """
    Generate a concise, structured summary from a candidate profile.
    """
    prompt = f"""
    Summarize this candidate profile into structured text including:
    - Total years of experience
    - Highest degree and field
    - Key skills

    Profile: {profile}
    """
    response = summarizer_llm.invoke([SystemMessage(content=prompt)])
    return response.content.strip()


def generate_query_summary(user_query: str) -> str:
    """
    Summarize a user query into structured text.
    """
    prompt = f"""
    Convert the following user query into a structured summary for matching:

    Query: {user_query}
    """
    response = summarizer_llm.invoke([SystemMessage(content=prompt)])
    return response.content.strip()

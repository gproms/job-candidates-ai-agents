import json
import traceback
from langchain_openai.chat_models.base import ChatOpenAI
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure OPENAI_API_KEY is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set in the .env file or environment.")


def interpret_query(query: str) -> str:
    """
    Uses the LLM to generate Python filter logic for a user query.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    prompt = (
        f"Create a Python function named `filter_func(profile)` that evaluates if a profile matches "
        f"the following query: {query}. The function should return True if the profile matches and "
        f"False otherwise. Do not include extra explanations or formatting."
    )
    return llm.invoke(prompt).strip()



def validate_query_code(code: str) -> str:
    """
    Uses the LLM to validate the generated code and improve it if necessary.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    prompt = (
        f"You are a Python expert. Analyze the following Python code for logic errors, potential corner cases, "
        f"and runtime issues. Provide an improved or fixed version of the code if necessary.\n\nCode:\n{code}"
    )
    response = llm.invoke(prompt)
    print(f"Validated/Improved Code:\n{response}")  # Debugging output
    return response


def execute_filter(profiles: Dict[str, Dict[str, Any]], filter_func_code: str) -> List[Dict[str, Any]]:
    """
    Executes the generated filter function on the profiles and returns the matching profiles.
    """
    # Define the function in the current scope
    try:
        exec(filter_func_code, globals())
        assert 'filter_func' in globals(), "filter_func was not defined in the code."
    except Exception as e:
        print(f"Error defining filter_func:\n{e}")
        return []

    # Apply the filter function to profiles
    results = []
    for profile in profiles.values():
        try:
            if filter_func(profile):
                results.append(profile)
        except Exception as e:
            print(f"Error applying filter to profile {profile['name']}:\n{e}")
    return results


def search_profiles(profiles: Dict[str, Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Main search function that uses AI agents to interpret, validate, and execute a query.
    """
    print(f"Processing Query: {query}")

    # Step 1: Interpret the query into Python filter logic
    filter_func_code = interpret_query(query)

    # Step 2: Validate and refine the generated code
    validated_code = validate_query_code(filter_func_code)

    # Step 3: Execute the filter logic
    results = execute_filter(profiles, validated_code)

    return results

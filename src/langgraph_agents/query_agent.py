import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from typing import Dict
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def validate_and_convert_profiles(profiles: Dict) -> Dict:
    """
    Ensure profiles contain consistent data types, particularly for `duration_years`.
    Converts `duration_years` to float where possible.
    """
    for candidate_id, profile in profiles.items():
        for experience in profile.get("Experience", []):
            if "duration_years" in experience:
                try:
                    experience["duration_years"] = float(experience["duration_years"])
                except (ValueError, TypeError):
                    experience["duration_years"] = 0.0  # Default to 0.0 if conversion fails
    return profiles

def sanitize_generated_code(code: str) -> str:
    """
    Sanitize and validate the generated Python code.
    Ensures proper syntax for dictionary comprehensions.
    """
    try:
        sanitized_code = code.strip()
        if sanitized_code.startswith("{") and sanitized_code.endswith("}"):
            # Ensure it parses as valid Python
            eval(sanitized_code, {"profiles": {}})
        else:
            raise SyntaxError("Generated code is not a valid dictionary comprehension.")
    except SyntaxError as e:
        logger.error(f"Sanitization failed: {e}")
        raise
    return sanitized_code

def interpret_and_filter_profiles(query: str, profiles: Dict) -> Dict:
    """
    Interpret the query and apply a filter to the profiles using generated Python code.
    """
    # Validate and sanitize profiles
    profiles = validate_and_convert_profiles(profiles)

    # Prompt for the agent
    prompt = f"""
    You are an assistant that writes Python functions to filter profiles from a given dataset.
    The dataset is a JSON object where keys are candidate IDs and values are profiles.
    Each profile has fields like "Experience", "Education", "Skills", and "Summary".

    ### Rules for Writing Code:
    1. Always return a Python function named `filter_profiles`.
    2. The function should take one required parameter: `profiles` (the dataset).
    3. Optionally, if the query specifies a maximum number of results, the function should include an additional parameter `max_results` with a default value of `None`. 
       - If `max_results` is provided, limit the results to that number.
       - If `max_results` is `None`, return all matching profiles.
    4. Use proper Python syntax, including indentation.
    5. Use a loop and conditional statements to filter profiles.
    6. Ensure the function returns a dictionary of filtered profiles.

    ### Example Query:
    - Query: "Find one candidate with 4+ years of experience in AI or Python."
    - Function Output:
    def filter_profiles(profiles, max_results=1):
        filtered_profiles = {{}}
        count = 0
        for key, value in profiles.items():
            if any(exp.get("duration_years", 0) >= 4 for exp in value.get("Experience", [])) and (
                "AI" in value.get("Skills", []) or "Python" in value.get("Skills", [])
            ):
                filtered_profiles[key] = value
                count += 1
                if max_results and count >= max_results:
                    break
        return filtered_profiles

    ### Example Query Without a Limit:
    - Query: "Find all candidates with 4 years of experience."
    - Function Output:
    def filter_profiles(profiles, max_results=None):
        filtered_profiles = {{}}
        for key, value in profiles.items():
            if any(exp.get("duration_years", 0) == 4 for exp in value.get("Experience", [])):
                filtered_profiles[key] = value
        return filtered_profiles

    Write Python code to filter profiles based on this query:
    Query: {query}
    Respond with only the code. Ensure the function uses proper indentation and handles the `max_results` parameter as optional.
    """

    # Send prompt to LLM
    response = llm.invoke([SystemMessage(content=prompt)])
    generated_code = response.content.strip()

    # Remove Markdown-like formatting
    if generated_code.startswith("```") and generated_code.endswith("```"):
        generated_code = generated_code.strip("```").strip("python")

    print("\n=== Generated Code ===")
    print(generated_code)

    # Execute the generated function
    try:
        # Compile and execute the function code
        exec(generated_code, globals())
        # Call the function defined in the generated code
        filtered_profiles = filter_profiles(profiles)
    except Exception as e:
        logger.error(f"Failed to execute generated function. Error: {e}")
        return {}

    return filtered_profiles


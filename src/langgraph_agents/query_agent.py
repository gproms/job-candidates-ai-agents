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
    You are a Python assistant that writes code to filter or rank profiles from a given dataset.
    The dataset is a JSON object where keys are candidate IDs and values are profiles.

    The user can ask any query about the dataset, such as:
    - "Find candidates with 4+ years of experience in AI or Python."
    - "Find the candidate with the most years of work experience."
    - "Find all candidates with experience in Agile methodologies."

    ### Instructions:
    1. Write a Python function named `filter_profiles` that processes the dataset to answer the query.
    2. Your function must take one parameter: `profiles` (the dataset).
    3. If applicable, your function can take additional parameters based on the query (e.g., `max_results` or a condition).
    4. Use the fields in the dataset like `Experience`, `Education`, `Skills`, and `Summary`.
    5. Ensure your function handles missing fields gracefully using `.get()`.

    ### Example Input Query:
    - Query: "Find candidates with 4+ years of experience in AI or Python."

    ### Example Output Code:
    def filter_profiles(profiles):
        filtered_profiles = {{}}
        for key, value in profiles.items():
            if any(exp.get("duration_years", 0) >= 4 for exp in value.get("Experience", [])) and (
                "AI" in value.get("Skills", []) or "Python" in value.get("Skills", [])
            ):
                filtered_profiles[key] = value
        return filtered_profiles

    Write a Python function to answer this query:
    Query: {query}
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

